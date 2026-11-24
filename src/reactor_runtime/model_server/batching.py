"""Which waiting requests the model steps together, and when: the batching policy, on its own.

:class:`BatchPolicy` holds no sockets, no threads and no clock of its own: the
model server hands it requests and the time, and it answers with the next batch, the
requests whose timeout passed, and when to look again. That keeps the policy
:class:`~reactor_runtime.model_server.ModelServer` documents testable on its own.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

LIVE_SECONDS = 2.0
"""How recently a session must have opened, reset or sent a request to count
as live: a session with a client steps every frame, while a session runtime
that serves no client sends nothing. Only live sessions are waited for when a
batch fills."""


@dataclass(eq=False)
class Waiting:
    """One session's request, waiting for a step.

    Two requests are the same only if they are the same object: a request's
    payload may hold arrays, which do not compare as a single truth value.

    Attributes:
        connection: Names the session's connection on the model server.
        request_id: The id its answer carries.
        payload: The request.
        arrived: When it arrived, in the model server's monotonic clock.
        deadline: When it expires if it is still waiting, or ``None``.
    """

    connection: int
    request_id: int
    payload: Any
    arrived: float
    deadline: float | None
    order: int = field(default=0)


@dataclass(frozen=True)
class Turn:
    """What the policy decided at one moment.

    Attributes:
        batch: The requests to step now, in arrival order; empty when nothing
            is ready yet.
        expired: Requests whose timeout passed while they waited; they never
            reach a step.
        wake_at: When to ask again if a batch is still filling, or ``None``:
            the earlier of the end of the batch window and the earliest
            deadline of a waiting request.
    """

    batch: list[Waiting]
    expired: list[Waiting]
    wake_at: float | None


class BatchPolicy:
    """The batching rules: one request per session, arrival order, a window to fill a batch.

    * A session has at most one request waiting or running: :meth:`add`
      refuses a second until :meth:`finished` releases the first.
    * Requests are taken in arrival order, up to ``max_batch`` at a time.
    * Once a request waits, the batch may wait up to ``batch_window`` seconds
      after it arrived for others, and is taken early once ``max_batch``
      requests wait or every live session that can send one has one waiting.
      A session is live when it opened, reset, sent a request, or got a
      result or an expiry in the last :data:`LIVE_SECONDS` (:meth:`live`,
      :meth:`finished`): a
      session runtime that serves no client sends nothing, so it holds a
      batch back for at most that long, and a session whose step takes longer
      than that still counts once its result is back.
      A session whose request is running, its future included, cannot send
      another, so it does not count. A model that schedules work across its
      own stages, and returns futures, sets ``batch_window`` to 0 and is
      stepped as soon as a request arrives.
    * A request whose deadline passed while it waited is expired, never stepped.

    Args:
        max_batch: The most requests one step receives, at least 1.
        batch_window: Seconds a batch may wait to fill, at least 0.
    """

    def __init__(self, max_batch: int, batch_window: float) -> None:
        self._max_batch = max_batch
        self._window = batch_window
        self._waiting: list[Waiting] = []
        self._busy: set[int] = set()
        self._order = itertools.count()
        # When each session last sent a request, to tell live sessions apart.
        self._last_sent: dict[int, float] = {}

    def add(self, request: Waiting) -> bool:
        """Queue *request*; return ``False`` if its session already has one waiting or running."""
        if request.connection in self._busy:
            return False
        request.order = next(self._order)
        self._last_sent[request.connection] = request.arrived
        self._busy.add(request.connection)
        self._waiting.append(request)
        return True

    def live(self, connection: int, now: float) -> None:
        """Count *connection* as live from *now*: it opened, or a session started on it."""
        self._last_sent[connection] = now

    def cancel(self, connection: int, request_id: int) -> bool:
        """Withdraw a waiting request; return whether it was still waiting."""
        for index, request in enumerate(self._waiting):
            if request.connection == connection and request.request_id == request_id:
                del self._waiting[index]
                self._busy.discard(connection)
                return True
        return False

    def drop(self, connection: int) -> list[Waiting]:
        """Withdraw every waiting request of a session that left or reset, and return them."""
        dropped = [r for r in self._waiting if r.connection == connection]
        self._waiting = [r for r in self._waiting if r.connection != connection]
        self._busy.discard(connection)
        return dropped

    def finished(self, connections: list[int], now: float) -> None:
        """Release the sessions whose requests a step just answered, live from *now*.

        A session that just got its result sends its next request soon, so
        it counts as live from *now* whatever the step took.
        """
        for connection in connections:
            self._busy.discard(connection)
            self._last_sent[connection] = now

    @property
    def waiting(self) -> int:
        """How many requests wait."""
        return len(self._waiting)

    def take(self, now: float, open_sessions: Collection[int]) -> Turn:
        """Decide what to step at *now*, given the sessions that are open.

        Of the *open_sessions*, the live ones count (see :data:`LIVE_SECONDS`),
        and those whose request is running count as unable to send, so a
        batch is full once every other live one has a request waiting. The
        batch taken is removed from the queue, and its sessions stay busy
        until :meth:`finished`.
        """
        self._last_sent = {c: t for c, t in self._last_sent.items() if c in open_sessions}
        expired = [r for r in self._waiting if r.deadline is not None and r.deadline <= now]
        if expired:
            self._waiting = [r for r in self._waiting if r not in expired]
            # An expired request's session gets its answer now, as from a
            # step, and is as likely to send again.
            for request in expired:
                self._busy.discard(request.connection)
                self._last_sent[request.connection] = now
        if not self._waiting:
            return Turn([], expired, None)
        first = self._waiting[0]
        waiting = {r.connection for r in self._waiting}
        # A session with a request waiting always counts; another counts
        # while it is live and not running a request.
        can_send = waiting | {
            c
            for c, sent in self._last_sent.items()
            if now - sent < LIVE_SECONDS and c not in self._busy
        }
        full = len(self._waiting) >= min(self._max_batch, max(len(can_send), 1))
        until = first.arrived + self._window
        if not full and now < until:
            deadlines = [r.deadline for r in self._waiting if r.deadline is not None]
            # A live session the batch still waits for stops counting when it
            # has been quiet for LIVE_SECONDS: look again then, since the batch
            # may be full without it. Each of these is after now.
            quiet = [self._last_sent[c] + LIVE_SECONDS for c in can_send if c not in waiting]
            return Turn([], expired, min([until, *deadlines, *quiet]))
        batch = self._waiting[: self._max_batch]
        self._waiting = self._waiting[self._max_batch :]
        return Turn(batch, expired, None)
