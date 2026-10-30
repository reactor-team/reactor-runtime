"""Which waiting requests the model steps together, and when: the batching policy, on its own.

:class:`BatchPolicy` holds no sockets, no threads and no clock of its own: the
worker hands it requests and the time, and it answers with the next batch, the
requests whose timeout passed, and when to look again. That keeps the policy
:class:`~reactor_runtime.model_server.ModelServer` documents testable on its own.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class Waiting:
    """One session's request, waiting for a step.

    Two requests are the same only if they are the same object: a request's
    payload may hold arrays, which do not compare as a single truth value.

    Attributes:
        connection: Names the session's connection on the worker.
        request_id: The id its answer carries.
        payload: The request.
        arrived: When it arrived, in the worker's monotonic clock.
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
      requests wait or every open session has one waiting.
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

    def add(self, request: Waiting) -> bool:
        """Queue *request*; return ``False`` if its session already has one waiting or running."""
        if request.connection in self._busy:
            return False
        request.order = next(self._order)
        self._busy.add(request.connection)
        self._waiting.append(request)
        return True

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

    def finished(self, connections: list[int]) -> None:
        """Release the sessions whose requests a step just answered."""
        for connection in connections:
            self._busy.discard(connection)

    @property
    def waiting(self) -> int:
        """How many requests wait."""
        return len(self._waiting)

    def take(self, now: float, open_sessions: int) -> Turn:
        """Decide what to step at *now*, given how many sessions are open.

        The batch taken is removed from the queue, and its sessions stay busy
        until :meth:`finished`.
        """
        expired = [r for r in self._waiting if r.deadline is not None and r.deadline <= now]
        if expired:
            self._waiting = [r for r in self._waiting if r not in expired]
            for request in expired:
                self._busy.discard(request.connection)
        if not self._waiting:
            return Turn([], expired, None)
        first = self._waiting[0]
        full = len(self._waiting) >= min(self._max_batch, max(open_sessions, 1))
        until = first.arrived + self._window
        if not full and now < until:
            deadlines = [r.deadline for r in self._waiting if r.deadline is not None]
            return Turn([], expired, min([until, *deadlines]))
        batch = self._waiting[: self._max_batch]
        self._waiting = self._waiting[self._max_batch :]
        return Turn(batch, expired, None)
