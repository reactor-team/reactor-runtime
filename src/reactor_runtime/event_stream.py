"""The egress journal.

The single-writer journal the runtime records its facts on. The runner emits
every transition and notable signal here; an HTTP egress route streams them out.
This is the egress half of the public/private inversion: instead of composing a
platform reporter object in, the runtime journals neutral
:data:`~reactor_runtime.core.model.RunnerEvent` facts and lets an external
consumer mirror them.

A consumer reconciles against the runtime as the source of truth — it reads a
:class:`SessionSnapshot` for the current state, then subscribes from the
snapshot's sequence to replay anything it missed and follow live. The consumer's
view is only ever a mirror; the runtime stays authoritative.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from reactor_runtime.core import (
    ConnectionEvent,
    RunnerEvent,
    SessionState,
    TransitionEvent,
)


@dataclass(frozen=True)
class SessionSnapshot:
    """A point-in-time view of the session, for a consumer to reconcile against.

    Attributes:
        state: The current session state, or ``None`` before the first
            transition has been journalled.
        connections: The number of connections currently open.
        last_seq: The sequence number of the last event emitted. Pass it to
            :meth:`EventStream.subscribe` to resume immediately after it.
    """

    state: SessionState | None
    connections: int
    last_seq: int


class EventStream:
    """A single-writer journal of runner events with resumable subscription.

    Only the runner emits; many consumers may subscribe. Each event is assigned a
    monotonic sequence number, so a consumer that drops can resume from the last
    one it saw without missing or repeating an event. The journal also tracks the
    session state and connection count so it can answer
    :meth:`snapshot` for a reconciling consumer.

    Memory is session-scoped. The journal keeps every event for the life of the
    session so a consumer can resume from any point, and each subscriber holds an
    unbounded queue. A consumer that stops reading without ending its subscription
    keeps its queue growing; ending the subscription (closing the iterator or
    letting it be collected) deregisters it.
    """

    def __init__(self) -> None:
        """Start an empty journal with no subscribers."""
        self._seq = 0
        self._history: list[tuple[int, RunnerEvent]] = []
        self._subscribers: set[asyncio.Queue[RunnerEvent]] = set()
        self._state: SessionState | None = None
        self._connections = 0

    def emit(self, event: RunnerEvent) -> None:
        """Append an event to the journal and deliver it to live subscribers.

        Assigns the next sequence number, folds the event into the tracked
        session state, and hands it to every current subscriber.
        """
        self._seq += 1
        self._history.append((self._seq, event))
        self._fold(event)
        for queue in self._subscribers:
            queue.put_nowait(event)

    def subscribe(self, since: int | None = None) -> AsyncIterator[RunnerEvent]:
        """Return an iterator over events after *since*, then live ones.

        Registers immediately — before the first iteration — so an event emitted
        after this call is delivered even if the caller has not started consuming
        yet. With *since* set, every event with a greater sequence number is
        replayed in order before live delivery begins, so a consumer resumes
        exactly where it left off. With *since* ``None``, delivery starts from the
        next event. The iterator runs until the caller stops consuming it.
        """
        start = self._seq if since is None else since
        queue: asyncio.Queue[RunnerEvent] = asyncio.Queue()
        self._subscribers.add(queue)
        backlog = [event for seq, event in self._history if seq > start]
        return self._stream(queue, backlog)

    async def _stream(
        self, queue: asyncio.Queue[RunnerEvent], backlog: list[RunnerEvent]
    ) -> AsyncIterator[RunnerEvent]:
        """Yield the captured backlog, then live events, deregistering on exit.

        The queue is registered before the backlog is captured with no await in
        between, so the two are disjoint: the backlog holds everything up to the
        subscribe call and the queue holds only what is emitted after it.
        """
        try:
            for event in backlog:
                yield event
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    def snapshot(self) -> SessionSnapshot:
        """Return the current session state for a consumer to reconcile against."""
        return SessionSnapshot(
            state=self._state,
            connections=self._connections,
            last_seq=self._seq,
        )

    def _fold(self, event: RunnerEvent) -> None:
        """Update the tracked session view from one event."""
        if isinstance(event, TransitionEvent):
            self._state = event.transition.to_state
        elif isinstance(event, ConnectionEvent):
            self._connections = max(0, self._connections + (1 if event.opened else -1))
