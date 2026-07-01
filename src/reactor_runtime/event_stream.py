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

Memory is bounded so a busy, hours-long session cannot grow the journal without
limit. Replay history is capped at a fixed number of recent events, and each
subscriber's live queue is capped too — a consumer that falls behind has its
oldest queued events dropped rather than buffered forever. Because every event
carries its own monotonic sequence number all the way to the wire, a consumer
detects a drop as a jump in the sequence numbers it receives and can reconcile
from a fresh :meth:`snapshot`; the runtime never blocks on a slow reader.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from reactor_runtime.core import (
    RunnerEvent,
    SessionEvent,
    SessionState,
    TransitionEvent,
)

DEFAULT_HISTORY_LIMIT = 4096
"""How many recent events the journal retains for replay by default.

A consumer can resume at most this many events back; a ``since`` older than the
oldest retained event replays from the oldest one, and the gap is visible as a
jump in sequence numbers.
"""

DEFAULT_SUBSCRIBER_LIMIT = 4096
"""How many undelivered events a single subscriber's queue holds by default.

Past this, the oldest queued event is dropped to admit the newest, so a stalled
or slow consumer bounds its own memory instead of the writer's.
"""

_SubscriberQueue = asyncio.Queue[tuple[int, "RunnerEvent"]]


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
    session state and connection count so it can answer :meth:`snapshot` for a
    reconciling consumer.

    Memory is bounded on both axes. Replay history keeps only the most recent
    ``history_limit`` events, so a consumer cannot replay further back than that.
    Each subscriber holds a queue capped at ``subscriber_limit``; when a consumer
    falls behind and the queue fills, its oldest queued event is dropped to make
    room for the newest. Both drops — replaying past the retained history, or a
    full subscriber queue — surface to the consumer the same way: a gap in the
    sequence numbers it receives, which it reconciles against a fresh snapshot.
    """

    def __init__(
        self,
        *,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        subscriber_limit: int = DEFAULT_SUBSCRIBER_LIMIT,
    ) -> None:
        """Start an empty journal with no subscribers.

        Args:
            history_limit: The number of recent events retained for replay.
            subscriber_limit: The number of undelivered events a single
                subscriber's queue holds before its oldest is dropped.
        """
        self._seq = 0
        self._history: deque[tuple[int, RunnerEvent]] = deque(maxlen=history_limit)
        self._subscriber_limit = subscriber_limit
        self._subscribers: set[_SubscriberQueue] = set()
        self._state: SessionState | None = None
        self._connections = 0

    def emit(self, event: RunnerEvent) -> None:
        """Append an event to the journal and deliver it to live subscribers.

        Assigns the next sequence number, folds the event into the tracked
        session state, and hands it to every current subscriber. A subscriber
        whose queue is full has its oldest queued event dropped so the newest is
        always admitted — the writer is never held back by a slow reader.

        This is the single writer: it must be called only on the one event loop
        that owns the stream, so the ``+= 1`` sequence bump and the history
        append need no lock. Cross-thread producers marshal onto that loop before
        emitting (for example via ``loop.call_soon_threadsafe``) rather than
        calling this directly.
        """
        self._seq += 1
        item = (self._seq, event)
        self._history.append(item)
        self._fold(event)
        for queue in self._subscribers:
            self._offer(queue, item)

    def _offer(self, queue: _SubscriberQueue, item: tuple[int, RunnerEvent]) -> None:
        """Hand *item* to a subscriber, shedding its oldest event if the queue is full.

        A full queue means the consumer is not keeping up. Dropping the oldest
        queued event (rather than refusing the newest) keeps the consumer on the
        live tail; the dropped sequence numbers leave a hole it detects as a jump
        in the ids it receives. This runs synchronously within :meth:`emit` with
        no ``await`` between the drop and the enqueue, so no consumer interleaves:
        a queue is only ever full when no getter is currently waiting on it.
        """
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
            queue.put_nowait(item)

    def subscribe(self, since: int | None = None) -> AsyncIterator[tuple[int, RunnerEvent]]:
        """Return an iterator over ``(seq, event)`` pairs after *since*, then live ones.

        Registers immediately — before the first iteration — so an event emitted
        after this call is delivered even if the caller has not started consuming
        yet. With *since* set, every retained event with a greater sequence
        number is replayed in order before live delivery begins, so a consumer
        resumes exactly where it left off; a *since* older than the oldest
        retained event replays from there, and the missing span shows as a jump
        in the yielded sequence numbers. With *since* ``None``, delivery starts
        from the next event. The iterator runs until the caller stops consuming
        it.
        """
        start = self._seq if since is None else since
        queue: _SubscriberQueue = asyncio.Queue(self._subscriber_limit)
        self._subscribers.add(queue)
        backlog = [item for item in self._history if item[0] > start]
        return self._stream(queue, backlog)

    async def _stream(
        self, queue: _SubscriberQueue, backlog: list[tuple[int, RunnerEvent]]
    ) -> AsyncIterator[tuple[int, RunnerEvent]]:
        """Yield the captured backlog, then live events, deregistering on exit.

        The queue is registered before the backlog is captured with no await in
        between, so the two are disjoint: the backlog holds everything retained
        up to the subscribe call and the queue holds only what is emitted after
        it.
        """
        try:
            for item in backlog:
                yield item
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
        """Update the tracked session view from one event.

        Connection occupancy rides the transition itself: a move whose event is
        ``CONNECTION_OPENED`` or ``CONNECTION_CLOSED`` adjusts the count, and
        every other move (including a ``CONNECTION_ANSWERED`` self-loop) leaves
        it alone.
        """
        if isinstance(event, TransitionEvent):
            transition = event.transition
            self._state = transition.to_state
            if transition.event is SessionEvent.CONNECTION_OPENED:
                self._connections += 1
            elif transition.event is SessionEvent.CONNECTION_CLOSED:
                self._connections = max(0, self._connections - 1)
