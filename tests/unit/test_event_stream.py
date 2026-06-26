import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast

from reactor_runtime.core import (
    ErrorEvent,
    RunnerEvent,
    SessionEvent,
    SessionState,
    Transition,
    TransitionEvent,
)
from reactor_runtime.event_stream import EventStream, SessionSnapshot


def _transition(to_state: SessionState) -> TransitionEvent:
    return TransitionEvent(Transition(SessionEvent.START_SESSION, SessionState.READY, to_state))


def _conn(event: SessionEvent, from_state: SessionState, to_state: SessionState) -> TransitionEvent:
    return TransitionEvent(Transition(event, from_state, to_state))


async def _next(agen: AsyncIterator[RunnerEvent]) -> RunnerEvent:
    return await asyncio.wait_for(anext(agen), timeout=1.0)


async def _aclose(agen: AsyncIterator[RunnerEvent]) -> None:
    await cast(AsyncGenerator[RunnerEvent, None], agen).aclose()


def test_snapshot_starts_empty() -> None:
    assert EventStream().snapshot() == SessionSnapshot(state=None, connections=0, last_seq=0)


def test_snapshot_reflects_transitions_and_connections() -> None:
    stream = EventStream()
    stream.emit(_transition(SessionState.WAITING))
    stream.emit(_conn(SessionEvent.CONNECTION_OPENED, SessionState.WAITING, SessionState.STREAMING))
    stream.emit(
        _conn(SessionEvent.CONNECTION_OPENED, SessionState.STREAMING, SessionState.STREAMING)
    )
    stream.emit(
        _conn(SessionEvent.CONNECTION_CLOSED, SessionState.STREAMING, SessionState.STREAMING)
    )

    snapshot = stream.snapshot()
    assert snapshot.state is SessionState.STREAMING
    assert snapshot.connections == 1
    assert snapshot.last_seq == 4


def test_answer_self_loop_leaves_the_count_untouched() -> None:
    stream = EventStream()
    stream.emit(_conn(SessionEvent.CONNECTION_OPENED, SessionState.WAITING, SessionState.STREAMING))
    stream.emit(
        _conn(SessionEvent.CONNECTION_ANSWERED, SessionState.STREAMING, SessionState.STREAMING)
    )
    assert stream.snapshot().connections == 1


def test_connection_count_never_goes_negative() -> None:
    stream = EventStream()
    stream.emit(
        _conn(SessionEvent.CONNECTION_CLOSED, SessionState.STREAMING, SessionState.ORPHANED)
    )
    assert stream.snapshot().connections == 0


async def test_subscribe_replays_events_after_since() -> None:
    stream = EventStream()
    stream.emit(ErrorEvent("a"))
    stream.emit(ErrorEvent("b"))
    stream.emit(ErrorEvent("c"))

    agen = stream.subscribe(since=1)
    assert [await _next(agen), await _next(agen)] == [ErrorEvent("b"), ErrorEvent("c")]
    await _aclose(agen)


async def test_subscribe_without_since_skips_the_backlog() -> None:
    stream = EventStream()
    stream.emit(ErrorEvent("old"))

    agen = stream.subscribe()
    pending = asyncio.create_task(_next(agen))
    await asyncio.sleep(0)  # let the subscription register before the next emit
    stream.emit(ErrorEvent("new"))

    assert await pending == ErrorEvent("new")
    await _aclose(agen)


async def test_events_after_subscribe_arrive_before_first_iteration() -> None:
    stream = EventStream()
    agen = stream.subscribe()
    # Emitted after subscribe but before the first iteration: eager registration
    # means it is still delivered rather than lost.
    stream.emit(ErrorEvent("between"))
    assert await _next(agen) == ErrorEvent("between")
    await _aclose(agen)


async def test_since_beyond_the_journal_still_delivers_live_events() -> None:
    stream = EventStream()
    stream.emit(ErrorEvent("a"))
    agen = stream.subscribe(since=100)
    stream.emit(ErrorEvent("b"))
    assert await _next(agen) == ErrorEvent("b")
    await _aclose(agen)


async def test_live_event_reaches_every_subscriber() -> None:
    stream = EventStream()
    first = stream.subscribe()
    second = stream.subscribe()
    waiting = [asyncio.create_task(_next(first)), asyncio.create_task(_next(second))]
    await asyncio.sleep(0)

    stream.emit(ErrorEvent("x"))

    assert [await task for task in waiting] == [ErrorEvent("x"), ErrorEvent("x")]
    await _aclose(first)
    await _aclose(second)


async def test_resume_after_a_gap_has_no_dupes_or_holes() -> None:
    stream = EventStream()
    stream.emit(ErrorEvent("1"))

    agen = stream.subscribe(since=0)
    assert await _next(agen) == ErrorEvent("1")
    await _aclose(agen)  # consumer drops

    stream.emit(ErrorEvent("2"))
    stream.emit(ErrorEvent("3"))

    resumed = stream.subscribe(since=stream.snapshot().last_seq - 2)
    assert [await _next(resumed), await _next(resumed)] == [ErrorEvent("2"), ErrorEvent("3")]
    await _aclose(resumed)


async def test_ending_a_subscription_deregisters_it() -> None:
    stream = EventStream()
    agen = stream.subscribe()
    pending = asyncio.create_task(_next(agen))
    await asyncio.sleep(0)
    assert len(stream._subscribers) == 1

    stream.emit(ErrorEvent("x"))
    await pending
    await _aclose(agen)

    assert len(stream._subscribers) == 0
