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

_Item = tuple[int, RunnerEvent]


def _transition(to_state: SessionState) -> TransitionEvent:
    return TransitionEvent(Transition(SessionEvent.START_SESSION, SessionState.READY, to_state))


def _conn(event: SessionEvent, from_state: SessionState, to_state: SessionState) -> TransitionEvent:
    return TransitionEvent(Transition(event, from_state, to_state))


async def _next(agen: AsyncIterator[_Item]) -> _Item:
    return await asyncio.wait_for(anext(agen), timeout=1.0)


async def _aclose(agen: AsyncIterator[_Item]) -> None:
    await cast(AsyncGenerator[_Item, None], agen).aclose()


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
    assert [await _next(agen), await _next(agen)] == [(2, ErrorEvent("b")), (3, ErrorEvent("c"))]
    await _aclose(agen)


async def test_subscribe_without_since_skips_the_backlog() -> None:
    stream = EventStream()
    stream.emit(ErrorEvent("old"))

    agen = stream.subscribe()
    pending = asyncio.create_task(_next(agen))
    await asyncio.sleep(0)  # let the subscription register before the next emit
    stream.emit(ErrorEvent("new"))

    assert await pending == (2, ErrorEvent("new"))
    await _aclose(agen)


async def test_events_after_subscribe_arrive_before_first_iteration() -> None:
    stream = EventStream()
    agen = stream.subscribe()
    # Emitted after subscribe but before the first iteration: eager registration
    # means it is still delivered rather than lost.
    stream.emit(ErrorEvent("between"))
    assert await _next(agen) == (1, ErrorEvent("between"))
    await _aclose(agen)


async def test_since_beyond_the_journal_still_delivers_live_events() -> None:
    stream = EventStream()
    stream.emit(ErrorEvent("a"))
    agen = stream.subscribe(since=100)
    stream.emit(ErrorEvent("b"))
    assert await _next(agen) == (2, ErrorEvent("b"))
    await _aclose(agen)


async def test_live_event_reaches_every_subscriber() -> None:
    stream = EventStream()
    first = stream.subscribe()
    second = stream.subscribe()
    waiting = [asyncio.create_task(_next(first)), asyncio.create_task(_next(second))]
    await asyncio.sleep(0)

    stream.emit(ErrorEvent("x"))

    assert [await task for task in waiting] == [(1, ErrorEvent("x")), (1, ErrorEvent("x"))]
    await _aclose(first)
    await _aclose(second)


async def test_resume_after_a_gap_has_no_dupes_or_holes() -> None:
    stream = EventStream()
    stream.emit(ErrorEvent("1"))

    agen = stream.subscribe(since=0)
    assert await _next(agen) == (1, ErrorEvent("1"))
    await _aclose(agen)  # consumer drops

    stream.emit(ErrorEvent("2"))
    stream.emit(ErrorEvent("3"))

    resumed = stream.subscribe(since=stream.snapshot().last_seq - 2)
    assert [await _next(resumed), await _next(resumed)] == [
        (2, ErrorEvent("2")),
        (3, ErrorEvent("3")),
    ]
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


def test_history_is_bounded_to_the_limit() -> None:
    stream = EventStream(history_limit=3)
    for i in range(5):
        stream.emit(ErrorEvent(str(i)))

    # The journal keeps only the most recent `history_limit` events, but the
    # sequence counter keeps climbing so the numbers stay globally monotonic.
    assert stream.snapshot().last_seq == 5
    assert list(stream._history) == [
        (3, ErrorEvent("2")),
        (4, ErrorEvent("3")),
        (5, ErrorEvent("4")),
    ]


async def test_replay_is_capped_and_the_dropped_span_shows_as_a_gap() -> None:
    stream = EventStream(history_limit=2)
    for i in range(4):  # seqs 1..4
        stream.emit(ErrorEvent(str(i)))

    # Asking to replay from the very start only yields what history still holds;
    # the first delivered sequence number is 3, so a consumer that asked after 0
    # sees seqs 1 and 2 are missing.
    agen = stream.subscribe(since=0)
    replayed = [await _next(agen), await _next(agen)]
    assert replayed == [(3, ErrorEvent("2")), (4, ErrorEvent("3"))]
    await _aclose(agen)


async def test_slow_subscriber_drops_its_oldest_and_stays_on_the_live_tail() -> None:
    stream = EventStream(subscriber_limit=2)
    agen = stream.subscribe()
    for i in range(5):  # seqs 1..5, none consumed yet
        stream.emit(ErrorEvent(str(i)))

    # The queue held at most two events; the oldest were dropped as newer ones
    # arrived, so the consumer resumes on the freshest tail and sees the gap in
    # the sequence numbers (1..3 are missing).
    tail = [await _next(agen), await _next(agen)]
    assert tail == [(4, ErrorEvent("3")), (5, ErrorEvent("4"))]
    await _aclose(agen)


async def test_a_slow_subscriber_does_not_starve_a_prompt_one() -> None:
    stream = EventStream(subscriber_limit=2)
    slow = stream.subscribe()  # never read
    prompt = stream.subscribe()

    # The prompt subscriber drains each event as it arrives, so its queue never
    # fills and it loses nothing even under the same small limit; the slow one
    # only bounds its own memory, and emit never blocks on either.
    got = []
    for i in range(5):  # seqs 1..5
        stream.emit(ErrorEvent(str(i)))
        got.append(await _next(prompt))
    assert got == [(seq, ErrorEvent(str(seq - 1))) for seq in range(1, 6)]
    await _aclose(prompt)
    await _aclose(slow)
