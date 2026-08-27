import time

import pytest

from reactor_runtime.core import JOURNAL_EVENTS, SessionEvent, SessionState, Transition
from reactor_runtime.runner import SessionStateMachine

# Every static (count-independent) edge as (start, event, end). Each is a single
# deterministic send from a fresh machine. Occupancy edges depend on the live
# connection count and are exercised separately below.
LEGAL_EDGES: list[tuple[SessionState, SessionEvent, SessionState]] = [
    (SessionState.CREATED, SessionEvent.INITIALIZATION_SUCCESS, SessionState.READY),
    (SessionState.CREATED, SessionEvent.INITIALIZATION_FAIL, SessionState.TERMINATED),
    # INITIALIZING is the loading self-loop: legal only in CREATED, changing no
    # state, and rejected everywhere else.
    (SessionState.CREATED, SessionEvent.INITIALIZING, SessionState.CREATED),
    (SessionState.READY, SessionEvent.START_SESSION, SessionState.WAITING),
    # Eviction is terminal from every live state — an idle eviction or a crash.
    (SessionState.CREATED, SessionEvent.EVICTION, SessionState.TERMINATED),
    (SessionState.READY, SessionEvent.EVICTION, SessionState.TERMINATED),
    (SessionState.WAITING, SessionEvent.EVICTION, SessionState.TERMINATED),
    (SessionState.STREAMING, SessionEvent.EVICTION, SessionState.TERMINATED),
    (SessionState.ORPHANED, SessionEvent.EVICTION, SessionState.TERMINATED),
    (SessionState.CLOSING, SessionEvent.EVICTION, SessionState.TERMINATED),
    (SessionState.WAITING, SessionEvent.STOP_SESSION, SessionState.CLOSING),
    (SessionState.WAITING, SessionEvent.TIMEOUT, SessionState.CLOSING),
    (SessionState.STREAMING, SessionEvent.STOP_SESSION, SessionState.CLOSING),
    (SessionState.ORPHANED, SessionEvent.STOP_SESSION, SessionState.CLOSING),
    (SessionState.ORPHANED, SessionEvent.TIMEOUT, SessionState.CLOSING),
    (SessionState.CLOSING, SessionEvent.CLEANUP_COMPLETE, SessionState.READY),
    # A negotiation answer self-loops in every state, so the answer is always
    # journalled wherever the session happens to be.
    *[(state, SessionEvent.CONNECTION_ANSWERED, state) for state in SessionState],
]


def expect_state(sm: SessionStateMachine, state: SessionState) -> None:
    # Routing the check through a call keeps the type checker from narrowing the
    # property to a single literal across the consecutive asserts of a walk.
    assert sm.current_state is state


@pytest.mark.parametrize(("start", "event", "end"), LEGAL_EDGES)
def test_legal_edge_flips_state(
    start: SessionState, event: SessionEvent, end: SessionState
) -> None:
    sm = SessionStateMachine(initial_state=start)
    assert sm.send(event) is True
    assert sm.current_state is end


@pytest.mark.parametrize(("start", "event", "end"), LEGAL_EDGES)
def test_illegal_static_edge_is_rejected_from_every_other_state(
    start: SessionState, event: SessionEvent, end: SessionState
) -> None:
    legal_starts = {edge_start for edge_start, edge_event, _ in LEGAL_EDGES if edge_event is event}
    for other in SessionState:
        if other in legal_starts:
            continue
        sm = SessionStateMachine(initial_state=other)
        assert sm.send(event) is False
        assert sm.current_state is other


def test_default_initial_state_is_created() -> None:
    assert SessionStateMachine().current_state is SessionState.CREATED


# -- occupancy, derived from the connection count -----------------------------


def test_first_connection_opens_streaming() -> None:
    sm = SessionStateMachine(initial_state=SessionState.WAITING)
    assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1) is True
    assert sm.current_state is SessionState.STREAMING


def test_reconnect_from_orphaned_resumes_streaming() -> None:
    sm = SessionStateMachine(initial_state=SessionState.ORPHANED)
    assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1) is True
    assert sm.current_state is SessionState.STREAMING


def test_additional_connection_is_a_streaming_self_loop() -> None:
    sm = SessionStateMachine(initial_state=SessionState.WAITING)
    assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1) is True
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=2) is True
    assert sm.current_state is SessionState.STREAMING
    (transition,) = seen
    assert transition.from_state is SessionState.STREAMING
    assert transition.to_state is SessionState.STREAMING
    assert transition.is_session_start is False
    assert transition.is_session_end is False


def test_close_with_others_remaining_stays_streaming_then_last_orphans() -> None:
    sm = SessionStateMachine(initial_state=SessionState.WAITING)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=2)
    assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=1) is True
    expect_state(sm, SessionState.STREAMING)
    assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=2) is True
    expect_state(sm, SessionState.ORPHANED)


def test_connection_open_is_illegal_outside_an_open_session() -> None:
    for state in (
        SessionState.CREATED,
        SessionState.READY,
        SessionState.CLOSING,
        SessionState.TERMINATED,
    ):
        sm = SessionStateMachine(initial_state=state)
        assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1) is False
        assert sm.current_state is state


def test_connection_close_is_illegal_outside_streaming() -> None:
    for state in SessionState:
        if state is SessionState.STREAMING:
            continue
        sm = SessionStateMachine(initial_state=state)
        assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=1) is False
        assert sm.current_state is state


def test_count_resets_across_a_session_restart() -> None:
    # A stale count from a prior streaming span must not survive into a new one,
    # or the last close of the new session would fail to orphan it.
    sm = SessionStateMachine(initial_state=SessionState.WAITING)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=2)
    sm.send(SessionEvent.STOP_SESSION)
    sm.send(SessionEvent.CLEANUP_COMPLETE)
    sm.send(SessionEvent.START_SESSION)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=3)
    expect_state(sm, SessionState.STREAMING)
    assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=3) is True
    expect_state(sm, SessionState.ORPHANED)


# -- connection answered, a count-neutral self-loop legal in every state ------


def test_answer_self_loops_in_every_state_without_changing_state() -> None:
    for state in SessionState:
        sm = SessionStateMachine(initial_state=state)
        seen: list[Transition] = []
        sm.on_transition(seen.append)
        assert sm.send(SessionEvent.CONNECTION_ANSWERED, conn_id=7) is True
        assert sm.current_state is state
        (transition,) = seen
        assert transition.from_state is state
        assert transition.to_state is state
        assert transition.detail == {"conn_id": 7}


def test_answer_does_not_reset_the_live_count() -> None:
    # An answer self-loop must leave occupancy alone, or it would zero the count
    # and the last close would fail to orphan the session.
    sm = SessionStateMachine(initial_state=SessionState.WAITING)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=2)
    sm.send(SessionEvent.CONNECTION_ANSWERED, conn_id=3)
    expect_state(sm, SessionState.STREAMING)
    assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=1) is True
    expect_state(sm, SessionState.STREAMING)
    assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=2) is True
    expect_state(sm, SessionState.ORPHANED)


# -- journal events, count-neutral self-loops legal in every state ------------


@pytest.mark.parametrize("event", sorted(JOURNAL_EVENTS, key=lambda e: e.name))
@pytest.mark.parametrize("state", list(SessionState))
def test_journal_event_self_loops_in_every_state(event: SessionEvent, state: SessionState) -> None:
    sm = SessionStateMachine(initial_state=state)
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    assert sm.send(event, payload="x") is True
    assert sm.current_state is state
    (transition,) = seen
    assert transition.from_state is state
    assert transition.to_state is state
    assert transition.detail == {"payload": "x"}
    assert transition.is_session_start is False
    assert transition.is_session_end is False


def test_journal_event_does_not_reset_the_live_count() -> None:
    # A journal self-loop must leave occupancy alone, or a chunk_ready between
    # two closes would zero the count and the last close would fail to orphan
    # the session.
    sm = SessionStateMachine(initial_state=SessionState.WAITING)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1)
    sm.send(SessionEvent.CONNECTION_OPENED, conn_id=2)
    for event in JOURNAL_EVENTS:
        assert sm.send(event) is True
    expect_state(sm, SessionState.STREAMING)
    assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=1) is True
    expect_state(sm, SessionState.STREAMING)
    assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=2) is True
    expect_state(sm, SessionState.ORPHANED)


def test_send_stamps_the_transition_with_the_apply_time() -> None:
    sm = SessionStateMachine(initial_state=SessionState.READY)
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    before = time.time_ns() // 1_000_000
    assert sm.send(SessionEvent.START_SESSION) is True
    after = time.time_ns() // 1_000_000
    (transition,) = seen
    assert before <= transition.ts_ms <= after


# -- the move record and listener contract ------------------------------------


def test_full_lifecycle_walk() -> None:
    sm = SessionStateMachine()
    assert sm.send(SessionEvent.INITIALIZATION_SUCCESS) is True
    assert sm.send(SessionEvent.START_SESSION) is True
    assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1) is True
    expect_state(sm, SessionState.STREAMING)
    assert sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=1) is True
    expect_state(sm, SessionState.ORPHANED)
    assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=2) is True
    expect_state(sm, SessionState.STREAMING)
    assert sm.send(SessionEvent.STOP_SESSION) is True
    assert sm.send(SessionEvent.CLEANUP_COMPLETE) is True
    expect_state(sm, SessionState.READY)


def test_rejected_event_notifies_no_listener() -> None:
    sm = SessionStateMachine(initial_state=SessionState.READY)
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=1) is False
    assert seen == []


def test_listener_receives_the_recorded_transition() -> None:
    sm = SessionStateMachine(initial_state=SessionState.READY)
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    sm.send(SessionEvent.START_SESSION, session_id="s-1")
    assert len(seen) == 1
    (transition,) = seen
    assert transition.event is SessionEvent.START_SESSION
    assert transition.from_state is SessionState.READY
    assert transition.to_state is SessionState.WAITING
    assert transition.detail == {"session_id": "s-1"}
    assert transition.is_session_start is True


def test_listeners_fire_in_registration_order_after_state_flips() -> None:
    sm = SessionStateMachine(initial_state=SessionState.READY)
    order: list[str] = []

    def first(_: Transition) -> None:
        # The state is already the target by the time a listener runs.
        assert sm.current_state is SessionState.WAITING
        order.append("first")

    def second(_: Transition) -> None:
        order.append("second")

    sm.on_transition(first)
    sm.on_transition(second)
    sm.send(SessionEvent.START_SESSION)
    assert order == ["first", "second"]


def test_self_loop_notifies_without_changing_state() -> None:
    sm = SessionStateMachine(initial_state=SessionState.STREAMING)
    seen: list[Transition] = []
    sm.on_transition(seen.append)
    assert sm.send(SessionEvent.CONNECTION_OPENED, conn_id=7) is True
    assert sm.current_state is SessionState.STREAMING
    (transition,) = seen
    assert transition.from_state is SessionState.STREAMING
    assert transition.to_state is SessionState.STREAMING
    assert transition.is_session_start is False
    assert transition.is_session_end is False
