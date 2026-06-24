from reactor_runtime.core import SessionEvent, SessionState, Transition


def test_transition_records_the_move() -> None:
    t = Transition(
        event=SessionEvent.START_SESSION,
        from_state=SessionState.READY,
        to_state=SessionState.WAITING,
        detail={"session_id": "s-1"},
    )
    assert t.event is SessionEvent.START_SESSION
    assert t.from_state is SessionState.READY
    assert t.to_state is SessionState.WAITING
    assert t.detail["session_id"] == "s-1"


def test_detail_defaults_to_empty() -> None:
    t = Transition(SessionEvent.CONNECTION_OPENED, SessionState.STREAMING, SessionState.STREAMING)
    assert t.detail == {}


def test_session_start_is_only_ready_to_waiting() -> None:
    start = Transition(SessionEvent.START_SESSION, SessionState.READY, SessionState.WAITING)
    assert start.is_session_start is True
    assert start.is_session_end is False

    rejoin = Transition(
        SessionEvent.CONNECTION_OPENED, SessionState.ORPHANED, SessionState.STREAMING
    )
    assert rejoin.is_session_start is False


def test_session_end_is_only_closing_to_ready() -> None:
    end = Transition(SessionEvent.CLEANUP_COMPLETE, SessionState.CLOSING, SessionState.READY)
    assert end.is_session_end is True
    assert end.is_session_start is False


def test_per_connection_self_loop_is_neither_boundary() -> None:
    opened = Transition(
        SessionEvent.CONNECTION_OPENED, SessionState.STREAMING, SessionState.STREAMING
    )
    assert opened.is_session_start is False
    assert opened.is_session_end is False
