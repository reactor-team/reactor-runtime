import json
import time

from reactor_runtime.core import (
    ConnId,
    SessionEvent,
    SessionState,
    Transition,
    TransitionEvent,
)
from reactor_runtime.http.events import format_sse, runner_event_to_dict


def _transition() -> TransitionEvent:
    return TransitionEvent(
        Transition(SessionEvent.START_SESSION, SessionState.READY, SessionState.WAITING, ts_ms=123)
    )


def test_transition_event_renders_named_states_ts_and_detail() -> None:
    assert runner_event_to_dict(_transition()) == {
        "type": "transition",
        "event": "start_session",
        "from": "ready",
        "to": "waiting",
        "ts": 123,
        "detail": {},
    }


def test_every_envelope_carries_the_stamped_ts() -> None:
    before = time.time_ns() // 1_000_000
    move = TransitionEvent(
        Transition(SessionEvent.START_SESSION, SessionState.READY, SessionState.WAITING)
    )
    after = time.time_ns() // 1_000_000
    assert before <= runner_event_to_dict(move)["ts"] <= after


def test_transition_carries_connection_detail() -> None:
    move = TransitionEvent(
        Transition(
            SessionEvent.CONNECTION_OPENED,
            SessionState.WAITING,
            SessionState.STREAMING,
            detail={"conn_id": ConnId(3)},
            ts_ms=123,
        )
    )
    assert runner_event_to_dict(move) == {
        "type": "transition",
        "event": "connection_opened",
        "from": "waiting",
        "to": "streaming",
        "ts": 123,
        "detail": {"conn_id": 3},
    }


def test_transition_carries_the_negotiation_answer() -> None:
    move = TransitionEvent(
        Transition(
            SessionEvent.CONNECTION_ANSWERED,
            SessionState.WAITING,
            SessionState.WAITING,
            detail={"conn_id": ConnId(4), "answer": {"type": "answer", "sdp": "v=0..."}},
            ts_ms=123,
        )
    )
    assert runner_event_to_dict(move) == {
        "type": "transition",
        "event": "connection_answered",
        "from": "waiting",
        "to": "waiting",
        "ts": 123,
        "detail": {"conn_id": 4, "answer": {"type": "answer", "sdp": "v=0..."}},
    }


def _journal(
    event: SessionEvent, state: SessionState, detail: dict[str, object]
) -> dict[str, object]:
    return runner_event_to_dict(
        TransitionEvent(Transition(event, state, state, detail=detail, ts_ms=123))
    )


def test_command_rides_a_self_loop_envelope() -> None:
    envelope = _journal(
        SessionEvent.COMMAND,
        SessionState.STREAMING,
        {"name": "set_mode", "args": {"mode": "turbo"}, "conn_id": ConnId(2)},
    )
    assert envelope == {
        "type": "transition",
        "event": "command",
        "from": "streaming",
        "to": "streaming",
        "ts": 123,
        "detail": {"name": "set_mode", "args": {"mode": "turbo"}, "conn_id": 2},
    }


def test_clip_ready_rides_a_self_loop_envelope() -> None:
    detail = {
        "session_id": "rec-1",
        "kind": "snap",
        "start_marker": 10.0,
        "end_marker": 40.0,
        "now_marker": 40.0,
        "predicted_ready_at_ms": 1234,
        "playlist_url": "/clips?session_id=rec-1&start=10.000&end=40.000",
    }
    envelope = _journal(SessionEvent.CLIP_READY, SessionState.STREAMING, dict(detail))
    assert envelope["event"] == "clip_ready"
    assert envelope["detail"] == detail


def test_chunk_ready_rides_a_self_loop_envelope() -> None:
    envelope = _journal(
        SessionEvent.CHUNK_READY, SessionState.CLOSING, {"recording_id": "rec-1", "idx": -1}
    )
    assert envelope == {
        "type": "transition",
        "event": "chunk_ready",
        "from": "closing",
        "to": "closing",
        "ts": 123,
        "detail": {"recording_id": "rec-1", "idx": -1},
    }


def test_metric_rides_a_self_loop_envelope() -> None:
    envelope = _journal(SessionEvent.METRIC, SessionState.STREAMING, {"name": "fps", "value": 30.0})
    assert envelope["event"] == "metric"
    assert envelope["detail"] == {"name": "fps", "value": 30.0}


def test_error_rides_a_self_loop_envelope() -> None:
    envelope = _journal(SessionEvent.ERROR, SessionState.ORPHANED, {"message": "boom"})
    assert envelope["event"] == "error"
    assert envelope["detail"] == {"message": "boom"}


def test_format_sse_frames_id_and_data() -> None:
    message = format_sse(7, _transition())

    assert message.startswith("id: 7\n")
    assert message.endswith("\n\n")
    body = json.loads(message.split("data: ", 1)[1].strip())
    assert body["type"] == "transition"
    assert body["ts"] == 123
