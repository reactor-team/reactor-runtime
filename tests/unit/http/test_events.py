import json

from reactor_runtime.core import (
    ClipReadyEvent,
    ConnId,
    ErrorEvent,
    InboundCommandEvent,
    SessionEvent,
    SessionMetricEvent,
    SessionState,
    Transition,
    TransitionEvent,
)
from reactor_runtime.http.events import format_sse, runner_event_to_dict


def _transition() -> TransitionEvent:
    return TransitionEvent(
        Transition(SessionEvent.START_SESSION, SessionState.READY, SessionState.WAITING)
    )


def test_transition_event_renders_named_states_and_detail() -> None:
    assert runner_event_to_dict(_transition()) == {
        "type": "transition",
        "event": "start_session",
        "from": "ready",
        "to": "waiting",
        "detail": {},
    }


def test_transition_carries_connection_detail() -> None:
    move = TransitionEvent(
        Transition(
            SessionEvent.CONNECTION_OPENED,
            SessionState.WAITING,
            SessionState.STREAMING,
            detail={"conn_id": ConnId(3)},
        )
    )
    assert runner_event_to_dict(move) == {
        "type": "transition",
        "event": "connection_opened",
        "from": "waiting",
        "to": "streaming",
        "detail": {"conn_id": 3},
    }


def test_transition_carries_the_negotiation_answer() -> None:
    move = TransitionEvent(
        Transition(
            SessionEvent.CONNECTION_ANSWERED,
            SessionState.WAITING,
            SessionState.WAITING,
            detail={"conn_id": ConnId(4), "answer": {"type": "answer", "sdp": "v=0..."}},
        )
    )
    assert runner_event_to_dict(move) == {
        "type": "transition",
        "event": "connection_answered",
        "from": "waiting",
        "to": "waiting",
        "detail": {"conn_id": 4, "answer": {"type": "answer", "sdp": "v=0..."}},
    }


def test_command_event() -> None:
    assert runner_event_to_dict(InboundCommandEvent("set_mode", {"mode": "turbo"}, ConnId(2))) == {
        "type": "command",
        "name": "set_mode",
        "args": {"mode": "turbo"},
        "conn_id": 2,
    }


def test_clip_ready_event() -> None:
    assert runner_event_to_dict(ClipReadyEvent("clip-1")) == {
        "type": "clip_ready",
        "clip_id": "clip-1",
    }


def test_metric_event() -> None:
    assert runner_event_to_dict(SessionMetricEvent("fps", 30.0)) == {
        "type": "metric",
        "name": "fps",
        "value": 30.0,
    }


def test_error_event() -> None:
    assert runner_event_to_dict(ErrorEvent("boom")) == {"type": "error", "message": "boom"}


def test_format_sse_frames_id_and_data() -> None:
    message = format_sse(7, _transition())

    assert message.startswith("id: 7\n")
    assert message.endswith("\n\n")
    body = json.loads(message.split("data: ", 1)[1].strip())
    assert body["type"] == "transition"
