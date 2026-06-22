import json

from reactor_runtime.core import (
    ClipReadyEvent,
    ConnectionEvent,
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


def test_transition_event_renders_named_states() -> None:
    assert runner_event_to_dict(_transition()) == {
        "type": "transition",
        "event": "start_session",
        "from": "ready",
        "to": "waiting",
    }


def test_connection_event() -> None:
    assert runner_event_to_dict(ConnectionEvent(ConnId(3), opened=True)) == {
        "type": "connection",
        "conn_id": 3,
        "opened": True,
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
