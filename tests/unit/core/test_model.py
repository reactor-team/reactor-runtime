from dataclasses import dataclass

from reactor_runtime.core import (
    ClientConnected,
    Command,
    ConnId,
    EndReason,
    FileUploaded,
    InboundCommandEvent,
    ReactorEvent,
    RunnerEvent,
    SessionEnded,
    SessionStarted,
    TransitionEvent,
    UploadedFile,
)
from reactor_runtime.core.session import SessionEvent, SessionState, Transition


@dataclass
class SetPrompt(Command):
    prompt: str


def test_command_is_an_open_set() -> None:
    cmd = SetPrompt(prompt="a cat")
    assert isinstance(cmd, Command)
    assert cmd.prompt == "a cat"


def test_reactor_events_are_a_closed_authoritative_set() -> None:
    started = SessionStarted(session_id="s-1")
    ended = SessionEnded(session_id="s-1", reason=EndReason.STOPPED)
    connected = ClientConnected(conn_id=ConnId(1), total=1)

    assert isinstance(started, ReactorEvent)
    assert isinstance(ended, ReactorEvent)
    assert isinstance(connected, ReactorEvent)
    assert ended.reason is EndReason.STOPPED


def test_file_uploaded_carries_the_fetched_bytes() -> None:
    event = FileUploaded(
        file=UploadedFile(upload_id="u-1", name="cat.png", mime_type="image/png", data=b"\x89PNG"),
        conn_id=ConnId(2),
    )
    assert isinstance(event, ReactorEvent)
    assert event.file.data == b"\x89PNG"
    assert event.conn_id == ConnId(2)


def test_runner_event_union_membership() -> None:
    transition = Transition(SessionEvent.START_SESSION, SessionState.READY, SessionState.WAITING)
    egress: list[RunnerEvent] = [
        TransitionEvent(transition),
        InboundCommandEvent(name="set_prompt", args={"prompt": "hi"}, conn_id=ConnId(1)),
    ]
    for event in egress:
        assert isinstance(event, RunnerEvent)

    assert not isinstance(SessionStarted(session_id="s-1"), RunnerEvent)


def test_inbound_command_event_conn_id_is_optional() -> None:
    assert InboundCommandEvent(name="ping", args={}).conn_id is None
