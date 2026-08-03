from dataclasses import dataclass, fields

from reactor_runtime.core import (
    ClientConnected,
    Command,
    ConnId,
    EndReason,
    FileUploaded,
    ReactorEvent,
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
        file=UploadedFile(name="cat.png", mime_type="image/png", data=b"\x89PNG"),
        conn_id=ConnId(2),
    )
    assert isinstance(event, ReactorEvent)
    assert event.file.data == b"\x89PNG"
    assert event.conn_id == ConnId(2)


def test_uploaded_file_does_not_carry_the_upload_id() -> None:
    file = UploadedFile(name="cat.png", mime_type="image/png", data=b"\x89PNG")
    assert not hasattr(file, "upload_id")


def test_uploaded_file_size_is_measured_from_the_bytes() -> None:
    file = UploadedFile(name="cat.png", mime_type="image/png", data=b"\x89PNG")
    assert file.size == 4
    assert UploadedFile(name="empty.bin", mime_type="application/octet-stream", data=b"").size == 0


def test_uploaded_file_size_cannot_be_set_apart_from_the_bytes() -> None:
    field_names = {field.name for field in fields(UploadedFile)}
    assert "size" not in field_names


def test_transition_event_wraps_the_move() -> None:
    transition = Transition(SessionEvent.START_SESSION, SessionState.READY, SessionState.WAITING)
    event = TransitionEvent(transition)
    assert event.transition is transition
    assert not isinstance(SessionStarted(session_id="s-1"), TransitionEvent)
