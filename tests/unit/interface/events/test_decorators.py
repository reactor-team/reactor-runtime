import pytest

from reactor_runtime.core import Command, InputField, UploadedFile
from reactor_runtime.core.fields import NO_DEFAULT
from reactor_runtime.interface.events.decorators import (
    CONNECTED_ATTR,
    DISCONNECTED_ATTR,
    EVENT_ATTR,
    FILE_UPLOADED_ATTR,
    SESSION_ENDED_ATTR,
    SESSION_STARTED_ATTR,
    EventHandler,
    connected,
    disconnected,
    event,
    file_uploaded,
    make_command,
    session_ended,
    session_started,
)


class Model:
    @event(name="set_brightness", description="Adjust brightness")
    async def set_brightness(
        self, level: float = InputField(default=1.0, ge=0.0, le=1.0)
    ) -> None: ...

    @event(name="ping")
    def ping(self, client: str, message: str) -> None: ...

    @session_started
    def on_session_start(self) -> None: ...

    @session_ended
    def on_session_end(self) -> None: ...

    @connected
    def on_connect(self) -> None: ...

    @disconnected
    def on_disconnect(self) -> None: ...

    @file_uploaded
    def on_file(self, uploaded_file: UploadedFile) -> None: ...


def test_event_stamps_handler_metadata() -> None:
    meta = getattr(Model.set_brightness, EVENT_ATTR)
    assert isinstance(meta, EventHandler)
    assert meta.name == "set_brightness"
    assert meta.description == "Adjust brightness"
    assert meta.is_async is True


def test_event_synthesises_a_command_mirroring_the_signature() -> None:
    meta = getattr(Model.set_brightness, EVENT_ATTR)
    command = meta.command
    assert issubclass(command, Command)
    assert command.name == "set_brightness"
    assert set(command.__command_fields__) == {"level"}
    assert command.__command_fields__["level"].info.ge == 0.0


def test_reserved_params_are_stripped_from_the_command() -> None:
    meta = getattr(Model.ping, EVENT_ATTR)
    assert meta.reserved == ("client",)
    assert set(meta.command.__command_fields__) == {"message"}


def test_lifecycle_decorators_tag_their_methods() -> None:
    assert getattr(Model.on_connect, CONNECTED_ATTR) is True
    assert getattr(Model.on_disconnect, DISCONNECTED_ATTR) is True
    assert getattr(Model.on_file, FILE_UPLOADED_ATTR) is True


def test_session_lifecycle_decorators_are_distinct_from_connection_hooks() -> None:
    assert getattr(Model.on_session_start, SESSION_STARTED_ATTR) is True
    assert getattr(Model.on_session_end, SESSION_ENDED_ATTR) is True
    # Session hooks are not tagged as connection hooks, and vice versa.
    assert not hasattr(Model.on_session_start, CONNECTED_ATTR)
    assert not hasattr(Model.on_connect, SESSION_STARTED_ATTR)


def test_event_rejects_reserved_lifecycle_name() -> None:
    with pytest.raises(ValueError, match="reserved"):

        class Bad:
            @event(name="connected")
            def handler(self) -> None: ...


def test_file_uploaded_requires_the_uploaded_file_param() -> None:
    with pytest.raises(TypeError, match="uploaded_file"):

        @file_uploaded
        def handler(self: object, wrong_name: UploadedFile) -> None: ...


def test_make_command_builds_a_named_command() -> None:
    command = make_command("set_mode", [("mode", str, "turbo"), ("seed", int)])
    assert command.name == "set_mode"
    assert command.__name__ == "SetMode"
    assert set(command.__command_fields__) == {"mode", "seed"}
    assert command.__command_fields__["mode"].info.default == "turbo"
    assert command.__command_fields__["seed"].info.default is NO_DEFAULT
