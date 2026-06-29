import enum
from collections.abc import Callable

import pytest

from reactor_runtime import (
    Input,
    ModelMessage,
    Output,
    ReactorModel,
    Video,
    connected,
    event,
    file_uploaded,
    session_ended,
    session_started,
)
from reactor_runtime.core import Command, InputField, UploadedFile
from reactor_runtime.interface.model import ContractError, ModelContract


class Reply(ModelMessage):
    """The generated reply."""

    image_url: str


class OutTracks(Output):
    main_video: Video


class InTracks(Input):
    camera: Video


class EchoModel(ReactorModel):
    """A tiny echo model."""

    output: OutTracks
    input: InTracks

    @event(name="set_prompt", description="Set the prompt")
    async def set_prompt(self, prompt: str = InputField(min_length=1, max_length=8)) -> None: ...

    @event(name="generate", description="Generate from a prompt")
    async def generate(self, prompt: str, seed: int = 0) -> Reply:
        return Reply(image_url="x")

    @event(name="upload")
    async def upload(self, file: UploadedFile) -> None: ...

    @session_started
    def on_session_start(self) -> None: ...

    @session_ended
    def on_session_end(self) -> None: ...

    @connected
    def on_connect(self) -> None: ...

    @file_uploaded
    def on_file(self, uploaded_file: UploadedFile) -> None: ...


class Speed(enum.IntEnum):
    SLOW = 1
    FAST = 2


class SpeedModel(ReactorModel):
    @event(name="set_speed")
    async def set_speed(self, speed: Speed) -> None: ...


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(EchoModel)
    register_model(SpeedModel)


def contract() -> ModelContract:
    return ModelContract.of(EchoModel)


def test_of_returns_the_cached_contract() -> None:
    assert ModelContract.of(EchoModel) is EchoModel.__reactor_contract__


def test_of_raises_for_a_class_without_a_contract() -> None:
    with pytest.raises(TypeError, match="no model contract"):
        ModelContract.of(object)


def test_commands_are_collected_by_wire_name() -> None:
    assert set(contract().commands) == {"set_prompt", "generate", "upload"}


def test_model_identifier_is_the_snake_case_class_name() -> None:
    assert contract().model == "echo_model"
    assert contract().description == "A tiny echo model."


def test_response_type_comes_from_the_handler_return_annotation() -> None:
    assert contract().commands["generate"].response is Reply
    assert contract().commands["set_prompt"].response is None


def test_messages_are_discovered_from_command_responses() -> None:
    assert set(contract().messages) == {"reply"}
    assert contract().messages["reply"] is Reply


def test_tracks_merge_inputs_and_outputs() -> None:
    tracks = contract().tracks
    assert set(tracks) == {"main_video", "camera"}
    assert tracks["main_video"].direction.value == "out"
    assert tracks["camera"].direction.value == "in"


def test_lifecycle_hooks_are_recorded_by_scope() -> None:
    lifecycle = contract().lifecycle
    assert lifecycle.session_started is not None
    assert lifecycle.session_ended is not None
    assert lifecycle.connected is not None
    assert lifecycle.file_uploaded is not None
    assert lifecycle.disconnected is None


def test_validate_returns_a_typed_command() -> None:
    cmd = contract().validate("generate", {"prompt": "a cat", "seed": 3})
    assert isinstance(cmd, contract().commands["generate"].command)
    assert vars(cmd)["prompt"] == "a cat"
    assert vars(cmd)["seed"] == 3


def test_validate_fills_defaults_for_omitted_optional_fields() -> None:
    cmd = contract().validate("generate", {"prompt": "a cat"})
    assert vars(cmd)["seed"] == 0


def test_validate_rejects_unknown_command() -> None:
    with pytest.raises(ContractError) as excinfo:
        contract().validate("nope", {})
    assert excinfo.value.field == "nope"


def test_validate_rejects_missing_required_argument() -> None:
    with pytest.raises(ContractError) as excinfo:
        contract().validate("generate", {})
    assert excinfo.value.field == "prompt"
    assert "required" in excinfo.value.reason


def test_validate_rejects_unexpected_argument() -> None:
    with pytest.raises(ContractError) as excinfo:
        contract().validate("generate", {"prompt": "a", "extra": 1})
    assert excinfo.value.field == "extra"


def test_validate_rejects_wrong_type() -> None:
    with pytest.raises(ContractError) as excinfo:
        contract().validate("generate", {"prompt": 5})
    assert excinfo.value.field == "prompt"


def test_validate_enforces_field_constraints() -> None:
    with pytest.raises(ContractError) as excinfo:
        contract().validate("set_prompt", {"prompt": ""})
    assert excinfo.value.field == "prompt"
    assert "min_length" in excinfo.value.reason


def test_validate_accepts_an_upload_reference() -> None:
    cmd = contract().validate("upload", {"file": {"upload_id": "u-1"}})
    assert isinstance(cmd, Command)
    # The reference is kept verbatim; the runtime fetches the bytes, not validate.
    assert vars(cmd)["file"] == {"upload_id": "u-1"}


def test_validate_coerces_an_enum_value_to_its_member() -> None:
    cmd = ModelContract.of(SpeedModel).validate("set_speed", {"speed": 2})
    assert vars(cmd)["speed"] is Speed.FAST


def test_override_that_drops_the_decorator_removes_the_command() -> None:
    class Base(ReactorModel):
        @event(name="go")
        async def go(self) -> None: ...

    class Derived(Base):
        async def go(self) -> None: ...  # overrides without the decorator

    assert "go" in ModelContract.of(Base).commands
    assert "go" not in ModelContract.of(Derived).commands


def test_duplicate_command_name_is_rejected_at_build() -> None:
    with pytest.raises(ValueError, match="duplicate command name"):

        class Bad(ReactorModel):
            @event(name="go")
            async def first(self) -> None: ...

            @event(name="go")
            async def second(self) -> None: ...


def test_a_track_declared_as_both_directions_is_rejected() -> None:
    class Dupe(Output):
        shared: Video

    class DupeIn(Input):
        shared: Video

    # Both directions register globally; the clash surfaces when the union is read.
    with pytest.raises(ValueError, match="both input and output"):
        _ = ModelContract.of(EchoModel).tracks
