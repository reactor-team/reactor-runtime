import enum
from collections.abc import Callable
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from reactor_runtime import ModelMessage as LateReply


class Reply(ModelMessage):
    """The generated reply."""

    image_url: str


class OutTracks(Output):
    main_video: Video


class InTracks(Input):
    camera: Video


class EchoModel(ReactorModel):
    """A tiny echo model."""

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


def test_a_plain_override_keeps_the_inherited_command_and_base_handler() -> None:
    class Base(ReactorModel):
        @event(name="go")
        async def go(self) -> None: ...

    class Derived(Base):
        async def go(self) -> None: ...  # plain override is not an override

    # A plain (undecorated) override neither un-declares the command nor rebinds
    # it: the command stands and runs the base method. Overriding requires
    # re-applying @event — see test_redeclaring_with_event_overrides_the_inherited_command.
    assert "go" in ModelContract.of(Base).commands
    derived = ModelContract.of(Derived).commands["go"]
    assert derived.handler is Base.__dict__["go"]
    assert derived.handler is not Derived.__dict__["go"]


def test_redeclaring_with_event_overrides_the_inherited_command() -> None:
    class Base(ReactorModel):
        @event(name="go", description="base")
        async def go(self, x: int = 0) -> None: ...

    class Derived(Base):
        @event(name="go", description="derived")
        async def go(self, y: str = "a") -> None: ...

    # Re-applying @event overrides both the wire contract and the bound handler.
    spec = ModelContract.of(Derived).commands["go"]
    assert spec.description == "derived"
    assert "y" in spec.command.__command_fields__
    assert "x" not in spec.command.__command_fields__
    assert spec.handler is Derived.__dict__["go"]


def test_duplicate_command_name_is_rejected_at_build() -> None:
    with pytest.raises(ValueError, match="duplicate command name"):

        class Bad(ReactorModel):
            @event(name="go")
            async def first(self) -> None: ...

            @event(name="go")
            async def second(self) -> None: ...


# -- return annotations -------------------------------------------------------


def test_a_message_return_annotation_becomes_the_response() -> None:
    class Typed(ReactorModel):
        @event(name="go")
        async def go(self) -> Reply:
            return Reply(image_url="x")

    assert ModelContract.of(Typed).commands["go"].response is Reply


def test_a_none_return_annotation_has_no_response() -> None:
    class Void(ReactorModel):
        @event(name="go")
        async def go(self) -> None: ...

    assert ModelContract.of(Void).commands["go"].response is None


def test_an_unannotated_handler_has_no_response() -> None:
    # An absent annotation claims no response shape, so there is nothing for the
    # schema and the wire to disagree about. Only a stated one is held to.
    class Void(ReactorModel):
        @event(name="go")
        async def go(self): ...

    assert ModelContract.of(Void).commands["go"].response is None


def test_an_unresolvable_parameter_annotation_does_not_fail_the_model() -> None:
    # Only the return annotation is resolved here. A parameter is read where the
    # command is built, which falls back to Any, so adding a return annotation to a
    # handler that imports today cannot turn it into an import failure.
    class Late(ReactorModel):
        @event(name="go")
        async def go(self, subject: "LateReply") -> None: ...

    assert ModelContract.of(Late).commands["go"].response is None


def test_a_plain_return_annotation_is_rejected_at_build() -> None:
    # Only a ModelMessage reaches a client, so a dict is a reply the model states
    # and cannot deliver. The model fails to import rather than serve that contract.
    with pytest.raises(TypeError, match="which a client cannot receive"):

        class Bad(ReactorModel):
            @event(name="go")
            async def go(self) -> dict[str, int]:
                return {"count": 1}


def test_a_union_return_annotation_is_rejected_at_build() -> None:
    class Other(ModelMessage):
        """A second reply."""

        detail: str

    with pytest.raises(TypeError, match="no single response shape"):

        class Bad(ReactorModel):
            @event(name="go")
            async def go(self) -> Reply | Other:
                return Reply(image_url="x")


def test_an_optional_return_annotation_is_rejected_at_build() -> None:
    # A handler that can return None has no single response shape to publish, so
    # the author picks one.
    with pytest.raises(TypeError, match="no single response shape"):

        class Bad(ReactorModel):
            @event(name="go")
            async def go(self) -> Reply | None:
                return None


def test_a_return_annotation_that_does_not_resolve_is_rejected_at_build() -> None:
    # LateReply exists only for the type checker, so the contract cannot read the
    # response type at import time and must not fall back to "no response".
    with pytest.raises(TypeError, match="Cannot resolve the return annotation"):

        class Bad(ReactorModel):
            @event(name="go")
            async def go(self) -> "LateReply":
                raise NotImplementedError


def test_a_track_declared_as_both_directions_is_rejected() -> None:
    class Dupe(Output):
        shared: Video

    class DupeIn(Input):
        shared: Video

    # Both directions register globally; the clash surfaces when the union is read.
    with pytest.raises(ValueError, match="both input and output"):
        _ = ModelContract.of(EchoModel).tracks
