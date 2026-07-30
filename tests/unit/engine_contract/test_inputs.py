from typing import Any, ClassVar, Literal

import pytest

from reactor_runtime.engine_contract import (
    NO_DEFAULT,
    USER_INPUT_REGISTRY,
    AudioInput,
    Init,
    InputField,
    ModelInput,
    UserInput,
    VideoInput,
)


class Move(UserInput):
    direction: Literal["left", "right"]
    speed: float = InputField(default=1.0, ge=0.0, le=4.0, description="Steps per second.")


# -- field declaration ---------------------------------------------------------


def test_annotated_fields_become_the_payload() -> None:
    assert list(Move.__input_fields__) == ["direction", "speed"]
    assert Move.__input_annotations__["direction"] == Literal["left", "right"]


def test_a_field_without_a_default_is_required() -> None:
    assert Move.__input_fields__["direction"].required
    assert not Move.__input_fields__["speed"].required


def test_input_field_carries_its_constraints() -> None:
    spec = Move.__input_fields__["speed"]
    assert (spec.ge, spec.le) == (0.0, 4.0)
    assert spec.description == "Steps per second."


def test_a_plain_default_declares_an_unconstrained_field() -> None:
    class Zoom(UserInput):
        factor: float = 2.0

    assert Zoom.__input_fields__["factor"].default == 2.0
    assert Zoom.__input_fields__["factor"].ge is None


def test_required_fields_are_ordered_first() -> None:
    class Mixed(UserInput):
        optional: int = 1
        required: str

    assert list(Mixed.__input_fields__) == ["required", "optional"]


def test_a_class_variable_is_not_a_field() -> None:
    class Tuned(UserInput):
        limit: ClassVar[int] = 3
        value: int = 0

    assert list(Tuned.__input_fields__) == ["value"]
    assert Tuned.limit == 3


def test_a_mutable_default_is_rejected_at_declaration() -> None:
    with pytest.raises(TypeError, match="mutable list"):

        class Bad(UserInput):
            items: list[int] = InputField(default=[])


def test_a_subclass_inherits_its_bases_fields() -> None:
    class Sprint(Move):
        boost: float = 2.0

    assert list(Sprint.__input_fields__) == ["direction", "speed", "boost"]


# -- construction --------------------------------------------------------------


def test_construction_applies_declared_defaults() -> None:
    move = Move(direction="left")
    assert (move.direction, move.speed) == ("left", 1.0)


def test_a_missing_required_field_is_rejected() -> None:
    with pytest.raises(TypeError, match="missing required field"):
        Move()  # ty: ignore[missing-argument]


def test_an_undeclared_field_is_rejected() -> None:
    with pytest.raises(TypeError, match="has no field"):
        Move(direction="left", altitude=3)  # ty: ignore[unknown-argument]


def test_the_field_spec_never_leaks_as_a_value() -> None:
    # `speed: float = InputField(...)` leaves the spec bound to the class; an
    # instance must read the default it declared, not the spec object.
    assert Move(direction="left").speed == 1.0
    assert not hasattr(Move, "speed")


def test_timestamp_starts_unstamped_and_is_the_runtimes_to_set() -> None:
    move = Move(direction="left")
    assert move.timestamp_ms == 0
    move.timestamp_ms = 1234
    assert move.timestamp_ms == 1234


def test_equality_compares_the_declared_fields() -> None:
    assert Move(direction="left") == Move(direction="left", speed=1.0)
    assert Move(direction="left") != Move(direction="right")


# -- media ---------------------------------------------------------------------


def test_a_media_subclass_declares_its_modality_and_default_chunk() -> None:
    class Webcam(VideoInput):
        pass

    class Microphone(AudioInput):
        pass

    assert Webcam.media_kind == "video"
    assert Microphone.media_kind == "audio"
    assert Webcam.chunk_size == 1


def test_a_media_subclass_may_batch_frames() -> None:
    class Batched(VideoInput):
        chunk_size = 16

    assert Batched.chunk_size == 16
    assert Batched.__input_fields__ == {}


def test_a_media_subclass_cannot_declare_wire_fields() -> None:
    with pytest.raises(TypeError, match="declares wire field"):

        class Bad(VideoInput):
            gain: float = 1.0


def test_a_media_chunk_size_below_one_is_rejected() -> None:
    with pytest.raises(TypeError, match="at least 1"):

        class Bad(VideoInput):
            chunk_size = 0


def test_media_payload_starts_empty_for_the_runtime_to_fill() -> None:
    class Webcam(VideoInput):
        pass

    frame = Webcam()
    assert (frame.data, frame.pts) == (None, None)


# -- initialization ------------------------------------------------------------


def test_an_init_is_a_user_input_carrying_the_rollouts_starting_state() -> None:
    class WalkInit(Init):
        prompt: str = "a misty forest"
        seed: int = InputField(default=0, ge=0)

    assert issubclass(WalkInit, UserInput)
    assert WalkInit().prompt == "a misty forest"


def test_an_init_field_without_a_default_is_the_clients_to_supply() -> None:
    class StrictInit(Init):
        prompt: str

    assert StrictInit.__input_fields__["prompt"].default is NO_DEFAULT


# -- step input ----------------------------------------------------------------


def test_a_model_input_is_a_plain_dataclass() -> None:
    class StepInput(ModelInput):
        trajectory: Any
        scale: float = 1.0

    step = StepInput(trajectory=[1, 2])
    assert (step.trajectory, step.scale) == ([1, 2], 1.0)


# -- discovery -----------------------------------------------------------------


def test_declaring_a_subclass_registers_it() -> None:
    assert USER_INPUT_REGISTRY[f"{Move.__module__}.Move"] is Move


def test_the_contracts_own_bases_are_not_registered() -> None:
    registered = set(USER_INPUT_REGISTRY.values())
    assert registered.isdisjoint({UserInput, VideoInput, AudioInput, Init})
