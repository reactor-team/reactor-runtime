import pytest
from fake_engine import Camera, FakeEngine, FakeInit, Move, StrictEngine, StrictInit

from reactor_runtime.core.fields import NO_DEFAULT
from reactor_runtime.core.values import TrackDirection, TrackKind
from reactor_runtime.engine_contract import Init, InputField, UserInput, VideoInput
from reactor_runtime.interface.engine.reflection import (
    command_for,
    default_init,
    discover_inputs,
    init_values,
    missing_init_fields,
    track_holder,
    wire_name,
)

# -- discovery -----------------------------------------------------------------


def test_an_engine_declares_events_media_and_an_init() -> None:
    inputs = discover_inputs(FakeEngine)

    assert inputs.events == {"move": Move}
    assert set(inputs.media) == {"camera"}
    assert inputs.init is FakeInit


def test_media_carries_the_chunk_size_it_declared() -> None:
    assert discover_inputs(FakeEngine).media["camera"].chunk_size == 2


def test_an_engine_without_an_init_declares_none() -> None:
    class Bare:
        declared_inputs = (Move,)

    assert discover_inputs(Bare).init is None


def test_two_declarations_claiming_one_wire_name_are_rejected() -> None:
    class Rival(UserInput):
        pass

    Rival.__name__ = "Move"

    class Clashing:
        declared_inputs = (Move, Rival)

    with pytest.raises(TypeError, match="wire name 'move'"):
        discover_inputs(Clashing)


def test_two_init_classes_are_rejected() -> None:
    class OtherInit(Init):
        pass

    class Confused:
        declared_inputs = (FakeInit, OtherInit)

    with pytest.raises(TypeError, match="two Init classes"):
        discover_inputs(Confused)


def test_declarations_are_found_by_package_when_the_engine_names_none() -> None:
    from scoped_engine import Jump, ScopedEngine

    discovered = discover_inputs(ScopedEngine)

    assert discovered.events == {"jump": Jump}
    assert set(discovered.media) == {"lens"}


# -- wire names ----------------------------------------------------------------


def test_a_wire_name_is_the_class_name_in_snake_case() -> None:
    class LookAround(UserInput):
        pass

    assert wire_name(LookAround) == "look_around"


def test_every_init_is_served_as_init_whatever_it_is_called() -> None:
    assert wire_name(FakeInit) == "init"
    assert wire_name(StrictInit) == "init"


# -- commands ------------------------------------------------------------------


def test_a_command_mirrors_the_inputs_fields() -> None:
    command = command_for("move", Move)

    assert set(command.__command_fields__) == {"direction", "speed"}
    assert command.name == "move"


def test_a_command_carries_the_inputs_constraints() -> None:
    info = command_for("move", Move).__command_fields__["speed"].info

    assert (info.ge, info.le) == (0.0, 8.0)


def test_a_required_input_field_becomes_a_required_argument() -> None:
    info = command_for("init", StrictInit).__command_fields__["prompt"].info

    assert info.default is NO_DEFAULT


def test_a_command_validates_a_client_payload() -> None:
    from reactor_runtime.core.typespec import TypeSpec

    field = command_for("move", Move).__command_fields__["speed"]
    assert TypeSpec.of(float).check(2.0) is None
    assert field.spec.check("fast") is not None


# -- tracks --------------------------------------------------------------------


def test_media_becomes_an_input_track() -> None:
    holder = track_holder("FakeInput", discover_inputs(FakeEngine).media)

    assert holder is not None
    info = holder.__tracks__["camera"]
    assert (info.kind, info.direction) == (TrackKind.VIDEO, TrackDirection.IN)


def test_an_audio_declaration_becomes_an_audio_track() -> None:
    from reactor_runtime.engine_contract import AudioInput
    from reactor_runtime.interface.engine.store import MediaSpec

    class Mic(AudioInput):
        pass

    holder = track_holder("Holder", {"mic": MediaSpec("mic", Mic, 1)})
    assert holder is not None
    assert holder.__tracks__["mic"].kind is TrackKind.AUDIO


def test_an_engine_with_no_media_declares_no_holder() -> None:
    assert track_holder("Holder", {}) is None


# -- initialization ------------------------------------------------------------


def test_an_init_of_defaults_can_be_fabricated() -> None:
    fabricated = default_init(FakeInit)

    assert fabricated is not None
    assert init_values(fabricated) == {"shade": 8}


def test_an_init_with_a_required_field_cannot_be_fabricated() -> None:
    assert default_init(StrictInit) is None
    assert missing_init_fields(StrictInit) == ["prompt"]


def test_an_engine_without_an_init_opens_a_rollout_with_no_arguments() -> None:
    assert default_init(None) is None
    assert missing_init_fields(None) == []


def test_the_strict_engine_declares_the_field_the_client_owns() -> None:
    assert discover_inputs(StrictEngine).init is StrictInit


def test_a_declared_chunked_video_reports_its_modality() -> None:
    class Batched(VideoInput):
        chunk_size = 4

    class Engine:
        declared_inputs = (Batched,)

    spec = discover_inputs(Engine).media["batched"]
    assert (spec.chunk_size, spec.input_cls.media_kind) == (4, "video")


def test_a_field_description_survives_into_the_command() -> None:
    class Described(UserInput):
        gain: float = InputField(default=1.0, description="How hard to push.")

    info = command_for("described", Described).__command_fields__["gain"].info
    assert info.description == "How hard to push."


def test_the_camera_declaration_is_not_also_a_command() -> None:
    assert Camera not in discover_inputs(FakeEngine).events.values()
