import pytest

from reactor_runtime import InputField, InputState, UploadedFile


class State(InputState):
    required_axis: float
    speed: float = InputField(default=1.0, ge=0.0, le=10.0)
    label: str = InputField(default="idle")
    seed: int = 0
    image: UploadedFile = InputField(default=None)
    _started: bool = False
    _cache: int = 7


def test_fields_partition_by_name_and_type() -> None:
    assert set(State._public_fields) == {"required_axis", "speed", "label", "seed", "image"}
    assert State._private_fields == {"_started", "_cache"}
    assert State._upload_fields == {"image"}


def test_defaults_construct_a_fresh_instance() -> None:
    state = State(required_axis=0.0)
    assert state.speed == 1.0
    assert state.label == "idle"
    assert state.seed == 0
    assert state.image is None
    assert state._started is False
    assert state._cache == 7


def test_public_field_carries_its_constraints() -> None:
    info = State._public_fields["speed"]
    assert info.ge == 0.0
    assert info.le == 10.0


def test_a_field_without_a_default_is_required() -> None:
    with pytest.raises(TypeError):
        State()  # type: ignore[ty:missing-argument]  # omitting the required field is the case under test
    state = State(required_axis=2.5)
    assert state.required_axis == 2.5


def test_instances_are_independent() -> None:
    first = State(required_axis=0.0)
    second = State(required_axis=0.0)
    first.speed = 9.0
    assert second.speed == 1.0


def test_mutable_default_is_rejected_at_declaration() -> None:
    with pytest.raises(TypeError):

        class _Bad(InputState):
            items: list[int] = InputField(default=[1, 2])


def test_mutable_literal_default_is_rejected() -> None:
    with pytest.raises(TypeError):

        class _Bad(InputState):
            items: dict[str, int] = {"a": 1}  # noqa: RUF012 — the rejection under test
