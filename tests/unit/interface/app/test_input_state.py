import dataclasses

import pytest

from reactor_runtime import InputField, InputState, UploadedFile
from reactor_runtime.core.fields import NO_DEFAULT


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


def test_a_private_field_without_a_default_is_rejected() -> None:
    with pytest.raises(TypeError, match="private field '_cache' needs a default"):

        class _Bad(InputState):
            speed: float = 1.0
            _cache: int  # type: ignore[ty:dataclass-field-order]  # the rejection under test


# -- inheritance -------------------------------------------------------------


class BaseState(InputState):
    paused: bool = InputField(default=False)
    seed: int = InputField(default=42, ge=0)
    image: UploadedFile = InputField(default=None)
    _cache: int = 7


class FamilyState(BaseState):
    keys: str = InputField(default="")


def test_a_subclass_inherits_every_field() -> None:
    assert list(FamilyState._public_fields) == ["paused", "seed", "image", "keys"]
    assert FamilyState._private_fields == {"_cache"}
    assert FamilyState._upload_fields == {"image"}
    assert [f.name for f in dataclasses.fields(FamilyState)] == [
        "paused",
        "seed",
        "image",
        "_cache",
        "keys",
    ]
    state = FamilyState()
    assert (state.paused, state.seed, state.image, state._cache, state.keys) == (
        False,
        42,
        None,
        7,
        "",
    )


def test_an_inherited_field_keeps_its_constraints() -> None:
    assert FamilyState._public_fields["seed"].ge == 0


def test_a_subclass_with_no_fields_of_its_own_inherits_everything() -> None:
    class Same(BaseState):
        pass

    assert list(Same._public_fields) == ["paused", "seed", "image"]
    assert Same().seed == 42


def test_a_required_field_added_by_a_subclass_is_required() -> None:
    class Needs(BaseState):
        prompt: str

    assert Needs._public_fields["prompt"].default is NO_DEFAULT
    with pytest.raises(TypeError):
        Needs()  # type: ignore[ty:missing-argument]
    assert Needs(prompt="hi").prompt == "hi"


def test_a_redeclared_field_replaces_the_parents() -> None:
    class Stricter(BaseState):
        seed: int = InputField(default=3, ge=1, le=10)

    assert list(Stricter._public_fields) == ["paused", "seed", "image"]
    assert Stricter._public_fields["seed"].le == 10
    assert Stricter().seed == 3


def test_a_parent_default_redeclared_as_required_does_not_leak_through() -> None:
    class Strict(BaseState):
        seed: int = InputField(ge=0)

    with pytest.raises(TypeError):
        Strict()  # type: ignore[ty:missing-argument]
    assert Strict(seed=5).seed == 5


def test_a_redeclared_private_field_replaces_the_parents_default() -> None:
    class Warmer(BaseState):
        _cache: int = 11

    assert Warmer()._cache == 11
    with pytest.raises(TypeError, match="private field '_cache' needs a default"):

        class _Bad(BaseState):
            _cache: int


def test_an_upload_field_redeclared_as_plain_leaves_the_upload_set() -> None:
    class Plain(BaseState):
        image: str = "none"

    assert Plain._upload_fields == set()
    assert Plain().image == "none"


def test_two_declared_parents_both_contribute() -> None:
    class Other(InputState):
        speed: float = 1.0

    class Both(BaseState, Other):
        pass

    assert set(Both._public_fields) == {"paused", "seed", "image", "speed"}
    assert Both().speed == 1.0


def test_the_nearer_base_decides_whether_a_shared_name_is_an_upload() -> None:
    class AsText(InputState):
        image: str = "none"

    class TextWins(AsText, BaseState):
        pass

    class UploadWins(BaseState, AsText):
        pass

    assert TextWins._upload_fields == set()
    assert TextWins().image == "none"
    assert UploadWins._upload_fields == {"image"}
    assert UploadWins().image is None


def test_a_plain_mixin_contributes_no_fields() -> None:
    class Mixin:
        helper: int = 1

    class WithMixin(Mixin, BaseState):
        pass

    assert "helper" not in WithMixin._public_fields
    assert "helper" not in {f.name for f in dataclasses.fields(WithMixin)}


def test_a_root_state_class_is_declared_positionally() -> None:
    # No parent fields, so the constructor is the one a plain dataclass gives.
    assert State(1.5).required_axis == 1.5
