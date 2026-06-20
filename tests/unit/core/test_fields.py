import pytest

from reactor_runtime.core.fields import (
    NO_DEFAULT,
    FieldInfo,
    InputField,
    raise_if_default_invalid,
    raise_if_default_not_static,
    validate_field,
)


def test_input_field_carries_constraints() -> None:
    info = InputField(default=1.0, ge=0.0, le=2.0, description="a level")
    assert isinstance(info, FieldInfo)
    assert info.default == 1.0
    assert info.ge == 0.0
    assert info.le == 2.0
    assert info.description == "a level"
    assert info.moderate is True


def test_input_field_without_default_is_required() -> None:
    assert InputField().default is NO_DEFAULT


def test_input_field_rejects_default_factory() -> None:
    with pytest.raises(TypeError, match="default_factory"):
        InputField(default_factory=list)


@pytest.mark.parametrize(
    ("value", "ok"),
    [(1, True), (5, True), (-1, False), (6, False)],
)
def test_validate_field_bounds(value: int, ok: bool) -> None:
    info = InputField(ge=0, le=5)
    passed, _ = validate_field("x", value, info)
    assert passed is ok


def test_validate_field_choices() -> None:
    info = InputField(choices=["a", "b"])
    assert validate_field("x", "a", info)[0] is True
    passed, reason = validate_field("x", "z", info)
    assert passed is False
    assert "choices" in reason


def test_validate_field_length() -> None:
    info = InputField(min_length=2, max_length=4)
    assert validate_field("x", "abc", info)[0] is True
    assert validate_field("x", "a", info)[0] is False
    assert validate_field("x", "abcde", info)[0] is False


def test_validate_field_incomparable_value_is_a_reason_not_a_crash() -> None:
    passed, reason = validate_field("x", object(), InputField(ge=0))
    assert passed is False
    assert "comparable" in reason


@pytest.mark.parametrize("mutable", [[], {}, set()])
def test_raise_if_default_not_static_rejects_mutable(mutable: object) -> None:
    with pytest.raises(TypeError, match="mutable"):
        raise_if_default_not_static("Owner", "field", mutable)


def test_raise_if_default_not_static_allows_none_and_immutable() -> None:
    raise_if_default_not_static("Owner", "field", None)
    raise_if_default_not_static("Owner", "field", (1, 2))


def test_raise_if_default_invalid_checks_own_constraints() -> None:
    with pytest.raises(TypeError, match="constraints"):
        raise_if_default_invalid("Owner", "level", 9, InputField(default=9, le=1))


def test_raise_if_default_invalid_bypasses_sentinel_and_none() -> None:
    raise_if_default_invalid("Owner", "x", NO_DEFAULT, InputField(le=1))
    raise_if_default_invalid("Owner", "x", None, InputField(le=1))
