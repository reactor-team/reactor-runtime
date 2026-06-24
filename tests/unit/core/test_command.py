import dataclasses
from typing import Literal

import pytest

from reactor_runtime.core import Command, InputField, UploadedFile
from reactor_runtime.core.fields import NO_DEFAULT


class SetBrightness(Command):
    level: float = InputField(default=1.0, ge=0.0, le=1.0)
    mode: Literal["a", "b"] = "a"


def test_subclass_constructs_like_a_dataclass() -> None:
    cmd = SetBrightness(level=0.5)
    assert cmd.level == 0.5
    assert cmd.mode == "a"


def test_subclass_is_a_dataclass() -> None:
    assert dataclasses.is_dataclass(SetBrightness)


def test_name_is_derived_from_the_class_name() -> None:
    assert SetBrightness.name == "set_brightness"


def test_command_fields_are_resolved_and_cached() -> None:
    fields = SetBrightness.__command_fields__
    assert set(fields) == {"level", "mode"}
    assert fields["level"].info.ge == 0.0
    assert fields["level"].spec.check(0.5) is None
    assert fields["level"].spec.check(2.0) is None  # spec is type-only; bounds live on info


def test_input_field_default_is_unwrapped_for_the_dataclass() -> None:
    # The dataclass sees a plain float default, never the FieldInfo wrapper.
    field = next(f for f in dataclasses.fields(SetBrightness) if f.name == "level")
    assert field.default == 1.0


def test_required_field_has_no_default_info() -> None:
    class Required(Command):
        prompt: str

    assert Required.__command_fields__["prompt"].info.default is NO_DEFAULT
    with pytest.raises(TypeError):
        Required()  # type: ignore[call-arg]


def test_fields_without_defaults_are_ordered_first() -> None:
    # Declared default-first; the hook reorders so the dataclass accepts it.
    class Mixed(Command):
        with_default: int = 3
        required: str  # type: ignore[misc]  # mypy can't see the runtime reorder

    cmd = Mixed(required="hi")
    assert cmd.required == "hi"
    assert cmd.with_default == 3


def test_upload_fields_are_detected() -> None:
    class Upload(Command):
        file: UploadedFile
        maybe: UploadedFile | None = None
        note: str = "n"

    assert Upload.__upload_fields__ == frozenset({"file", "maybe"})


def test_unsupported_field_type_raises_at_definition_time() -> None:
    with pytest.raises(TypeError, match="unsupported field type"):

        class Bad(Command):
            value: complex


def test_out_of_range_default_raises_at_definition_time() -> None:
    with pytest.raises(TypeError, match="constraints"):

        class Bad(Command):
            level: int = InputField(default=9, le=1)


def test_type_mismatched_static_default_raises_at_definition_time() -> None:
    with pytest.raises(TypeError, match="does not match its type"):

        class Bad(Command):
            level: int = "hello"  # type: ignore[assignment]


def test_type_mismatched_input_field_default_raises_at_definition_time() -> None:
    with pytest.raises(TypeError, match="does not match its type"):

        class Bad(Command):
            count: int = InputField(default="x")


def test_raw_dataclass_field_is_rejected() -> None:
    with pytest.raises(TypeError, match="dataclasses"):

        class Bad(Command):
            items: list[str] = dataclasses.field(default_factory=list)


def test_mutable_static_default_is_rejected() -> None:
    with pytest.raises(TypeError, match="mutable"):

        class Bad(Command):
            items: list[str] = ["a"]  # noqa: RUF012 — the test asserts this is rejected
