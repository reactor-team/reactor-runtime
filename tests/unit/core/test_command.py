import dataclasses
from typing import ClassVar, Literal

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
        Required()  # type: ignore[ty:missing-argument]


def test_fields_without_defaults_are_ordered_first() -> None:
    # Declared default-first; the hook reorders so the dataclass accepts it.
    class Mixed(Command):
        with_default: int = 3
        required: str  # type: ignore[ty:dataclass-field-order]  # runtime reorders it

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
            level: int = "hello"  # type: ignore[ty:invalid-assignment]


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


# -- inheritance -------------------------------------------------------------


class SetBrightnessOn(SetBrightness):
    channel: UploadedFile | None = None
    steps: int  # type: ignore[ty:dataclass-field-order]  # runtime reorders it


def test_a_subclass_inherits_every_field() -> None:
    assert list(SetBrightnessOn.__command_fields__) == ["level", "mode", "channel", "steps"]
    assert SetBrightnessOn.__command_fields__["level"].info.le == 1.0
    assert SetBrightnessOn.__upload_fields__ == frozenset({"channel"})
    assert [f.name for f in dataclasses.fields(SetBrightnessOn)] == [
        "level",
        "mode",
        "steps",
        "channel",
    ]


def test_a_subclass_constructs_with_inherited_and_added_fields() -> None:
    cmd = SetBrightnessOn(steps=3)
    assert (cmd.level, cmd.mode, cmd.channel, cmd.steps) == (1.0, "a", None, 3)
    with pytest.raises(TypeError):
        SetBrightnessOn()  # type: ignore[ty:missing-argument]


def test_a_subclass_has_its_own_name() -> None:
    assert SetBrightnessOn.name == "set_brightness_on"


def test_a_subclass_with_no_fields_of_its_own_inherits_everything() -> None:
    class Same(SetBrightness):
        pass

    assert list(Same.__command_fields__) == ["level", "mode"]
    assert Same().level == 1.0


def test_a_redeclared_field_replaces_the_parents() -> None:
    class Dimmer(SetBrightness):
        level: float = InputField(default=0.2, ge=0.0, le=0.5)

    assert list(Dimmer.__command_fields__) == ["level", "mode"]
    assert Dimmer.__command_fields__["level"].info.le == 0.5
    assert Dimmer().level == 0.2


def test_a_parent_default_redeclared_as_required_does_not_leak_through() -> None:
    class Strict(SetBrightness):
        level: float

    assert Strict.__command_fields__["level"].info.default is NO_DEFAULT
    with pytest.raises(TypeError):
        Strict()  # type: ignore[ty:missing-argument]


def test_an_upload_field_redeclared_as_plain_leaves_the_upload_set() -> None:
    class Plain(SetBrightnessOn):
        channel: str = "main"

    assert Plain.__upload_fields__ == frozenset()


def test_an_inherited_field_redeclared_as_a_classvar_leaves_the_record() -> None:
    class Pinned(SetBrightness):
        level: ClassVar[float] = 0.5  # type: ignore[ty:invalid-attribute-override]  # the redeclaration under test

    assert list(Pinned.__command_fields__) == ["mode"]
    assert [f.name for f in dataclasses.fields(Pinned)] == ["mode"]
    assert Pinned().level == 0.5
    with pytest.raises(TypeError):
        Pinned(level=0.9)  # the rejection under test


def test_a_classvar_redeclaration_survives_field_reordering() -> None:
    # A required and a defaulted field in the same subclass trigger the
    # annotation reordering; the ClassVar must stay in the annotations so the
    # dataclass masks the parent's field.
    class Mixed(SetBrightness):
        level: ClassVar[float] = 0.5  # type: ignore[ty:invalid-attribute-override]  # the redeclaration under test
        defaulted: int = 1
        required: str  # type: ignore[ty:dataclass-field-order]  # runtime reorders it

    assert list(Mixed.__command_fields__) == ["mode", "defaulted", "required"]
    assert "level" not in {f.name for f in dataclasses.fields(Mixed)}
    with pytest.raises(TypeError):
        Mixed(required="r", level=0.9)  # the rejection under test


def test_the_nearer_base_decides_whether_a_shared_name_is_an_upload() -> None:
    class AsText(Command):
        channel: str = "main"

    class TextWins(AsText, SetBrightnessOn):
        pass

    class UploadWins(SetBrightnessOn, AsText):
        pass

    assert TextWins.__upload_fields__ == frozenset()
    assert UploadWins.__upload_fields__ == frozenset({"channel"})
