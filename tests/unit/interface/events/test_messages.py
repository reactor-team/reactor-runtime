import dataclasses
from typing import ClassVar

import pytest

from reactor_runtime import MessageField, ModelMessage
from reactor_runtime.core.fields import NO_DEFAULT


class Progress(ModelMessage):
    """How far generation has got."""

    step: int = MessageField(description="Current step")
    total: int = MessageField(default=100, description="Total steps")


def test_subclass_constructs_like_a_dataclass() -> None:
    assert Progress(step=1).total == 100


def test_subclass_is_a_dataclass() -> None:
    assert dataclasses.is_dataclass(Progress)


def test_name_is_derived_from_the_class_name() -> None:
    assert Progress.name == "progress"


def test_user_docstring_is_snapshotted() -> None:
    assert Progress.user_doc == "How far generation has got."


def test_payload_less_message_serialises_with_empty_data() -> None:
    class Done(ModelMessage):
        pass

    assert Done().to_wire_format() == {"type": "done", "data": {}}


def test_message_without_docstring_keeps_user_doc_none() -> None:
    class CurrentMode(ModelMessage):
        mode: str

    # @dataclass synthesises a signature docstring, but the snapshot stays empty.
    assert CurrentMode.user_doc is None


def test_field_descriptions_are_resolved() -> None:
    fields = Progress.__message_fields__
    assert fields["step"].description == "Current step"
    assert fields["step"].spec.to_json_schema() == {"type": "integer"}


def test_to_wire_format_envelope() -> None:
    assert Progress(step=3, total=9).to_wire_format() == {
        "type": "progress",
        "data": {"step": 3, "total": 9},
    }


def test_required_field_has_no_default() -> None:
    class CurrentMode(ModelMessage):
        mode: str = MessageField(description="the mode")

    field = CurrentMode.__message_fields__["mode"]
    assert field.description == "the mode"
    assert MessageField().default is NO_DEFAULT


def test_message_field_rejects_default_factory() -> None:
    with pytest.raises(TypeError, match="default_factory"):
        MessageField(default_factory=list)


def test_raw_dataclass_field_is_rejected() -> None:
    with pytest.raises(TypeError, match="dataclasses"):

        class Bad(ModelMessage):
            items: list[str] = dataclasses.field(default_factory=list)


def test_type_mismatched_default_raises_at_definition_time() -> None:
    with pytest.raises(TypeError, match="does not match its type"):

        class Bad(ModelMessage):
            count: int = MessageField(default="x")


# -- inheritance -------------------------------------------------------------


class Detailed(Progress):
    """Progress with a stage name."""

    stage: str = MessageField(description="Current stage")


def test_a_subclass_inherits_every_field() -> None:
    assert list(Detailed.__message_fields__) == ["step", "total", "stage"]
    assert Detailed.__message_fields__["total"].default == 100
    assert [f.name for f in dataclasses.fields(Detailed)] == ["step", "total", "stage"]


def test_a_subclass_constructs_with_inherited_and_added_fields() -> None:
    message = Detailed(step=1, stage="decode")
    assert message.total == 100
    assert message.to_wire_format() == {
        "type": "detailed",
        "data": {"step": 1, "total": 100, "stage": "decode"},
    }


def test_a_required_field_may_follow_an_inherited_default() -> None:
    with pytest.raises(TypeError):
        Detailed(step=1)  # type: ignore[ty:missing-argument]


def test_a_subclass_keeps_its_own_name_and_docstring() -> None:
    assert Detailed.name == "detailed"
    assert Detailed.user_doc == "Progress with a stage name."
    assert Progress.user_doc == "How far generation has got."


def test_a_subclass_with_no_fields_of_its_own_inherits_everything() -> None:
    class Same(Progress):
        pass

    assert list(Same.__message_fields__) == ["step", "total"]
    assert Same(step=2).to_wire_format()["data"] == {"step": 2, "total": 100}


def test_a_redeclared_field_replaces_the_parents() -> None:
    class Shorter(Progress):
        total: int = MessageField(default=10, description="Fewer steps")

    assert list(Shorter.__message_fields__) == ["step", "total"]
    assert Shorter.__message_fields__["total"].description == "Fewer steps"
    assert Shorter(step=1).total == 10


def test_a_parent_default_redeclared_as_required_does_not_leak_through() -> None:
    class Strict(Progress):
        total: int

    assert Strict.__message_fields__["total"].default is NO_DEFAULT
    with pytest.raises(TypeError):
        Strict(step=1)  # type: ignore[ty:missing-argument]
    assert Strict(step=1, total=5).total == 5


def test_an_inherited_field_redeclared_as_a_classvar_leaves_the_record() -> None:
    class Fixed(Progress):
        total: ClassVar[int] = 10  # type: ignore[ty:invalid-attribute-override]  # the redeclaration under test

    assert list(Fixed.__message_fields__) == ["step"]
    assert Fixed(step=1).to_wire_format()["data"] == {"step": 1}
    with pytest.raises(TypeError):
        Fixed(step=1, total=5)  # the rejection under test


def test_a_root_message_is_declared_positionally() -> None:
    assert Progress(4).step == 4
