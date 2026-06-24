import dataclasses

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
