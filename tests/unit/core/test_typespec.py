import enum
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from reactor_runtime.core import UploadedFile
from reactor_runtime.core.typespec import (
    AnySpec,
    DataclassSpec,
    DictSpec,
    EnumSpec,
    ListSpec,
    LiteralSpec,
    OptionalSpec,
    PrimitiveSpec,
    TypeSpec,
    UploadSpec,
)


class Color(enum.StrEnum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Point:
    x: int
    y: int = 0


@pytest.mark.parametrize(
    ("annotation", "spec_type"),
    [
        (int, PrimitiveSpec),
        (str, PrimitiveSpec),
        (bool, PrimitiveSpec),
        (float, PrimitiveSpec),
        (Literal["a", "b"], LiteralSpec),
        (list[int], ListSpec),
        (dict[str, int], DictSpec),
        (Color, EnumSpec),
        (Point, DataclassSpec),
        (UploadedFile, UploadSpec),
        (int | None, OptionalSpec),
        (Any, AnySpec),
    ],
)
def test_of_resolves_supported_types(annotation: object, spec_type: type) -> None:
    assert isinstance(TypeSpec.of(annotation), spec_type)


def test_of_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="unsupported field type"):
        TypeSpec.of(complex)


def test_of_rejects_wide_union() -> None:
    with pytest.raises(TypeError):
        TypeSpec.of(int | str)


def test_primitive_int_rejects_bool() -> None:
    spec = TypeSpec.of(int)
    assert spec.check(3) is None
    assert spec.check(True) is not None
    assert spec.check("3") is not None


def test_primitive_int_accepts_and_narrows_integral_float() -> None:
    # JSON / protobuf Struct carry every number as a float, so an int field must
    # accept an integral float and narrow it to int.
    spec = TypeSpec.of(int)
    assert spec.check(107.0) is None
    assert spec.coerce(107.0) == 107
    assert isinstance(spec.coerce(107.0), int)


def test_primitive_int_rejects_fractional_float() -> None:
    spec = TypeSpec.of(int)
    assert spec.check(107.5) is not None


def test_primitive_float_accepts_int_not_bool() -> None:
    spec = TypeSpec.of(float)
    assert spec.check(1) is None
    assert spec.check(1.5) is None
    assert spec.check(True) is not None


def test_literal_membership() -> None:
    spec = TypeSpec.of(Literal["a", "b"])
    assert spec.check("a") is None
    assert spec.check("c") is not None


def test_enum_accepts_member_and_value() -> None:
    spec = TypeSpec.of(Color)
    assert spec.check("red") is None
    assert spec.check(Color.BLUE) is None
    assert spec.check("green") is not None


def test_list_validates_elements_with_index() -> None:
    spec = TypeSpec.of(list[int])
    assert spec.check([1, 2]) is None
    assert spec.check("nope") is not None
    reason = spec.check([1, "x"])
    assert reason is not None
    assert "[1]" in reason


def test_dict_requires_string_keys_and_checks_values() -> None:
    spec = TypeSpec.of(dict[str, int])
    assert spec.check({"a": 1}) is None
    assert spec.check({"a": "x"}) is not None


def test_dataclass_checks_required_and_known_fields() -> None:
    spec = TypeSpec.of(Point)
    assert spec.check({"x": 1, "y": 2}) is None
    assert spec.check({"x": 1}) is None  # y has a default
    assert spec.check({"y": 2}) is not None  # x is required
    assert spec.check(Point(x=1)) is None


def test_optional_accepts_none() -> None:
    spec = TypeSpec.of(int | None)
    assert spec.check(None) is None
    assert spec.check(3) is None
    assert spec.check("x") is not None


def test_upload_accepts_reference_or_instance() -> None:
    spec = TypeSpec.of(UploadedFile)
    assert spec.check({"upload_id": "u-1"}) is None
    assert spec.check(UploadedFile(name="n", mime_type="t", data=b"")) is None
    assert spec.check({"name": "x"}) is not None


def test_schema_primitive() -> None:
    assert TypeSpec.of(int).to_json_schema() == {"type": "integer"}
    assert TypeSpec.of(str).to_json_schema() == {"type": "string"}


def test_schema_optional_is_any_of_null() -> None:
    schema = TypeSpec.of(str | None).to_json_schema()
    assert schema == {"anyOf": [{"type": "string"}, {"type": "null"}]}


def test_schema_list_and_dict() -> None:
    assert TypeSpec.of(list[int]).to_json_schema() == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert TypeSpec.of(dict[str, str]).to_json_schema() == {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }


def test_schema_enum_keys_by_member_type() -> None:
    assert TypeSpec.of(Color).to_json_schema() == {
        "type": "string",
        "enum": ["red", "blue"],
    }


def test_schema_literal_mixed_drops_type() -> None:
    schema = TypeSpec.of(Literal[1, "a"]).to_json_schema()
    assert schema == {"enum": [1, "a"]}


def test_schema_upload_is_a_ref() -> None:
    assert TypeSpec.of(UploadedFile).to_json_schema() == {
        "$ref": "#/components/schemas/ReactorUploadReference"
    }


def test_schema_dataclass_lists_required() -> None:
    schema = TypeSpec.of(Point).to_json_schema()
    assert schema["type"] == "object"
    assert schema["properties"]["x"] == {"type": "integer"}
    assert schema["required"] == ["x"]


def test_coerce_is_identity_for_primitives() -> None:
    assert TypeSpec.of(int).coerce(3) == 3
    assert TypeSpec.of(str).coerce("x") == "x"


def test_coerce_turns_an_enum_value_into_a_member() -> None:
    spec = TypeSpec.of(Color)
    coerced = spec.coerce("red")
    assert coerced is Color.RED
    assert spec.coerce(Color.BLUE) is Color.BLUE


def test_coerce_passes_through_optional_none() -> None:
    spec = TypeSpec.of(Color | None)
    assert spec.coerce(None) is None
    assert spec.coerce("red") is Color.RED


def test_coerce_recurses_into_list_and_dict() -> None:
    assert TypeSpec.of(list[Color]).coerce(["red", "blue"]) == [Color.RED, Color.BLUE]
    assert TypeSpec.of(dict[str, Color]).coerce({"a": "red"}) == {"a": Color.RED}
