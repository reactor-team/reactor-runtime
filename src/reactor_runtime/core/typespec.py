"""The one type system behind both validation and schema rendering.

A :class:`TypeSpec` is the normalised form of a single supported field
annotation, built once when the declaring class is created. Each spec answers
the only two questions the runtime asks of a field type: *is this value
acceptable?* (:meth:`TypeSpec.check`) and *how does this type read as JSON
Schema?* (:meth:`TypeSpec.to_json_schema`). Resolving an annotation into a spec
is itself the definition-time check that the type is supported at all —
:meth:`TypeSpec.of` raises on anything outside the supported set.

Keeping both answers behind one resolved form is the point: request-time
validation and the published schema are derived from the same traversal, so they
cannot drift.
"""

from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from reactor_runtime.core.model import UploadedFile

_NONE_TYPE = type(None)

_SUPPORTED = (
    "int, float, str, bool, Literal[...], T | None, list[T], dict[str, V], Enum, "
    "dataclass, UploadedFile"
)


class TypeSpec(ABC):
    """A resolved, validated representation of one supported field type."""

    @abstractmethod
    def check(self, value: Any) -> str | None:
        """Return ``None`` when *value* fits this type, else a failure reason."""

    def coerce(self, value: Any) -> Any:
        """Return *value* in this type's Python form, assuming it has passed :meth:`check`.

        The default returns the value untouched. Types whose wire form differs
        from their Python form — an enum's member value, say — override this so a
        validated payload yields a genuinely typed value.
        """
        return value

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        """Render this type as an OpenAPI 3.1 JSON Schema fragment."""

    @staticmethod
    def of(annotation: Any) -> TypeSpec:
        """Resolve a type annotation into its :class:`TypeSpec`.

        Args:
            annotation: A field's resolved type annotation.

        Returns:
            The spec describing *annotation*.

        Raises:
            TypeError: If *annotation* is not a supported field type.
        """
        if annotation is Any:
            return AnySpec()

        inner = _unwrap_optional(annotation)
        if inner is not None:
            return OptionalSpec(TypeSpec.of(inner))

        origin = get_origin(annotation)

        if origin is Literal:
            return LiteralSpec(get_args(annotation))

        if origin is list:
            args = get_args(annotation)
            return ListSpec(TypeSpec.of(args[0]) if args else AnySpec())

        if origin is dict:
            args = get_args(annotation)
            return DictSpec(TypeSpec.of(args[1]) if len(args) == 2 else AnySpec())

        if isinstance(annotation, type):
            return _spec_for_type(annotation)

        raise TypeError(f"unsupported field type {annotation!r}. Supported: {_SUPPORTED}.")


def _spec_for_type(annotation: type) -> TypeSpec:
    """Resolve a bare ``type`` annotation into its spec."""
    if annotation is bool:
        return PrimitiveSpec(bool, "boolean")
    if annotation is int:
        return PrimitiveSpec(int, "integer")
    if annotation is float:
        return PrimitiveSpec(float, "number")
    if annotation is str:
        return PrimitiveSpec(str, "string")
    if annotation is UploadedFile:
        return UploadSpec()
    if annotation is list:
        return ListSpec(AnySpec())
    if annotation is dict:
        return DictSpec(AnySpec())
    if issubclass(annotation, enum.Enum):
        return EnumSpec(annotation)
    if dataclasses.is_dataclass(annotation):
        return DataclassSpec.build(annotation)
    raise TypeError(f"unsupported field type {annotation!r}. Supported: {_SUPPORTED}.")


def _unwrap_optional(annotation: Any) -> Any | None:
    """Return the inner type of ``T | None``, or ``None`` when not optional.

    Only a two-member union with exactly one non-``None`` arm counts as optional;
    a wider union (``int | str``) is not, and resolving it later raises as an
    unsupported type.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        args = get_args(annotation)
        non_none = [arg for arg in args if arg is not _NONE_TYPE]
        if _NONE_TYPE in args and len(non_none) == 1:
            return non_none[0]
    return None


class AnySpec(TypeSpec):
    """An unconstrained field, accepting any value."""

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        return None

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        return {}


class PrimitiveSpec(TypeSpec):
    """A JSON scalar: ``int``, ``float``, ``str``, or ``bool``."""

    def __init__(self, py_type: type, json_type: str) -> None:
        self._py_type = py_type
        self._json_type = json_type

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        if self._py_type is bool:
            if isinstance(value, bool):
                return None
        elif self._py_type is int:
            # bool is never an int field value. The wire has no integer type — a
            # JSON number decodes to a float — so an integral float is accepted
            # and narrowed in coerce; a fractional float is not.
            if not isinstance(value, bool) and (
                isinstance(value, int) or (isinstance(value, float) and value.is_integer())
            ):
                return None
        elif self._py_type is float:
            # A JSON integer is an acceptable number; a bool is not.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return None
        elif isinstance(value, str):
            return None
        return f"expected {self._json_type}, got {type(value).__name__}"

    def coerce(self, value: Any) -> Any:  # noqa: D102 — contract on the base
        # An integral float arriving for an int field is narrowed to int; every
        # other primitive is already in its Python form.
        if self._py_type is int and isinstance(value, float):
            return int(value)
        return value

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        return {"type": self._json_type}


class LiteralSpec(TypeSpec):
    """A closed set of literal values from ``Literal[...]``."""

    def __init__(self, values: tuple[Any, ...]) -> None:
        self._values = values

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        if value in self._values:
            return None
        return f"{value!r} not one of {list(self._values)}"

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        return _enum_schema(list(self._values))


class EnumSpec(TypeSpec):
    """A Python :class:`enum.Enum`, matched against its member values."""

    def __init__(self, enum_cls: type[enum.Enum]) -> None:
        self._enum_cls = enum_cls
        self._members = [member.value for member in enum_cls]

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        if isinstance(value, self._enum_cls) or value in self._members:
            return None
        return f"{value!r} not a valid {self._enum_cls.__name__}"

    def coerce(self, value: Any) -> Any:  # noqa: D102 — contract on the base
        return self._enum_cls(value)

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        return _enum_schema(self._members)


class ListSpec(TypeSpec):
    """A homogeneous JSON array."""

    def __init__(self, item: TypeSpec) -> None:
        self._item = item

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        if not isinstance(value, list):
            return f"expected array, got {type(value).__name__}"
        for index, element in enumerate(value):
            reason = self._item.check(element)
            if reason is not None:
                return f"[{index}]: {reason}"
        return None

    def coerce(self, value: Any) -> Any:  # noqa: D102 — contract on the base
        return [self._item.coerce(element) for element in value]

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        return {"type": "array", "items": self._item.to_json_schema()}


class DictSpec(TypeSpec):
    """A JSON object with string keys and homogeneous values."""

    def __init__(self, value: TypeSpec) -> None:
        self._value = value

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        if not isinstance(value, Mapping):
            return f"expected object, got {type(value).__name__}"
        for key, element in value.items():
            if not isinstance(key, str):
                return f"key {key!r} is not a string"
            reason = self._value.check(element)
            if reason is not None:
                return f"[{key!r}]: {reason}"
        return None

    def coerce(self, value: Any) -> Any:  # noqa: D102 — contract on the base
        return {key: self._value.coerce(element) for key, element in value.items()}

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        return {"type": "object", "additionalProperties": self._value.to_json_schema()}


class DataclassSpec(TypeSpec):
    """A structured object, validated field by field against a dataclass."""

    def __init__(self, cls: type, fields: dict[str, TypeSpec], required: tuple[str, ...]) -> None:
        self._cls = cls
        self._fields = fields
        self._required = required

    @classmethod
    def build(cls, dataclass_type: type) -> DataclassSpec:
        """Resolve a dataclass into a spec over its fields."""
        try:
            hints = get_type_hints(dataclass_type)
        except Exception:
            hints = {field.name: field.type for field in dataclasses.fields(dataclass_type)}
        fields: dict[str, TypeSpec] = {}
        required: list[str] = []
        for field in dataclasses.fields(dataclass_type):
            fields[field.name] = TypeSpec.of(hints.get(field.name, field.type))
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                required.append(field.name)
        return cls(dataclass_type, fields, tuple(required))

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        if isinstance(value, self._cls):
            return None
        if not isinstance(value, Mapping):
            return f"expected object, got {type(value).__name__}"
        for name in self._required:
            if name not in value:
                return f"missing required field '{name}'"
        for name, element in value.items():
            spec = self._fields.get(name)
            if spec is None:
                continue
            reason = spec.check(element)
            if reason is not None:
                return f"{name}: {reason}"
        return None

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {name: spec.to_json_schema() for name, spec in self._fields.items()},
        }
        if self._required:
            schema["required"] = list(self._required)
        return schema


class OptionalSpec(TypeSpec):
    """A nullable wrapper around an inner type."""

    def __init__(self, inner: TypeSpec) -> None:
        self._inner = inner

    @property
    def inner(self) -> TypeSpec:
        """The wrapped non-null type."""
        return self._inner

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        if value is None:
            return None
        return self._inner.check(value)

    def coerce(self, value: Any) -> Any:  # noqa: D102 — contract on the base
        if value is None:
            return None
        return self._inner.coerce(value)

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        return {"anyOf": [self._inner.to_json_schema(), {"type": "null"}]}


class UploadSpec(TypeSpec):
    """A reference to a file uploaded through the Reactor upload protocol."""

    def check(self, value: Any) -> str | None:  # noqa: D102 — contract on the base
        if isinstance(value, UploadedFile):
            return None
        if isinstance(value, Mapping) and isinstance(value.get("upload_id"), str):
            return None
        return "expected an upload reference with a string 'upload_id'"

    def to_json_schema(self) -> dict[str, Any]:  # noqa: D102 — contract on the base
        return {"$ref": "#/components/schemas/ReactorUploadReference"}


def _enum_schema(values: list[Any]) -> dict[str, Any]:
    """Render a closed value set, keyed by JSON type when the members agree.

    A uniform set of ``int`` / ``float`` / ``bool`` / ``str`` members carries the
    matching ``type``; a mixed set emits a bare ``enum`` because no single
    ``type`` keyword could hold every member.
    """
    member_types = {type(value) for value in values}
    if member_types == {bool}:
        return {"type": "boolean", "enum": values}
    if member_types == {int}:
        return {"type": "integer", "enum": values}
    if member_types == {float}:
        return {"type": "number", "enum": values}
    if member_types == {str}:
        return {"type": "string", "enum": values}
    return {"enum": values}
