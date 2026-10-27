"""The typed values the two halves exchange: frozen dataclasses of data, checked strictly.

A request type, a result type, a signal, and every control message are declared
as frozen dataclasses whose fields are data only. :func:`validate` checks a
declaration once (at class definition, at load, at connect) and returns its
:class:`Schema`; :func:`dump` turns an instance into plain values, and
:func:`load` builds an instance back from plain values, driven by the
annotations of the type the receiver declared itself. The wire never names a
type: only the fields travel, so nothing is imported or looked up by name, and
:func:`load` runs no code of the class (no ``__init__``, no ``__post_init__``).

What a declaration may hold:

* frozen, a dataclass in every base, with no ``__post_init__``, no
  ``__init__`` or ``__new__`` of its own, and no ``InitVar``;
* every field required (no default) and in ``__init__``;
* field types from ``bool``, ``int``, ``float``, ``str``, ``bytes``,
  ``numpy.ndarray``, ``None``, another declaration like this one,
  ``list[T]``, ``tuple[T, ...]``, ``tuple[T1, T2, ...]``, ``dict[str, T]``,
  ``dict[int, T]``, and ``T | None``.

The checks are exact: a ``bool`` is not an ``int``, an ``int`` is not a
``bool``, a missing or an extra field is refused. A ``float`` field takes an
``int`` when dumped (and holds it as a ``float``); a numpy scalar is taken as
the Python value it holds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import types
import typing
from dataclasses import dataclass
from typing import Any, Union

import numpy as np

__all__ = [
    "Schema",
    "SchemaError",
    "contract_text",
    "describe",
    "dump",
    "fingerprint",
    "holds_buffers",
    "load",
    "validate",
]


class SchemaError(TypeError):
    """A declaration that is not data only, or a value that does not match its declaration."""


@dataclass(frozen=True)
class _Field:
    name: str
    kind: _Type


@dataclass(frozen=True)
class _Type:
    """One resolved annotation."""

    # bool, int, float, str, bytes, array, none, record, list, tuple, vartuple,
    # dict, or optional
    tag: str
    args: tuple[Any, ...] = ()

    def text(self) -> str:
        if self.tag == "record":
            return self.args[0].name
        if self.tag == "optional":
            return f"{self.args[0].text()} | None"
        if self.tag == "list":
            return f"list[{self.args[0].text()}]"
        if self.tag == "vartuple":
            return f"tuple[{self.args[0].text()}, ...]"
        if self.tag == "tuple":
            return f"tuple[{', '.join(a.text() for a in self.args)}]"
        if self.tag == "dict":
            return f"dict[{self.args[0]}, {self.args[1].text()}]"
        return {"array": "ndarray", "none": "None"}.get(self.tag, self.tag)


@dataclass(frozen=True, eq=False)
class Schema:
    """A validated declaration: its class, its fields in order, and a stable description.

    Attributes:
        cls: The declared class.
        name: Its qualified name, for messages and the description.
        fields: The fields, in declaration order.
    """

    cls: type
    name: str
    fields: tuple[_Field, ...]

    def text(self) -> str:
        """A stable, readable description of the fields, nested declarations included."""
        inner = ", ".join(f"{f.name}: {f.kind.text()}" for f in self.fields)
        return f"{self.name}({inner})"


_CACHE: dict[type, Schema] = {}
_SCALARS: dict[Any, str] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    bytes: "bytes",
    np.ndarray: "array",
    type(None): "none",
    None: "none",
}


def validate(cls: Any) -> Schema:
    """Check that *cls* is a data-only frozen dataclass and return its schema.

    Raises:
        SchemaError: *cls* breaks one of the rules this module lists, naming it.
    """
    cached = _CACHE.get(cls) if isinstance(cls, type) else None
    if cached is not None:
        return cached
    return _validate(cls, ())


def _validate(cls: Any, stack: tuple[type, ...]) -> Schema:
    cached = _CACHE.get(cls) if isinstance(cls, type) else None
    if cached is not None:
        return cached
    if not isinstance(cls, type):
        raise SchemaError(f"{cls!r} is not a class: declare a frozen dataclass")
    name = cls.__qualname__
    if cls in stack:
        raise SchemaError(f"{name} contains itself; a declaration cannot be recursive")
    if not dataclasses.is_dataclass(cls):
        raise SchemaError(f"{name} is not a dataclass: declare it @dataclass(frozen=True)")
    params = cls.__dataclass_params__  # type: ignore[attr-defined]
    if not params.frozen:
        raise SchemaError(f"{name} is not frozen: declare it @dataclass(frozen=True)")
    for base in cls.__mro__[:-1]:
        if not dataclasses.is_dataclass(base):
            raise SchemaError(f"{name} inherits from {base.__qualname__}, which is not a dataclass")
        own = vars(base)
        if "__post_init__" in own:
            raise SchemaError(
                f"{base.__qualname__} defines __post_init__; a declaration runs no code"
            )
        if "__new__" in own:
            raise SchemaError(f"{base.__qualname__} defines __new__; a declaration runs no code")
        init = own.get("__init__")
        if (
            init is not None
            and getattr(getattr(init, "__code__", None), "co_filename", "") != "<string>"
        ):
            raise SchemaError(f"{base.__qualname__} defines __init__; a declaration runs no code")
    try:
        hints = typing.get_type_hints(cls)
    except Exception as exc:
        raise SchemaError(f"the annotations of {name} do not resolve: {exc}") from None
    for field_name, hint in hints.items():
        if isinstance(hint, dataclasses.InitVar) or hint is dataclasses.InitVar:
            raise SchemaError(f"{name}.{field_name} is an InitVar; a declaration runs no code")
    fields = []
    for f in dataclasses.fields(cls):
        if not f.init:
            raise SchemaError(f"{name}.{f.name} is not in __init__; every field travels")
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            raise SchemaError(f"{name}.{f.name} has a default; every field is required")
        hint = hints.get(f.name)
        try:
            kind = _resolve(hint, (*stack, cls))
        except SchemaError as exc:
            raise SchemaError(f"{name}.{f.name}: {exc}") from None
        fields.append(_Field(f.name, kind))
    schema = Schema(cls=cls, name=f"{cls.__module__}.{name}", fields=tuple(fields))
    _CACHE[cls] = schema
    return schema


def _resolve(hint: Any, stack: tuple[type, ...]) -> _Type:
    if hint in _SCALARS:
        return _Type(_SCALARS[hint])
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        if len(rest) != 1 or len(args) != 2:
            raise SchemaError(f"{hint} is a union; only `T | None` is allowed")
        return _Type("optional", (_resolve(rest[0], stack),))
    if origin is list and len(args) == 1:
        return _Type("list", (_resolve(args[0], stack),))
    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return _Type("vartuple", (_resolve(args[0], stack),))
        return _Type("tuple", tuple(_resolve(a, stack) for a in args))
    if origin is dict and len(args) == 2:
        if args[0] not in (str, int):
            raise SchemaError(f"{hint} has {args[0]} keys; only str or int keys travel")
        return _Type("dict", (args[0].__name__, _resolve(args[1], stack)))
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return _Type("record", (_validate(hint, stack),))
    raise SchemaError(
        f"{hint!r} is not a type that travels: use bool, int, float, str, bytes, numpy.ndarray, "
        "None, a frozen dataclass, list[T], tuple[...], dict[str, T], or T | None"
    )


def describe(cls: Any) -> str:
    """Return a stable, readable description of *cls*'s schema, nested declarations included."""
    seen: dict[str, str] = {}

    def walk(schema: Schema) -> None:
        if schema.name in seen:
            return
        seen[schema.name] = schema.text()
        for f in schema.fields:
            for sub in _records(f.kind):
                walk(sub)

    walk(validate(cls))
    return "; ".join(seen[k] for k in seen)


def holds_buffers(cls: Any) -> bool:
    """Return whether *cls*'s schema holds an array or bytes anywhere, nested types included.

    Raises:
        SchemaError: *cls* is not a valid declaration.
    """

    def walk(kind: _Type) -> bool:
        if kind.tag in ("array", "bytes"):
            return True
        if kind.tag == "record":
            return any(walk(f.kind) for f in kind.args[0].fields)
        return any(walk(a) for a in kind.args if isinstance(a, _Type))

    return any(walk(f.kind) for f in validate(cls).fields)


def _records(kind: _Type) -> list[Schema]:
    if kind.tag == "record":
        return [kind.args[0]]
    out: list[Schema] = []
    for arg in kind.args:
        if isinstance(arg, _Type):
            out.extend(_records(arg))
    return out


def fingerprint(*, request_type: Any, result_type: Any, signals: tuple[Any, ...]) -> str:
    """Return the fingerprint of a worker contract: its request, result, and signal types.

    ``None`` for the request or result type stands for plain values (no
    declared type). Both halves compute it from the types they declare; the
    worker refuses a session whose fingerprint differs from its own.
    """
    return hashlib.sha256(contract_text(request_type, result_type, signals).encode()).hexdigest()[
        :16
    ]


def contract_text(request_type: Any, result_type: Any, signals: tuple[Any, ...]) -> str:
    """The text :func:`fingerprint` hashes: each declared type described, or ``plain``."""

    def one(cls: Any) -> str:
        return "plain" if cls is None else describe(cls)

    return " || ".join(
        [f"request: {one(request_type)}", f"result: {one(result_type)}"]
        + [f"signal {i}: {describe(s)}" for i, s in enumerate(signals)]
    )


# -- dump -----------------------------------------------------------------------------------


def dump(value: Any, cls: Any) -> dict[str, Any]:
    """Return *value*'s fields as plain values, checking each against *cls*'s declaration.

    Raises:
        SchemaError: *value* is not exactly a *cls*, or a field does not match.
    """
    schema = validate(cls)
    if type(value) is not schema.cls:
        raise SchemaError(f"expected a {schema.name}, got a {type(value).__qualname__}")
    return _dump_record(value, schema, schema.name)


def _dump_record(value: Any, schema: Schema, where: str) -> dict[str, Any]:
    return {
        f.name: _dump(getattr(value, f.name), f.kind, f"{where}.{f.name}") for f in schema.fields
    }


def _dump(value: Any, kind: _Type, where: str) -> Any:
    tag = kind.tag
    if isinstance(value, np.generic) and tag in ("bool", "int", "float", "optional"):
        value = value.item()
    if tag == "optional":
        return None if value is None else _dump(value, kind.args[0], where)
    if tag == "none":
        if value is not None:
            raise _mismatch(where, "None", value)
        return None
    if tag == "bool":
        if type(value) is not bool:
            raise _mismatch(where, "bool", value)
        return value
    if tag == "int":
        if type(value) is not int:
            raise _mismatch(where, "int", value)
        return value
    if tag == "float":
        if type(value) is float:
            return value
        if type(value) is int:
            return float(value)
        raise _mismatch(where, "float", value)
    if tag == "str":
        if type(value) is not str:
            raise _mismatch(where, "str", value)
        return value
    if tag == "bytes":
        if type(value) is not bytes:
            raise _mismatch(where, "bytes", value)
        return value
    if tag == "array":
        if not isinstance(value, np.ndarray):
            raise _mismatch(where, "numpy.ndarray", value)
        return value
    if tag == "record":
        schema = kind.args[0]
        if type(value) is not schema.cls:
            raise _mismatch(where, schema.name, value)
        return _dump_record(value, schema, where)
    if tag == "list":
        if type(value) is not list:
            raise _mismatch(where, "list", value)
        return [_dump(v, kind.args[0], f"{where}[{i}]") for i, v in enumerate(value)]
    if tag == "vartuple":
        if type(value) is not tuple:
            raise _mismatch(where, "tuple", value)
        return tuple(_dump(v, kind.args[0], f"{where}[{i}]") for i, v in enumerate(value))
    if tag == "tuple":
        if type(value) is not tuple or len(value) != len(kind.args):
            raise _mismatch(where, f"a tuple of {len(kind.args)}", value)
        return tuple(
            _dump(v, k, f"{where}[{i}]")
            for i, (v, k) in enumerate(zip(value, kind.args, strict=True))
        )
    if tag == "dict":
        key_type = str if kind.args[0] == "str" else int
        if type(value) is not dict:
            raise _mismatch(where, "dict", value)
        out = {}
        for k, v in value.items():
            if type(k) is not key_type:
                raise _mismatch(f"{where} key", kind.args[0], k)
            out[k] = _dump(v, kind.args[1], f"{where}[{k!r}]")
        return out
    raise SchemaError(f"{where}: unknown kind {tag}")  # pragma: no cover


def _mismatch(where: str, wanted: str, value: Any) -> SchemaError:
    return SchemaError(f"{where} must be {wanted}, got {type(value).__qualname__}")


# -- load -----------------------------------------------------------------------------------


def load(data: Any, cls: Any) -> Any:
    """Build a *cls* from plain *data*, checking every field against the declaration.

    The instance is built without calling the class's code: its fields are set
    directly, so a declaration cannot run anything as it is decoded. A
    ``tuple`` field takes a list too, as JSON writes tuples as lists.

    Raises:
        SchemaError: *data* is not a mapping of exactly *cls*'s fields, or a
            field's value does not match its declaration.
    """
    schema = validate(cls)
    return _load_record(data, schema, schema.name)


def _load_record(data: Any, schema: Schema, where: str) -> Any:
    if type(data) is not dict:
        raise _mismatch(where, f"the fields of {schema.name}", data)
    names = [f.name for f in schema.fields]
    missing = [n for n in names if n not in data]
    extra = [k for k in data if k not in names]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if extra:
            parts.append(f"unexpected {', '.join(map(str, extra))}")
        raise SchemaError(f"{where}: {'; '.join(parts)}")
    instance = object.__new__(schema.cls)
    for f in schema.fields:
        object.__setattr__(instance, f.name, _load(data[f.name], f.kind, f"{where}.{f.name}"))
    return instance


def _load(value: Any, kind: _Type, where: str) -> Any:
    tag = kind.tag
    if tag == "optional":
        return None if value is None else _load(value, kind.args[0], where)
    if tag in ("none", "bool", "int", "float", "str", "bytes"):
        wanted = {
            "none": type(None),
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "bytes": bytes,
        }[tag]
        if type(value) is not wanted:
            raise _mismatch(where, tag, value)
        return value
    if tag == "array":
        if not isinstance(value, np.ndarray):
            raise _mismatch(where, "numpy.ndarray", value)
        return value
    if tag == "record":
        return _load_record(value, kind.args[0], where)
    if tag == "list":
        if type(value) is not list:
            raise _mismatch(where, "list", value)
        return [_load(v, kind.args[0], f"{where}[{i}]") for i, v in enumerate(value)]
    if tag == "vartuple":
        if type(value) not in (tuple, list):
            raise _mismatch(where, "tuple", value)
        return tuple(_load(v, kind.args[0], f"{where}[{i}]") for i, v in enumerate(value))
    if tag == "tuple":
        if type(value) not in (tuple, list) or len(value) != len(kind.args):
            raise _mismatch(where, f"a tuple of {len(kind.args)}", value)
        return tuple(
            _load(v, k, f"{where}[{i}]")
            for i, (v, k) in enumerate(zip(value, kind.args, strict=True))
        )
    if tag == "dict":
        key_type = str if kind.args[0] == "str" else int
        if type(value) is not dict:
            raise _mismatch(where, "dict", value)
        out = {}
        for k, v in value.items():
            if type(k) is not key_type:
                raise _mismatch(f"{where} key", kind.args[0], k)
            out[k] = _load(v, kind.args[1], f"{where}[{k!r}]")
        return out
    raise SchemaError(f"{where}: unknown kind {tag}")  # pragma: no cover
