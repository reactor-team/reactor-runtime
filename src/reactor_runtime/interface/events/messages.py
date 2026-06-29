"""Typed outbound messages — :class:`ModelMessage`.

A model talks back to clients with typed messages. Subclass
:class:`ModelMessage` with dataclass fields, then send an instance; it travels as
``{"type": "<snake_case_name>", "data": {...}}``. Defining the subclass resolves
each field's type (so the contract can render it into the schema) and snapshots
the field descriptions and the class docstring that document the message to
clients.

Outbound messages are never validated against client input, so unlike a command
the resolved field types drive only schema rendering, not request-time checks.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, dataclass_transform, get_origin, get_type_hints

from reactor_runtime.core.fields import NO_DEFAULT, raise_if_default_not_static
from reactor_runtime.core.naming import pascal_to_snake
from reactor_runtime.core.typespec import TypeSpec

MESSAGE_REGISTRY: dict[str, type[ModelMessage]] = {}
"""Every :class:`ModelMessage` subclass that declared fields, by wire name.

Auto-populated when a field-bearing subclass is created. A model talks back with
any registered message — a command's reply or a message it only ever broadcasts —
so the rendered schema reads this registry rather than inferring messages from
command return types alone. Field-less abstract bases are left out.
"""


@dataclass(frozen=True)
class MessageFieldInfo:
    """A single message field's default and description.

    Build instances with :func:`MessageField` rather than constructing directly.

    Attributes:
        default: The field's default value, or :data:`NO_DEFAULT` when required.
        description: Human-readable description, surfaced in the rendered schema.
    """

    default: Any = NO_DEFAULT
    description: str | None = None


def MessageField(  # noqa: N802 — a capitalised factory reads as a type in field declarations
    default: Any = NO_DEFAULT,
    *,
    default_factory: Any = None,
    description: str | None = None,
) -> Any:
    """Declare a default value and description for a message field.

    Use as the default for a :class:`ModelMessage` field. The return type is
    ``Any`` so the call can stand in as the default of a field of any type.

    Args:
        default: Default value for the field.
        default_factory: Unsupported; passing one raises ``TypeError``. Defaults
            must be statically representable — use a literal ``default=...``.
        description: Human-readable description, surfaced in the rendered schema.

    Returns:
        A :class:`MessageFieldInfo` carrying the supplied default and description.

    Raises:
        TypeError: If ``default_factory`` is supplied.
    """
    if default_factory is not None:
        raise TypeError(
            "MessageField(default_factory=...) is not supported. Defaults must be "
            "statically representable — use a literal `default=...`."
        )
    return MessageFieldInfo(default=default, description=description)


@dataclass(frozen=True)
class MessageFieldSpec:
    """The resolved type and metadata of one message field.

    Attributes:
        spec: The field's resolved type, driving schema rendering.
        description: Human-readable description, or ``None``.
        default: The field's default value, or :data:`NO_DEFAULT` when required.
    """

    spec: TypeSpec
    description: str | None
    default: Any


@dataclass_transform(field_specifiers=(MessageField, MessageFieldInfo))
class ModelMessage:
    """Base class for typed outbound messages from model to client.

    Subclass with dataclass fields, then send an instance::

        class CurrentMode(ModelMessage):
            mode: str

        await self.send(CurrentMode(mode="turbo"))

    The client receives ``{"type": "current_mode", "data": {"mode": "turbo"}}``.
    Use :func:`MessageField` to attach a description to a field.
    """

    name: ClassVar[str]
    user_doc: ClassVar[str | None]
    __message_fields__: ClassVar[dict[str, MessageFieldSpec]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _build_message(cls)
        if not dataclasses.is_dataclass(cls):
            dataclasses.dataclass(cls)
        if "name" not in cls.__dict__:
            cls.name = pascal_to_snake(cls.__name__)
        if cls.__message_fields__:
            MESSAGE_REGISTRY[cls.name] = cls

    def to_wire_format(self) -> dict[str, Any]:
        """Serialise into the ``{"type": <name>, "data": {...}}`` wire envelope."""
        data: dict[str, Any] = {}
        if dataclasses.is_dataclass(self):
            for field in dataclasses.fields(self):
                data[field.name] = getattr(self, field.name)
        return {"type": self.name, "data": data}


def _is_classvar(annotation: Any) -> bool:
    """Return whether *annotation* is ``ClassVar`` or ``ClassVar[...]``."""
    return annotation is ClassVar or get_origin(annotation) is ClassVar


def _build_message(cls: type[ModelMessage]) -> None:
    """Resolve a message class's fields and cache them on it.

    Runs once per subclass, before ``@dataclass`` is applied. Snapshots the
    author's docstring (before ``@dataclass`` overwrites a missing one with a
    generated signature), unwraps ``MessageField`` defaults, resolves each
    field's type, and reorders so fields without a default come first.

    Args:
        cls: The freshly created message subclass.

    Raises:
        TypeError: If a field uses an unsupported type, a raw ``dataclasses.field``,
            or a mutable default.
    """
    cls.user_doc = cls.__dict__.get("__doc__")

    annotations: dict[str, Any] = getattr(cls, "__annotations__", {})
    if not annotations:
        cls.__message_fields__ = {}
        return

    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    owner = cls.__qualname__
    fields: dict[str, MessageFieldSpec] = {}
    no_default: list[str] = []
    has_default: list[str] = []

    for name in list(annotations):
        annotation = hints.get(name, annotations[name])
        if _is_classvar(annotation):
            continue

        raw = cls.__dict__.get(name, NO_DEFAULT)
        if isinstance(raw, dataclasses.Field):
            raise TypeError(
                f"{owner}.{name}: a raw `dataclasses.field(...)` is not supported on "
                "a message — use `MessageField(default=..., description=...)`."
            )

        spec = TypeSpec.of(annotation)
        description: str | None = None
        default: Any = NO_DEFAULT

        if isinstance(raw, MessageFieldInfo):
            description = raw.description
            default = raw.default
            if raw.default is NO_DEFAULT:
                if hasattr(cls, name):
                    delattr(cls, name)
                no_default.append(name)
            else:
                raise_if_default_not_static(owner, name, raw.default)
                setattr(cls, name, raw.default)
                has_default.append(name)
        elif raw is NO_DEFAULT:
            no_default.append(name)
        else:
            raise_if_default_not_static(owner, name, raw)
            default = raw
            has_default.append(name)

        if default is not NO_DEFAULT and default is not None:
            default_reason = spec.check(default)
            if default_reason is not None:
                raise TypeError(
                    f"{owner}: default for '{name}' does not match its type ({default_reason})."
                )

        fields[name] = MessageFieldSpec(spec=spec, description=description, default=default)

    if no_default and has_default:
        cls.__annotations__ = {key: annotations[key] for key in no_default + has_default}

    cls.__message_fields__ = fields
