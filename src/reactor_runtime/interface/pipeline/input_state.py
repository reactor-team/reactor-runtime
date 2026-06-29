"""Typed, client-mutable session state — :class:`InputState`.

A pipeline declares the parameters a client may change mid-session as fields on
an :class:`InputState` subclass. A fresh instance is created for each client
connection and read from inside ``inference()`` as ``self.state.<field>``.

Field visibility is by name:

- A **public** field (no leading underscore) becomes a ``set_<field>`` command
  the client can send. :class:`reactor_runtime.ReactorPipeline` generates the
  handler from the field's type and :func:`InputField` constraints, so the
  command is validated and documented exactly like a hand-written ``@event``.
- A **private** field (leading underscore) is a session-local cache the client
  never sees — derived values a custom ``@event`` handler maintains.
- A field typed :class:`UploadedFile` is a public upload slot: its ``set_``
  command carries an upload reference the runtime resolves to bytes first.
"""

from __future__ import annotations

import dataclasses
from types import UnionType
from typing import Any, ClassVar, Union, dataclass_transform, get_args, get_origin, get_type_hints

from reactor_runtime.core.fields import (
    NO_DEFAULT,
    FieldInfo,
    InputField,
    raise_if_default_invalid,
    raise_if_default_not_static,
)
from reactor_runtime.core.model import UploadedFile

_MISSING = object()


def _unwrap_optional(annotation: Any) -> Any:
    """Return the inner type of an ``X | None`` annotation, else *annotation*."""
    if get_origin(annotation) in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


@dataclass_transform(field_specifiers=(InputField, FieldInfo))
class InputState:
    """Base for a pipeline's typed, per-connection session state.

    Subclass with annotated fields. Use :func:`InputField` as a field default to
    attach validation constraints. Public fields are surfaced to the client as
    ``set_<field>`` commands; underscore-prefixed fields stay private. Declaring
    the subclass partitions its fields and turns it into a dataclass, so a fresh
    instance constructs from defaults at the start of every connection.

    A field declared without a default is a required field; a client must set it
    before its value is read. Mutable defaults (``list`` / ``dict`` / ``set``)
    are rejected at declaration, since one would be shared across sessions.
    """

    _public_fields: ClassVar[dict[str, FieldInfo]]
    _private_fields: ClassVar[set[str]]
    _upload_fields: ClassVar[set[str]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        public: dict[str, FieldInfo] = {}
        private: set[str] = set()
        uploads: set[str] = set()

        annotations = getattr(cls, "__annotations__", {})
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        # A dataclass requires every field without a default to precede those
        # with one, so the two are gathered separately and re-laid in that order.
        no_default: list[str] = []
        has_default: list[str] = []

        for name in list(annotations):
            raw = cls.__dict__.get(name, _MISSING)
            annotation = hints.get(name, annotations[name])
            is_upload = annotation is UploadedFile or _unwrap_optional(annotation) is UploadedFile

            if name.startswith("_"):
                private.add(name)
                has_default.append(name)
            elif is_upload:
                # An upload slot starts empty; the client fills it with a
                # set_<field> that carries an upload reference.
                uploads.add(name)
                public[name] = raw if isinstance(raw, FieldInfo) else FieldInfo(default=None)
                setattr(cls, name, None)
                has_default.append(name)
            elif isinstance(raw, FieldInfo):
                public[name] = raw
                if raw.default is NO_DEFAULT:
                    if hasattr(cls, name):
                        delattr(cls, name)
                    no_default.append(name)
                else:
                    raise_if_default_invalid(cls.__qualname__, name, raw.default, raw)
                    setattr(cls, name, raw.default)
                    has_default.append(name)
            elif raw is not _MISSING:
                raise_if_default_not_static(cls.__qualname__, name, raw)
                public[name] = FieldInfo(default=raw)
                has_default.append(name)
            else:
                public[name] = FieldInfo()
                no_default.append(name)

        cls.__annotations__ = {name: annotations[name] for name in no_default + has_default}
        cls._public_fields = public
        cls._private_fields = private
        cls._upload_fields = uploads

        if not dataclasses.is_dataclass(cls):
            dataclasses.dataclass(cls)
