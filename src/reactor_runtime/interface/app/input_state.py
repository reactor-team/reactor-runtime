"""Typed, client-mutable session state — :class:`InputState`.

An application declares the parameters a client may change mid-session as
fields on an :class:`InputState` subclass and names it as ``state: MyState``.
A fresh instance is built when a session starts and read as
``self.state.<field>`` for as long as the session lives.

Field visibility is by name:

- A **public** field (no leading underscore) becomes a ``set_<field>`` command
  the client can send. :class:`reactor_runtime.ReactorApp` generates the
  handler from the field's type and :func:`InputField` constraints, so the
  command is validated and documented exactly like a hand-written ``@event``.
- A **private** field (leading underscore) is a session-local cache the client
  never sees — derived values a custom ``@event`` handler maintains.
- A field typed :class:`UploadedFile` is a public upload slot: its ``set_``
  command carries an upload reference the runtime resolves to bytes first.
"""

from __future__ import annotations

from types import UnionType
from typing import Any, ClassVar, Union, dataclass_transform, get_args, get_origin, get_type_hints

from reactor_runtime.core.fields import (
    NO_DEFAULT,
    FieldInfo,
    InputField,
    apply_dataclass,
    inherited_record,
    own_annotations,
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
    """Base for an application's typed, session-scoped state.

    Subclass with annotated fields. Use :func:`InputField` as a field default to
    attach validation constraints. Public fields are surfaced to the client as
    ``set_<field>`` commands; underscore-prefixed fields stay private. Declaring
    the subclass partitions its fields and turns it into a dataclass, so a fresh
    instance constructs from defaults at the start of every session.

    A public field declared without a default is a required field; a client
    must set it before its value is read. A private field always needs a
    default, because nothing but the defaults builds the state at session
    start. Mutable defaults (``list`` / ``dict`` / ``set``) are rejected at
    declaration, since one would be shared across sessions.

    A subclass of a state class inherits every field of its parents. A field it
    declares again replaces the parent's, default and constraints included. The
    fields a subclass adds are keyword-only in its constructor, so a required
    field can follow an inherited one that has a default.
    """

    _public_fields: ClassVar[dict[str, FieldInfo]]
    _private_fields: ClassVar[set[str]]
    _upload_fields: ClassVar[set[str]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        public = inherited_record(cls, "_public_fields")
        private: set[str] = set()
        uploads: set[str] = set()
        for base in reversed(cls.__mro__[1:]):
            private |= base.__dict__.get("_private_fields", set())
            # Upload membership follows the same base that wins the field, so a
            # name one base declares as an upload and a nearer base as something
            # else is not an upload.
            base_uploads = base.__dict__.get("_upload_fields", set())
            for name in base.__dict__.get("_public_fields", {}):
                if name in base_uploads:
                    uploads.add(name)
                else:
                    uploads.discard(name)
        inherits_fields = bool(public or private)

        annotations = own_annotations(cls)
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        # A dataclass requires every field without a default to precede those
        # with one, so the two are gathered separately and re-laid in that order.
        no_default: list[str] = []
        has_default: list[str] = []
        class_vars: list[str] = []

        for name in list(annotations):
            raw = cls.__dict__.get(name, _MISSING)
            annotation = hints.get(name, annotations[name])
            if annotation is ClassVar or get_origin(annotation) is ClassVar:
                # A ClassVar is not a field. Redeclaring an inherited field as
                # one takes it out of the dataclass, so it leaves the record too.
                public.pop(name, None)
                private.discard(name)
                uploads.discard(name)
                class_vars.append(name)
                continue
            is_upload = annotation is UploadedFile or _unwrap_optional(annotation) is UploadedFile
            uploads.discard(name)

            if name.startswith("_"):
                if raw is _MISSING:
                    # A private field is the model's own; no client sets it, and
                    # the state is built with no arguments at session start.
                    raise TypeError(
                        f"{cls.__qualname__}: private field '{name}' needs a default. "
                        "The state is built from defaults when a session starts, and "
                        "no set_ command fills a private field."
                    )
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

        cls.__annotations__ = {
            name: annotations[name] for name in no_default + has_default + class_vars
        }
        cls._public_fields = public
        cls._private_fields = private
        cls._upload_fields = uploads

        apply_dataclass(cls, required=no_default, inherits=inherits_fields)
