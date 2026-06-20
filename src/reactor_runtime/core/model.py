"""Model-boundary vocabulary.

The types the model bridge keys off, split by authority. ``Command`` is the open
set a client authors and the contract validates before it reaches a handler. The
``ReactorEvent`` set is the closed, reactor-authoritative facts the runtime hands
the model directly — never from the wire, never validated. The ``RunnerEvent``
union is what the runtime journals outward for an external consumer to mirror.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, dataclass_transform, get_origin, get_type_hints

from reactor_runtime.core.fields import (
    NO_DEFAULT,
    FieldInfo,
    InputField,
    raise_if_default_invalid,
    raise_if_default_not_static,
)
from reactor_runtime.core.naming import pascal_to_snake
from reactor_runtime.core.session import Transition
from reactor_runtime.core.values import ConnId

if TYPE_CHECKING:
    from reactor_runtime.core.typespec import TypeSpec


@dataclass(frozen=True)
class CommandField:
    """The resolved type and constraints of one command field.

    Built once when the command class is created and cached on it, so the
    contract validates a payload and renders the schema from the same resolved
    form rather than re-walking annotations.

    Attributes:
        spec: The field's resolved type, driving both validation and schema.
        info: The field's default and constraint metadata.
    """

    spec: TypeSpec
    info: FieldInfo


@dataclass_transform(field_specifiers=(InputField, FieldInfo))
class Command:
    """Marker base for a user-authored command — the open set.

    A model declares its commands by subclassing with typed fields; the contract
    validates raw client arguments into one of these before dispatch. Untrusted
    by definition: a command originates on the wire, so it is proven valid before
    any handler sees it.

    Subclassing is the definition-time gate. Creating a subclass resolves every
    annotated field into a :class:`CommandField` (rejecting unsupported types and
    invalid defaults at import), turns the class into a dataclass, and caches the
    resolved fields on the class. There is no global registry — each command
    carries its own contract.
    """

    name: ClassVar[str]
    __command_fields__: ClassVar[dict[str, CommandField]]
    __upload_fields__: ClassVar[frozenset[str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _build_command(cls)
        if not dataclasses.is_dataclass(cls):
            dataclasses.dataclass(cls)
        if "name" not in cls.__dict__:
            cls.name = pascal_to_snake(cls.__name__)


def _is_classvar(annotation: Any) -> bool:
    """Return whether *annotation* is ``ClassVar`` or ``ClassVar[...]``."""
    return annotation is ClassVar or get_origin(annotation) is ClassVar


def _build_command(cls: type[Command]) -> None:
    """Resolve a command class's fields and cache them on it.

    Runs once per subclass, before ``@dataclass`` is applied. Unwraps
    ``InputField`` defaults so the dataclass sees plain values, resolves each
    field's :class:`TypeSpec` (the support check), records which fields are
    uploads, and reorders so fields without a default come first — the order a
    dataclass requires.

    Args:
        cls: The freshly created command subclass.

    Raises:
        TypeError: If a field uses an unsupported type, a raw ``dataclasses.field``,
            or a default that is mutable or out of its own constraints.
    """
    # Imported here, not at module level, so this vocabulary stays the leaf the
    # type system reads back from for upload detection — the cycle is broken by
    # deferring the import until a command is actually declared.
    from reactor_runtime.core.typespec import OptionalSpec, TypeSpec, UploadSpec

    annotations: dict[str, Any] = getattr(cls, "__annotations__", {})
    if not annotations:
        cls.__command_fields__ = {}
        cls.__upload_fields__ = frozenset()
        return

    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    owner = cls.__qualname__
    fields: dict[str, CommandField] = {}
    uploads: set[str] = set()
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
                "a command — use `InputField(default=..., ...)`."
            )

        spec = TypeSpec.of(annotation)

        if isinstance(raw, FieldInfo):
            info = raw
            raise_if_default_invalid(owner, name, info.default, info)
            if info.default is NO_DEFAULT:
                if hasattr(cls, name):
                    delattr(cls, name)
                no_default.append(name)
            else:
                setattr(cls, name, info.default)
                has_default.append(name)
        elif raw is NO_DEFAULT:
            info = FieldInfo()
            no_default.append(name)
        else:
            raise_if_default_not_static(owner, name, raw)
            info = FieldInfo(default=raw)
            has_default.append(name)

        if info.default is not NO_DEFAULT and info.default is not None:
            default_reason = spec.check(info.default)
            if default_reason is not None:
                raise TypeError(
                    f"{owner}: default for '{name}' does not match its type ({default_reason})."
                )

        fields[name] = CommandField(spec=spec, info=info)
        resolved = spec.inner if isinstance(spec, OptionalSpec) else spec
        if isinstance(resolved, UploadSpec):
            uploads.add(name)

    if no_default and has_default:
        cls.__annotations__ = {key: annotations[key] for key in no_default + has_default}

    cls.__command_fields__ = fields
    cls.__upload_fields__ = frozenset(uploads)


class EndReason(StrEnum):
    """Why a session ended."""

    STOPPED = "stopped"
    TIMED_OUT = "timed_out"
    EVICTED = "evicted"
    MODERATED = "moderated"
    ERROR = "error"


@dataclass(frozen=True)
class UploadedFile:
    """A file the runtime has fetched and vouched for, ready for the model.

    Attributes:
        upload_id: Identifier the client used to reference the file.
        name: Original file name.
        mime_type: Declared content type.
        data: The fetched bytes.
    """

    upload_id: str
    name: str
    mime_type: str
    data: bytes


@dataclass(frozen=True)
class ReactorEvent:
    """Base for the closed set of reactor-authoritative facts.

    Authored by the runtime, never by a client and never carried on the wire, so
    the model trusts these without validation.
    """


@dataclass(frozen=True)
class SessionStarted(ReactorEvent):
    """The session has begun.

    Attributes:
        session_id: Identifier for the session that started.
    """

    session_id: str


@dataclass(frozen=True)
class SessionEnded(ReactorEvent):
    """The session has ended.

    Attributes:
        session_id: Identifier for the session that ended.
        reason: Why the session ended.
    """

    session_id: str
    reason: EndReason


@dataclass(frozen=True)
class ClientConnected(ReactorEvent):
    """A client connection opened.

    Attributes:
        conn_id: The connection that opened.
        total: Live connection count after the open.
    """

    conn_id: ConnId
    total: int


@dataclass(frozen=True)
class ClientDisconnected(ReactorEvent):
    """A client connection closed.

    Attributes:
        conn_id: The connection that closed.
        total: Live connection count after the close.
    """

    conn_id: ConnId
    total: int


@dataclass(frozen=True)
class FileUploaded(ReactorEvent):
    """A client-uploaded file is available to the model.

    Attributes:
        file: The fetched, vouched-for file.
        conn_id: The connection that uploaded it.
    """

    file: UploadedFile
    conn_id: ConnId


@dataclass(frozen=True)
class TransitionEvent:
    """A session-state move, journalled for an external consumer.

    Attributes:
        transition: The move that was applied.
    """

    transition: Transition


@dataclass(frozen=True)
class ConnectionEvent:
    """A connection opened or closed.

    Attributes:
        conn_id: The connection.
        opened: ``True`` on open, ``False`` on close.
    """

    conn_id: ConnId
    opened: bool


@dataclass(frozen=True)
class ConnectionAnswered:
    """A transport produced the negotiation answer for one connection.

    Journalled so an external consumer can hand the answer back to the client
    that offered. The payload is transport-agnostic and opaque to the runtime:
    for WebRTC it is the SDP answer as a ``{"type", "sdp"}`` mapping. Precedes
    the :class:`ConnectionEvent` open, which fires only once the wire connects.

    Attributes:
        conn_id: The connection the answer belongs to.
        answer: The transport's answer as key-value pairs, relayed verbatim.
    """

    conn_id: ConnId
    answer: Mapping[str, str]


@dataclass(frozen=True)
class InboundCommandEvent:
    """A validated inbound command, journalled for moderation or audit.

    Attributes:
        name: The command name.
        args: The validated argument mapping.
        conn_id: The connection that sent it, when known.
    """

    name: str
    args: Mapping[str, Any]
    conn_id: ConnId | None = None


@dataclass(frozen=True)
class ClipReadyEvent:
    """A recorded clip is on disk and ready to fetch.

    Attributes:
        clip_id: Identifier for the clip.
    """

    clip_id: str


@dataclass(frozen=True)
class SessionMetricEvent:
    """A named session counter sample.

    Attributes:
        name: The metric name.
        value: The sampled value.
    """

    name: str
    value: float


@dataclass(frozen=True)
class ErrorEvent:
    """A notable error worth surfacing on the egress journal.

    Attributes:
        message: Human-readable description.
    """

    message: str


RunnerEvent = (
    TransitionEvent
    | ConnectionEvent
    | ConnectionAnswered
    | InboundCommandEvent
    | ClipReadyEvent
    | SessionMetricEvent
    | ErrorEvent
)
"""The egress union the runtime journals out for an external consumer to mirror.

The runtime records one of these facts and surfaces it for a consumer to map
onto its own world, rather than composing a platform object in directly.
"""
