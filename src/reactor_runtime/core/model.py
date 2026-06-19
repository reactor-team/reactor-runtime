"""Model-boundary vocabulary.

The types the model bridge keys off, split by authority. ``Command`` is the open
set a client authors and the contract validates before it reaches a handler. The
``ReactorEvent`` set is the closed, reactor-authoritative facts the runtime hands
the model directly — never from the wire, never validated. The ``RunnerEvent``
union is what the runtime journals outward for an external consumer to mirror.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from reactor_runtime.core.session import Transition
from reactor_runtime.core.values import ConnId


class Command:
    """Marker base for a user-authored command — the open set.

    A model declares its commands by subclassing with typed fields; the contract
    validates raw client arguments into one of these before dispatch. Untrusted
    by definition: a command originates on the wire, so it is proven valid before
    any handler sees it.
    """


class EndReason(StrEnum):
    """Why a session ended."""

    STOPPED = "stopped"
    TIMED_OUT = "timed_out"
    EVICTED = "evicted"
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
    | InboundCommandEvent
    | ClipReadyEvent
    | SessionMetricEvent
    | ErrorEvent
)
"""The egress union the runtime journals out for an external consumer to mirror.

The runtime records one of these facts and surfaces it for a consumer to map
onto its own world, rather than composing a platform object in directly.
"""
