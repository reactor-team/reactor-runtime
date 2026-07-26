"""Shared type and protocol vocabulary for the runtime.

The neutral foundation every other component imports — value types, the
transport-boundary protocols, the session vocabulary, the model-boundary
vocabulary, and the service-lifecycle contract. Types only, no behaviour beyond
small pure helpers, so it sits at the root of the dependency graph.
"""

from reactor_runtime.core.fields import FieldInfo, InputField
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    Command,
    CommandField,
    EndReason,
    FileUploaded,
    ReactorEvent,
    SessionEnded,
    SessionStarted,
    TransitionEvent,
    UploadedFile,
)
from reactor_runtime.core.service import RecordingConfig, RuntimeConfig, ServiceComponent
from reactor_runtime.core.session import (
    JOURNAL_EVENTS,
    SessionEvent,
    SessionState,
    Transition,
)
from reactor_runtime.core.transport import Connection, ConnectionSink
from reactor_runtime.core.typespec import TypeSpec
from reactor_runtime.core.values import (
    ConnectionCapabilities,
    ConnId,
    Health,
    HealthStatus,
    InputFrame,
    MediaBundle,
    MediaChunk,
    RuntimeState,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)

__all__ = [
    "JOURNAL_EVENTS",
    "ClientConnected",
    "ClientDisconnected",
    "Command",
    "CommandField",
    "ConnId",
    "Connection",
    "ConnectionCapabilities",
    "ConnectionSink",
    "EndReason",
    "FieldInfo",
    "FileUploaded",
    "Health",
    "HealthStatus",
    "InputField",
    "InputFrame",
    "MediaBundle",
    "MediaChunk",
    "ReactorEvent",
    "RecordingConfig",
    "RuntimeConfig",
    "RuntimeState",
    "ServiceComponent",
    "SessionEnded",
    "SessionEvent",
    "SessionStarted",
    "SessionState",
    "TrackData",
    "TrackDirection",
    "TrackInfo",
    "TrackKind",
    "Transition",
    "TransitionEvent",
    "TypeSpec",
    "UploadedFile",
]
