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
    ClipReadyEvent,
    Command,
    CommandField,
    EndReason,
    ErrorEvent,
    FileUploaded,
    InboundCommandEvent,
    ReactorEvent,
    RunnerEvent,
    SessionEnded,
    SessionMetricEvent,
    SessionStarted,
    TransitionEvent,
    UploadedFile,
)
from reactor_runtime.core.service import RuntimeConfig, ServiceComponent
from reactor_runtime.core.session import SessionEvent, SessionState, Transition
from reactor_runtime.core.transport import Connection, ConnectionSink
from reactor_runtime.core.typespec import TypeSpec
from reactor_runtime.core.values import (
    ConnectionCapabilities,
    ConnId,
    Health,
    HealthStatus,
    InputFrame,
    MediaBundle,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)

__all__ = [
    "ClientConnected",
    "ClientDisconnected",
    "ClipReadyEvent",
    "Command",
    "CommandField",
    "ConnId",
    "Connection",
    "ConnectionCapabilities",
    "ConnectionSink",
    "EndReason",
    "ErrorEvent",
    "FieldInfo",
    "FileUploaded",
    "Health",
    "HealthStatus",
    "InboundCommandEvent",
    "InputField",
    "InputFrame",
    "MediaBundle",
    "ReactorEvent",
    "RunnerEvent",
    "RuntimeConfig",
    "ServiceComponent",
    "SessionEnded",
    "SessionEvent",
    "SessionMetricEvent",
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
