"""Shared type and protocol vocabulary for the runtime.

The neutral foundation every other component imports — value types, the
transport-boundary protocols, the session vocabulary, the model-boundary
vocabulary, and the service-lifecycle contract. Types only, no behaviour beyond
small pure helpers, so it sits at the root of the dependency graph.
"""

from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    ClipReadyEvent,
    Command,
    ConnectionAnswered,
    ConnectionEvent,
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
    "ConnId",
    "Connection",
    "ConnectionAnswered",
    "ConnectionCapabilities",
    "ConnectionEvent",
    "ConnectionSink",
    "EndReason",
    "ErrorEvent",
    "FileUploaded",
    "Health",
    "HealthStatus",
    "InboundCommandEvent",
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
    "UploadedFile",
]
