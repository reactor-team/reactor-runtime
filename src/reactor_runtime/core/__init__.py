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
    StartingInputApplied,
    TransitionEvent,
    UploadedFile,
)
from reactor_runtime.core.service import (
    RecordingConfig,
    RuntimeConfig,
    ServiceComponent,
    StepResultsConfig,
)
from reactor_runtime.core.session import (
    JOURNAL_EVENTS,
    SessionEvent,
    SessionState,
    Transition,
)
from reactor_runtime.core.stats import TransportReading, TransportTrackReading
from reactor_runtime.core.transport import Connection, ConnectionSink, TransportStatsSource
from reactor_runtime.core.typespec import TypeSpec
from reactor_runtime.core.values import (
    ClientConnectionStat,
    ClientStatsBatch,
    ClientTrackDirection,
    ClientTrackStat,
    CommandFailure,
    CompletedStep,
    ConnectionCapabilities,
    ConnId,
    FrameStage,
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
    "ClientConnectionStat",
    "ClientDisconnected",
    "ClientStatsBatch",
    "ClientTrackDirection",
    "ClientTrackStat",
    "Command",
    "CommandFailure",
    "CommandField",
    "CompletedStep",
    "ConnId",
    "Connection",
    "ConnectionCapabilities",
    "ConnectionSink",
    "EndReason",
    "FieldInfo",
    "FileUploaded",
    "FrameStage",
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
    "StartingInputApplied",
    "StepResultsConfig",
    "TrackData",
    "TrackDirection",
    "TrackInfo",
    "TrackKind",
    "Transition",
    "TransitionEvent",
    "TransportReading",
    "TransportStatsSource",
    "TransportTrackReading",
    "TypeSpec",
    "UploadedFile",
]
