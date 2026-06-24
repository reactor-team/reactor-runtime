"""Shared type and protocol vocabulary for the runtime.

The neutral foundation every other component imports — value types, the
transport-boundary protocols, the session vocabulary, the model-boundary
vocabulary, and the service-lifecycle contract. Types only, no behaviour beyond
small pure helpers, so it sits at the root of the dependency graph.
"""

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
    "ConnId",
    "Connection",
    "ConnectionCapabilities",
    "ConnectionSink",
    "Health",
    "HealthStatus",
    "InputFrame",
    "MediaBundle",
    "SessionEvent",
    "SessionState",
    "TrackData",
    "TrackDirection",
    "TrackInfo",
    "TrackKind",
    "Transition",
]
