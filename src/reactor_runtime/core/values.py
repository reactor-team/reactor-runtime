"""Neutral value vocabulary shared across the runtime.

The plain data every component passes around: connection identifiers, inbound
and outbound media, the capabilities a transport advertises, and the health a
component reports. These carry no behaviour beyond small pure helpers and
depend on nothing else in the package, so they sit at the root of the import
graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, NewType

import numpy.typing as npt

ConnId = NewType("ConnId", int)
"""Identifier for one client connection within a session.

Allocated centrally so that multiple ingresses cannot collide. A type distinct
from a bare ``int`` keeps connection ids from being confused with counts,
indices, or other integers at the boundaries.
"""


@dataclass(frozen=True, eq=False)
class InputFrame:
    """A single inbound media frame with its presentation timestamp.

    Carries the decoded payload as a NumPy array plus a transport-agnostic
    ``pts`` so model code can align frames across tracks (or against
    data-channel events) without per-track bookkeeping.

    Equality is identity-based on purpose: the generated array comparison would
    raise on a non-scalar array and the array is unhashable, and two frames
    wrapping byte-identical pixels are not the same frame anyway.

    Attributes:
        data: Decoded payload. Video is ``(H, W, 3)`` ``uint8`` RGB for one
            frame; audio is ``(1, M)`` ``int16`` mono samples.
        pts: Presentation timestamp in seconds, or ``None`` when the transport
            could not provide one.
    """

    data: npt.NDArray[Any]
    pts: float | None = None


class TrackKind(StrEnum):
    """The kind of media a track carries."""

    VIDEO = "video"
    AUDIO = "audio"


class TrackDirection(StrEnum):
    """Direction of flow for a track, from the model's perspective.

    ``IN`` is client to model (e.g. a webcam feed); ``OUT`` is model to client
    (e.g. generated video). Bidirectional flow is expressed as two tracks with
    distinct names.
    """

    IN = "in"
    OUT = "out"


@dataclass(frozen=True)
class TrackInfo:
    """Immutable metadata describing a single media track.

    Attributes:
        name: Track identifier, unique within a session (e.g. ``"main_video"``).
        kind: The kind of media the track carries.
        rate: Native rate in units per second — frames/sec for video,
            samples/sec for audio. ``0`` when unknown or not applicable.
        direction: Flow direction from the model's perspective.
    """

    name: str
    kind: TrackKind
    rate: float = 0.0
    direction: TrackDirection = TrackDirection.OUT


@dataclass(eq=False)
class TrackData:
    """One track's payload inside a :class:`MediaBundle`.

    Attributes:
        info: Metadata for the track.
        data: Payload array. Video is ``(H, W, 3)`` ``uint8`` RGB for one frame
            (or ``(N, H, W, 3)`` for a batch); audio is ``(1, M)`` ``int16``
            mono samples.
    """

    info: TrackInfo
    data: npt.NDArray[Any]


@dataclass
class MediaBundle:
    """Synchronised multi-track media unit emitted by the model.

    Groups one or more named tracks that belong to the same logical time
    interval and flows from the model out to the transport. Keyed by track name
    for O(1) lookup.

    Attributes:
        tracks: Track name to its payload.
    """

    tracks: dict[str, TrackData] = field(default_factory=dict)

    def get_track(self, name: str) -> TrackData | None:
        """Return the payload for *name*, or ``None`` when absent."""
        return self.tracks.get(name)

    def get_tracks(self) -> list[TrackData]:
        """Return every track in the bundle."""
        return list(self.tracks.values())

    def get_tracks_by_kind(self, kind: TrackKind) -> list[TrackData]:
        """Return the tracks matching *kind*."""
        return [track for track in self.tracks.values() if track.info.kind == kind]


@dataclass(frozen=True)
class ConnectionCapabilities:
    """What media a connection's wire can carry.

    Makes heterogeneous sessions sound: a connection with no media is never sent
    the video track, while a full WebRTC client is. A transport advertises what it
    can carry so media is routed only to connections that can receive it, rather
    than relying on a silent no-op. The data channel is universal — every
    connection carries control messages — so only media is optional.

    Attributes:
        carries_video: The wire can deliver outbound video.
        carries_audio: The wire can deliver outbound audio.
    """

    carries_video: bool = False
    carries_audio: bool = False


class HealthStatus(Enum):
    """A component's readiness, used to aggregate process readiness."""

    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    HEALTHY = "healthy"


_SEVERITY: dict[HealthStatus, int] = {
    HealthStatus.UNHEALTHY: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.HEALTHY: 2,
}


@dataclass(frozen=True)
class Health:
    """The health a component reports, aggregated into process readiness.

    Attributes:
        status: The component's readiness.
        detail: Optional human-readable explanation, useful when not healthy.
    """

    status: HealthStatus
    detail: str | None = None

    @classmethod
    def healthy(cls, detail: str | None = None) -> Health:
        """Return a healthy report."""
        return cls(HealthStatus.HEALTHY, detail)

    @classmethod
    def aggregate(cls, parts: Iterable[Health]) -> Health:
        """Combine component reports into one, keeping the worst status.

        An empty input is healthy: a process with nothing to report is ready.
        The details of every non-healthy part are joined so the reason for a
        degraded or unhealthy roll-up is preserved.
        """
        worst = HealthStatus.HEALTHY
        details: list[str] = []
        for part in parts:
            if _SEVERITY[part.status] < _SEVERITY[worst]:
                worst = part.status
            if part.status is not HealthStatus.HEALTHY and part.detail:
                details.append(part.detail)
        return cls(worst, "; ".join(details) or None)
