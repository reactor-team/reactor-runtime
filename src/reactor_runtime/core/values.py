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

import numpy as np
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

    @property
    def frame_count(self) -> int:
        """How many frames the bundle carries.

        A batched video track is ``(N, H, W, 3)`` and carries ``N`` frames;
        anything else — an unbatched ``(H, W, 3)`` video track, or an audio-only
        bundle — is one frame. When several video tracks are batched they must
        agree on ``N``.

        Raises:
            ValueError: If two batched video tracks disagree on their frame count.
        """
        counts = {
            track.data.shape[0]
            for track in self.get_tracks_by_kind(TrackKind.VIDEO)
            if track.data.ndim == 4
        }
        if not counts:
            return 1
        if len(counts) > 1:
            raise ValueError(f"batched video tracks disagree on frame count: {sorted(counts)}")
        return counts.pop()


def split_batch(bundle: MediaBundle) -> list[MediaBundle]:
    """Split a multi-frame bundle into one bundle per frame.

    A batched video track is ``(N, H, W, 3)``; an unbatched one is ``(H, W, 3)``
    and is repeated into every frame. Audio is divided proportionally across the
    frames. All batched video tracks must agree on ``N``.

    Args:
        bundle: The bundle to split.

    Returns:
        One single-frame bundle per batched frame, or ``[bundle]`` unchanged when
        there is nothing to split.

    Raises:
        ValueError: If two batched video tracks disagree on the batch size.
    """
    video_tracks = bundle.get_tracks_by_kind(TrackKind.VIDEO)
    if not video_tracks:
        return [bundle]

    batched = [(track, track.data.shape[0]) for track in video_tracks if track.data.ndim == 4]
    if not batched:
        return [bundle]

    n_frames = batched[0][1]
    if n_frames == 1:
        # Squeeze the batch dimension into a fresh bundle rather than editing the
        # caller's: the multi-frame path below also leaves the input untouched,
        # and a producer must be able to read back what it submitted.
        squeezed = dict(bundle.tracks)
        for track, _ in batched:
            squeezed[track.info.name] = TrackData(info=track.info, data=track.data[0])
        return [MediaBundle(tracks=squeezed)]

    for track, size in batched:
        if size != n_frames:
            raise ValueError(
                f"Video track '{track.info.name}' has batch size {size}, "
                f"expected {n_frames} (from '{batched[0][0].info.name}')"
            )

    video_splits: dict[str, list[Any]] = {}
    for track in video_tracks:
        if track.data.ndim == 4:
            video_splits[track.info.name] = list(track.data)
        else:
            video_splits[track.info.name] = [track.data] * n_frames

    audio_splits: dict[str, list[Any]] = {}
    for track in bundle.get_tracks_by_kind(TrackKind.AUDIO):
        audio = track.data
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        audio_splits[track.info.name] = np.array_split(audio, n_frames, axis=1)

    info_by_name = {track.info.name: track.info for track in bundle.get_tracks()}
    result: list[MediaBundle] = []
    for index in range(n_frames):
        tracks: dict[str, TrackData] = {}
        for name, frames in video_splits.items():
            tracks[name] = TrackData(info=info_by_name[name], data=frames[index])
        for name, chunks in audio_splits.items():
            tracks[name] = TrackData(info=info_by_name[name], data=chunks[index])
        result.append(MediaBundle(tracks=tracks))
    return result


@dataclass(frozen=True)
class MediaChunk:
    """A batch of finished media plus the rate it should play out at.

    What the model hands downstream on each emission. The model's only media
    concern is producing this; pacing it to a steady wire cadence, timestamping
    it for a recording, and filling gaps when the model pauses are all consumer
    concerns, decided against :attr:`fps`.

    Attributes:
        bundle: The produced media, one payload per track. A video track may be
            batched (``(N, H, W, 3)``) to carry several frames in one chunk.
        fps: The nominal rate, in frames per second, at which the chunk's frames
            should play out — the model's measured throughput when it emitted
            with a compute time, else its declared rate. Always positive.
        n_frames: How many frames the chunk carries (the batch size, or ``1``).
    """

    bundle: MediaBundle
    fps: float
    n_frames: int = 1

    def frames(self) -> list[MediaBundle]:
        """Split the chunk into one single-frame bundle per carried frame."""
        return split_batch(self.bundle)


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
