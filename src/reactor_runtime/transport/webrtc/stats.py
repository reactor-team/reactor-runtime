"""WebRTC peer statistics types.

Shared data types sampled by :class:`~reactor_runtime.transport.webrtc.peer.WebRTCPeer`
and surfaced through :class:`~reactor_runtime.transport.webrtc.connection.WebRTCConnection`.
"""

from __future__ import annotations

from dataclasses import dataclass

from reactor_runtime.core import TrackDirection


@dataclass(frozen=True)
class TrackStat:
    """A single track's sampled transport statistics.

    Attributes:
        name: The track name the sample belongs to.
        direction: The track's flow direction, from the model's perspective.
        fps: Frames per second observed, or ``None`` when unavailable.
        bitrate_bps: Bits per second observed, or ``None`` when unavailable.
        packet_loss: Packets lost in the sampling window, or ``None``.
        jitter: Inter-arrival jitter in seconds, or ``None`` when unavailable.
    """

    name: str
    direction: TrackDirection
    fps: float | None = None
    bitrate_bps: int | None = None
    packet_loss: int | None = None
    jitter: float | None = None


@dataclass(frozen=True)
class PeerStats:
    """A snapshot of a peer's transport statistics.

    Attributes:
        rtt_seconds: Round-trip time in seconds, or ``None`` when unavailable.
        tracks: Per-track samples gathered in the same cycle.
    """

    rtt_seconds: float | None = None
    tracks: tuple[TrackStat, ...] = ()
