"""WebRTC peer statistics types.

Shared data types sampled by :class:`~reactor_runtime.transport.webrtc.peer.WebRTCPeer`
and surfaced through :class:`~reactor_runtime.transport.webrtc.connection.WebRTCConnection`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        packets_sent: Packets this outbound track has put on the wire, or
            ``None`` for an inbound track. Cumulative for the track's life, so
            two samples give the rate between them — the measure of whether
            outbound media is keeping up with real time.
    """

    name: str
    direction: TrackDirection
    fps: float | None = None
    bitrate_bps: int | None = None
    packet_loss: int | None = None
    jitter: float | None = None
    packets_sent: int | None = None


@dataclass(frozen=True)
class OutboundMediaHealth:
    """How much outbound media a peer manufactured or discarded, cumulatively.

    Outbound audio rides a sample clock: libwebrtc timestamps it by counting the
    samples it is handed, so a 10 ms frame the feeder cannot fill is 10 ms the
    stream never accounts for, and a sample discarded on the way to the wire
    pulls every later sample earlier against the video. Neither shows up as loss
    at the client — the packets that do arrive are contiguous — so these
    counters are the only place a session reports it, in the units it happened
    in.

    Attributes:
        silence_frames: 10 ms frames the audio feeder filled with silence
            because the outbound buffer had none ready.
        dropped_samples: Audio samples discarded to cap the outbound buffer.
        dropped_bundles: Media bundles discarded because the peer's frame queue
            was full.
        dropped_frames: Frames the pacer discarded because its queue was full.
    """

    silence_frames: int = 0
    dropped_samples: int = 0
    dropped_bundles: int = 0
    dropped_frames: int = 0


@dataclass(frozen=True)
class PeerStats:
    """A snapshot of a peer's transport statistics.

    Attributes:
        rtt_seconds: Round-trip time in seconds, or ``None`` when unavailable.
        tracks: Per-track samples gathered in the same cycle.
        media: Cumulative counts of outbound media manufactured or discarded
            inside the runtime, which no transport-level statistic reports.
    """

    rtt_seconds: float | None = None
    tracks: tuple[TrackStat, ...] = ()
    media: OutboundMediaHealth = field(default_factory=OutboundMediaHealth)
