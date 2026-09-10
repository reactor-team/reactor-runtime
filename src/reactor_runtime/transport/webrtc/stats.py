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

    Every count is cumulative for the life of the track, so what one window cost
    is the difference between two samples. A reading the peer could not take
    arrives as ``None``, which is not the same as a reading of zero.

    Which fields carry a value follows from which side of the track this process
    is on. Outbound, it knows what it encoded and put on the wire, and it learns
    the loss second-hand from the receiver's own RTCP reports. Inbound, it is the
    receiver, so the loss and the jitter are its own measurements.

    Attributes:
        name: The track name the sample belongs to.
        direction: The track's flow direction, from the model's perspective.
        packets_sent: Packets put on the wire. Outbound only.
        packets_received: Packets taken off the wire. Inbound only.
        packets_lost: Packets that never arrived. Inbound, this process counted
            them; outbound, the receiver reported them, so they lag the wire by
            an RTCP round and stay ``None`` until the first report arrives.
        retransmitted_packets_sent: Packets sent again to repair a loss the
            receiver reported. Outbound only, and the cost of a lossy path that
            is still holding together.
        bytes_sent: Payload bytes put on the wire. Outbound only. The rate
            between two samples is the bitrate the track actually achieved,
            which is the figure to compare against the bandwidth estimate.
        bytes_received: Payload bytes taken off the wire. Inbound only.
        frames_sent: Video frames encoded and sent. Outbound only, and the rate
            between two samples is the frame rate that reached the wire — the
            one a viewer sees, as distinct from the rate the model produced.
        frames_decoded: Video frames decoded. Inbound only.
        frames_dropped: Video frames the decoder discarded. Inbound only.
        nacks: Retransmission requests the stream carried. Which end asked
            follows from the direction: inbound, this process asked the sender;
            outbound, the receiver asked this process. Either way it rises
            before the loss does, because a request answered in time repairs
            the stream — so a path going bad shows here while the picture is
            still intact.
        keyframe_requests: Requests to restart the stream from a fresh
            keyframe, which a decoder sends once repair has been outrun. The
            step past *nacks*: one asks for a packet again, this one says
            nothing further can be decoded. Picture Loss Indications and Full
            Intra Refresh requests are summed, because they are the same
            request in two codec dialects and no reader wants them apart.
        jitter: Inter-arrival jitter in seconds. Inbound only.
        rtt_seconds: Round trip of this stream, as the receiver measured it.
            Outbound only. This is the round trip media actually took, which is
            not the one ICE measures with its connectivity checks: the checks
            can keep succeeding on a path whose media queue has grown.
        loss_ratio: Fraction of this stream the receiver reports as lost,
            between ``0.0`` and ``1.0``. Outbound only, and a measurement rather
            than a figure derived here — it is the loss over the window the
            receiver's report covers, not over the sampling interval.
    """

    name: str
    direction: TrackDirection
    packets_sent: int | None = None
    packets_received: int | None = None
    packets_lost: int | None = None
    retransmitted_packets_sent: int | None = None
    bytes_sent: int | None = None
    bytes_received: int | None = None
    frames_sent: int | None = None
    frames_decoded: int | None = None
    frames_dropped: int | None = None
    nacks: int | None = None
    keyframe_requests: int | None = None
    jitter: float | None = None
    rtt_seconds: float | None = None
    loss_ratio: float | None = None


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
        rtt_seconds: Round trip of the ICE connectivity checks on the nominated
            pair, in seconds, or ``None`` when unavailable. This is the path's
            round trip and not the one media took — for that, read
            :attr:`TrackStat.rtt_seconds`, which the receiver measures on the
            stream itself and which the checks can under-report.
        available_outgoing_bitrate_bps: What congestion control believes the
            path to the client will carry, in bits per second, or ``None`` when
            the engine has no estimate yet. This is the number that separates a
            model that stopped producing frames from a network that stopped
            accepting them: read against the rate of *bytes_sent*, an estimate
            that collapsed while the send rate followed it down is the path
            giving way, and one that stayed high while the send rate fell is
            not.
        tracks: Per-track samples gathered in the same cycle.
        media: Cumulative counts of outbound media manufactured or discarded
            inside the runtime, which no transport-level statistic reports.
    """

    rtt_seconds: float | None = None
    available_outgoing_bitrate_bps: float | None = None
    tracks: tuple[TrackStat, ...] = ()
    media: OutboundMediaHealth = field(default_factory=OutboundMediaHealth)
