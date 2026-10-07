"""Translate a WebRTC stats sample into a transport-neutral reading.

The connection samples its peer every few seconds (:mod:`.stats`), in
libwebrtc's own terms. A reader outside the transport holds a
:class:`~reactor_runtime.core.TransportReading` instead, so this is where the
WebRTC fields become named metrics. The names match a client's own reading
where both measure the same thing, the way the client's batch is read next
door (:mod:`.client_stats`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from reactor_runtime.core import (
    ClientTrackDirection,
    TrackDirection,
    TransportReading,
    TransportTrackReading,
)
from reactor_runtime.transport.webrtc.stats import PeerStats, TrackStat

# What the client sees a track as: what the model sends, the client receives.
_CLIENT_DIRECTION = {
    TrackDirection.OUT: ClientTrackDirection.RECVONLY,
    TrackDirection.IN: ClientTrackDirection.SENDONLY,
}


def transport_reading(
    sample: PeerStats, previous: PeerStats | None, elapsed_s: float | None
) -> TransportReading:
    """Return *sample* as a neutral reading.

    A track's byte counts are totals since it started, so its bitrate is the
    difference from the *previous* sample over *elapsed_s*, the time between
    the two. A track with no previous sample, or a total that went backwards,
    has no bitrate in this reading.
    """
    before: dict[tuple[str, TrackDirection], int | None] = {}
    if previous is not None and elapsed_s is not None and elapsed_s > 0:
        before = {(t.name, t.direction): _total_bytes(t) for t in previous.tracks}
    media = sample.media
    return TransportReading(
        metrics=_finite(
            {
                "connection_rtt_ms": _ms(sample.rtt_seconds),
                "available_outgoing_bitrate_bps": sample.available_outgoing_bitrate_bps,
                "silence_frames": media.silence_frames,
                "dropped_samples": media.dropped_samples,
                "dropped_bundles": media.dropped_bundles,
                "dropped_frames": media.dropped_frames,
            }
        ),
        tracks=tuple(
            _track(track, before.get((track.name, track.direction)), elapsed_s)
            for track in sample.tracks
        ),
    )


def _track(track: TrackStat, before: int | None, elapsed_s: float | None) -> TransportTrackReading:
    outbound = track.direction is TrackDirection.OUT
    total = _total_bytes(track)
    bitrate = None
    if total is not None and before is not None and elapsed_s and total >= before:
        bitrate = (total - before) * 8 / elapsed_s
    metrics: dict[str, float | int | None] = {
        "bitrate_bps": bitrate,
        "packets_lost": track.packets_lost,
        "nack_count": track.nacks,
        "keyframe_requests": track.keyframe_requests,
        "frames_per_second": track.frames_per_second,
        "frame_width": track.frame_width,
        "frame_height": track.frame_height,
    }
    if outbound:
        metrics |= {
            "packets_sent": track.packets_sent,
            "retransmitted_packets_sent": track.retransmitted_packets_sent,
            "frames_sent": track.frames_sent,
            "target_bitrate_bps": track.target_bitrate_bps,
            "round_trip_time_ms": _ms(track.rtt_seconds),
            "loss_ratio": track.loss_ratio,
        }
    else:
        metrics |= {
            "packets_received": track.packets_received,
            "frames_decoded": track.frames_decoded,
            "frames_dropped": track.frames_dropped,
            "jitter_ms": _ms(track.jitter),
        }
    return TransportTrackReading(
        track_name=track.name,
        kind=track.kind,
        direction=_CLIENT_DIRECTION[track.direction],
        codec=_codec_name(track.codec),
        metrics=_finite(metrics),
    )


def _total_bytes(track: TrackStat) -> int | None:
    return track.bytes_sent if track.direction is TrackDirection.OUT else track.bytes_received


def _codec_name(mime_type: str | None) -> str:
    """Return the codec part of a mime type: ``"video/VP9"`` is ``"VP9"``."""
    if not mime_type:
        return ""
    return mime_type.rpartition("/")[2]


def _ms(seconds: float | None) -> float | None:
    return seconds * 1000.0 if seconds is not None else None


def _finite(metrics: Mapping[str, float | int | None]) -> dict[str, float]:
    """Keep the values a reading can carry: present, finite numbers."""
    return {
        name: float(value)
        for name, value in metrics.items()
        if value is not None and math.isfinite(value)
    }
