"""Count what the model emits during a session, for the runtime's own stats.

The runtime reports its own view of a session alongside what clients report
about theirs. The model's output is the part of that view no transport sees:
how many frames it produced on each output track and how evenly it produced
them. :class:`ModelOutput` keeps those counts for the session that is running.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from reactor_runtime.core import ConnId, PeerStats, TrackDirection, TrackKind, TrackStat


class ModelOutput:
    """Per-session counts of the frames the model emitted, by output track.

    Emissions are counted from the thread the model emits on, and a reading is
    taken from the runtime's loop, so every access holds a lock. A reading
    covers the window since the previous one: the frame rate and the longest
    gap are what that window saw, while the frame total runs for the whole
    session. :meth:`reset` starts a new session from zero.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._tracks: tuple[str, ...] = ()
        self._frames: dict[str, int] = {}
        self._window_frames: dict[str, int] = {}
        self._last_emit: dict[str, float] = {}
        self._longest_gap: dict[str, float] = {}
        self._window_start = clock()

    def declare(self, tracks: Iterable[str]) -> None:
        """Name the output tracks the model declares.

        A declared track the model has emitted nothing on reads zero frames
        rather than being absent, which tells a silent track apart from a
        track this model does not have.
        """
        with self._lock:
            self._tracks = tuple(tracks)

    def reset(self) -> None:
        """Start counting a new session from zero.

        The gap between the last frame of one session and the first of the
        next is the model waiting for a client, so no gap crosses the reset.
        """
        with self._lock:
            self._frames.clear()
            self._window_frames.clear()
            self._last_emit.clear()
            self._longest_gap.clear()
            self._window_start = self._clock()

    def emitted(self, track: str, frames: int) -> None:
        """Count one emission of *frames* frames on *track*.

        Counted in frames because one emission can carry a batch of them. The
        gap is measured between emissions, undivided by the batch, so a model
        that emits a batch at a time has the play-out length of a batch as its
        normal gap and a stall shows above it.
        """
        now = self._clock()
        with self._lock:
            previous = self._last_emit.get(track)
            if previous is not None:
                gap = now - previous
                if gap > self._longest_gap.get(track, 0.0):
                    self._longest_gap[track] = gap
            self._last_emit[track] = now
            self._frames[track] = self._frames.get(track, 0) + frames
            self._window_frames[track] = self._window_frames.get(track, 0) + frames

    def take(self) -> dict[str, dict[str, float]]:
        """Read every output track's counts and start the next window.

        Returns:
            Per track name, its metrics: ``frames_emitted`` (the session's
            total), ``frames_per_second`` (over the window), and, once the track
            has emitted, ``max_emit_interval_ms`` (the longest gap the window
            saw, when it saw two emissions) and ``ms_since_last_emit``, which
            shows a stall that is still going before the next emission ends it.
        """
        now = self._clock()
        with self._lock:
            elapsed = now - self._window_start
            names = dict.fromkeys((*self._tracks, *self._frames))
            readings: dict[str, dict[str, float]] = {}
            for name in names:
                metrics: dict[str, float] = {
                    "frames_emitted": float(self._frames.get(name, 0)),
                }
                if elapsed > 0:
                    metrics["frames_per_second"] = self._window_frames.get(name, 0) / elapsed
                if name in self._longest_gap:
                    metrics["max_emit_interval_ms"] = self._longest_gap[name] * 1000.0
                last = self._last_emit.get(name)
                if last is not None:
                    metrics["ms_since_last_emit"] = (now - last) * 1000.0
                readings[name] = metrics
            self._window_frames.clear()
            self._longest_gap.clear()
            self._window_start = now
            return readings


INTERVAL_SECONDS = 5.0
"""How often the runner journals a ``runtime_stats`` reading during a session."""

# What the client sees a track as: what the model sends, the client receives.
_CLIENT_DIRECTION = {TrackDirection.OUT: "recvonly", TrackDirection.IN: "sendonly"}

_TrackKey = tuple[ConnId, str, TrackDirection]


class TransportRates:
    """Turn each track's cumulative byte count into a bitrate between readings.

    A transport sample counts bytes since the track started, so a bitrate is
    the difference from the previous sample over the time between the two,
    which the transport stamps on each sample (``taken_at``). A track's first
    sample, or one the transport has not replaced since the last reading, has
    no rate.
    """

    def __init__(self) -> None:
        self._previous: dict[_TrackKey, tuple[int, float]] = {}

    def bitrate(
        self, key: _TrackKey, total_bytes: int | None, taken_at: float | None
    ) -> float | None:
        """Return the bits per second of *key* since its previous sample."""
        if total_bytes is None or taken_at is None:
            return None
        previous = self._previous.get(key)
        self._previous[key] = (total_bytes, taken_at)
        if previous is None or taken_at <= previous[1] or total_bytes < previous[0]:
            return None
        return (total_bytes - previous[0]) * 8 / (taken_at - previous[1])

    def keep(self, conn_ids: Iterable[ConnId]) -> None:
        """Forget the tracks of every connection not in *conn_ids*."""
        live = set(conn_ids)
        self._previous = {key: value for key, value in self._previous.items() if key[0] in live}

    def clear(self) -> None:
        """Forget every track, at session end."""
        self._previous.clear()


def to_detail(
    *,
    observed_at_ms: int,
    model_output: Mapping[str, Mapping[str, float]],
    output_kinds: Mapping[str, TrackKind],
    samples: Iterable[tuple[ConnId, PeerStats]],
    rates: TransportRates,
) -> dict[str, Any]:
    """Shape one ``runtime_stats`` reading as a ``metric`` fact's detail.

    The shape follows a ``client_stats`` reading's, so a consumer reads both
    the same way: tracks carry ``track_name``, ``kind``, ``direction`` (as the
    client sees the track) and ``codec`` (``"VP9"``, ``"opus"``), and metric
    names match the client's where both measure the same thing. A metric value
    that isn't a finite number is left out, since JSON has no way to carry it.

    Args:
        observed_at_ms: When the reading was taken, in Unix milliseconds.
        model_output: Per output track, the metrics :meth:`ModelOutput.take`
            returned.
        output_kinds: The kind of each output track the model declares.
        samples: Each connection's latest transport sample.
        rates: The bitrates' running state, updated by this call.
    """
    samples = list(samples)
    rates.keep(cid for cid, _ in samples)
    return {
        "observed_at": observed_at_ms,
        "model_output": [
            {
                "track_name": name,
                "kind": str(output_kinds[name]) if name in output_kinds else "",
                "direction": _CLIENT_DIRECTION[TrackDirection.OUT],
                "metrics": _finite(metrics),
            }
            for name, metrics in model_output.items()
        ],
        "connection_stats": [_connection(cid, sample, rates) for cid, sample in samples],
    }


def _connection(cid: ConnId, sample: PeerStats, rates: TransportRates) -> dict[str, Any]:
    media = sample.media
    metrics = {
        "connection_rtt_ms": _ms(sample.rtt_seconds),
        "available_outgoing_bitrate_bps": sample.available_outgoing_bitrate_bps,
        "silence_frames": media.silence_frames,
        "dropped_samples": media.dropped_samples,
        "dropped_bundles": media.dropped_bundles,
        "dropped_frames": media.dropped_frames,
    }
    return {
        "conn_id": int(cid),
        "metrics": _finite(metrics),
        "track_stats": [_track(cid, track, sample.taken_at, rates) for track in sample.tracks],
    }


def _track(
    cid: ConnId, track: TrackStat, taken_at: float | None, rates: TransportRates
) -> dict[str, Any]:
    outbound = track.direction is TrackDirection.OUT
    total_bytes = track.bytes_sent if outbound else track.bytes_received
    metrics: dict[str, float | int | None] = {
        "bitrate_bps": rates.bitrate((cid, track.name, track.direction), total_bytes, taken_at),
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
    if track.kind is TrackKind.AUDIO:
        # The transport counts frames for video alone; an audio track's zero
        # there is not a measurement.
        for name in ("frames_sent", "frames_decoded", "frames_dropped"):
            metrics.pop(name, None)
    return {
        "track_name": track.name,
        "kind": str(track.kind) if track.kind is not None else "",
        "direction": _CLIENT_DIRECTION[track.direction],
        "codec": _codec_name(track.codec),
        "metrics": _finite(metrics),
    }


def _codec_name(mime_type: str | None) -> str:
    """Return the codec part of a mime type: ``"video/VP9"`` is ``"VP9"``."""
    if not mime_type:
        return ""
    return mime_type.rpartition("/")[2]


def _ms(seconds: float | None) -> float | None:
    return seconds * 1000.0 if seconds is not None else None


def _finite(metrics: Mapping[str, float | int | None]) -> dict[str, float]:
    """Keep the values JSON can carry: present, finite numbers."""
    return {
        name: float(value)
        for name, value in metrics.items()
        if value is not None and math.isfinite(value)
    }
