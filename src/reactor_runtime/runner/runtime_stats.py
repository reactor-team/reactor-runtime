"""Count what the model emits and receives during a session, for the runtime's own stats.

The runtime reports its own view of a session alongside what clients report
about theirs. The model's media is the part of that view no transport sees:
how many frames it produced on each output track and how evenly it produced
them, which :class:`ModelOutput` keeps, and how many frames each input track
received and dropped, which :class:`ModelInput` keeps.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from reactor_runtime.core import (
    ClientTrackDirection,
    ConnId,
    TrackKind,
    TransportReading,
    TransportTrackReading,
)
from reactor_runtime.runner.client_stats import frame_stages_detail


class ModelOutput:
    """Per-session counts of the frames the model emitted, by output track.

    Emissions are counted from the thread the model emits on, and a reading is
    taken from the runtime's loop, so every access holds a lock. A reading
    covers the window since the previous one: the frame rate and the longest
    gap are what that window saw, while the frame total runs for the whole
    session. :meth:`reset` starts a new session from zero, and an emission
    stamped with another session's number is left out of its count.
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
        self._session: int | None = None

    def declare(self, tracks: Iterable[str]) -> None:
        """Name the output tracks the model declares.

        A declared track the model has emitted nothing on reads zero frames
        rather than being absent, which tells a silent track apart from a
        track this model does not have.
        """
        with self._lock:
            self._tracks = tuple(tracks)

    def reset(self, session: int | None = None) -> None:
        """Start counting *session* from zero.

        The gap between the last frame of one session and the first of the
        next is the model waiting for a client, so no gap crosses the reset.

        Args:
            session: The number of the session that starts, as the model counts
                them. From here on, an emission stamped with any other number
                belongs to another session and is not counted.
        """
        with self._lock:
            self._session = session
            self._frames.clear()
            self._window_frames.clear()
            self._last_emit.clear()
            self._longest_gap.clear()
            self._window_start = self._clock()

    def emitted(self, track: str, frames: int, session: int | None = None) -> None:
        """Count one emission of *frames* frames on *track*.

        Counted in frames because one emission can carry a batch of them. The
        gap is measured between emissions, undivided by the batch, so a model
        that emits a batch at a time has the play-out length of a batch as its
        normal gap and a stall shows above it.

        Args:
            track: The output track the frames were emitted on.
            frames: How many frames the emission carried.
            session: The number of the session the model emitted them in, or
                ``None`` when the emitter does not count sessions. An emission
                of a session other than the one :meth:`reset` started is not
                counted. The check runs under the same lock as the reset, so
                an emission is counted for the session that is current when
                it lands.
        """
        with self._lock:
            if session is not None and self._session is not None and session != self._session:
                return
            # Read under the lock, so concurrent emissions store their times
            # in the order they were taken.
            now = self._clock()
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
        with self._lock:
            now = self._clock()
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


class ModelInput:
    """Per-session counts of the frames the model's input tracks received and dropped.

    A track's buffer drops its oldest frames when it is full, so a model that
    reads too slowly loses input without seeing it. Counting the drops here
    makes the loss visible. Frames arrive from the transport's threads and a
    reading is taken on the runtime's loop, so every access holds a lock. Both
    counts run for the whole session; :meth:`reset` starts the next one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tracks: tuple[str, ...] = ()
        self._received: dict[str, int] = {}
        self._dropped: dict[str, int] = {}

    def declare(self, tracks: Iterable[str]) -> None:
        """Name the input tracks the model declares, so an idle track reads zero."""
        with self._lock:
            self._tracks = tuple(tracks)

    def reset(self) -> None:
        """Start counting a new session from zero."""
        with self._lock:
            self._received.clear()
            self._dropped.clear()

    def received(self, track: str, dropped: int) -> None:
        """Count one frame arriving on *track*, and the *dropped* frames it pushed out."""
        with self._lock:
            self._received[track] = self._received.get(track, 0) + 1
            if dropped:
                self._dropped[track] = self._dropped.get(track, 0) + dropped

    def take(self) -> dict[str, dict[str, float]]:
        """Read every input track's counts.

        Returns:
            Per track name, ``frames_received`` and ``frames_dropped``, each the
            session's total.
        """
        with self._lock:
            names = dict.fromkeys((*self._tracks, *self._received))
            return {
                name: {
                    "frames_received": float(self._received.get(name, 0)),
                    "frames_dropped": float(self._dropped.get(name, 0)),
                }
                for name in names
            }


INTERVAL_SECONDS = 5.0
"""How often the runner journals a ``runtime_stats`` reading during a session."""


def to_detail(
    *,
    observed_at_ms: int,
    model_output: Mapping[str, Mapping[str, float]],
    output_kinds: Mapping[str, TrackKind],
    readings: Iterable[tuple[ConnId, TransportReading]],
    model_input: Mapping[str, Mapping[str, float]] | None = None,
    input_kinds: Mapping[str, TrackKind] | None = None,
) -> dict[str, Any]:
    """Shape one ``runtime_stats`` reading as a ``metric`` fact's detail.

    The shape follows a ``client_stats`` reading's, so a consumer reads both
    the same way: tracks carry ``track_name``, ``kind``, ``direction`` (as the
    client sees the track) and ``codec``, beside their ``metrics``. A metric
    value that isn't a finite number is left out, since JSON has no way to
    carry it.

    Args:
        observed_at_ms: When the reading was taken, in Unix milliseconds.
        model_output: Per output track, the metrics :meth:`ModelOutput.take`
            returned.
        output_kinds: The kind of each output track the model declares.
        readings: Each connection's latest transport reading.
        model_input: Per input track, the metrics :meth:`ModelInput.take`
            returned. A model with no input tracks has none.
        input_kinds: The kind of each input track the model declares.
    """
    input_kinds = input_kinds or {}
    return {
        "observed_at": observed_at_ms,
        "model_input": [
            {
                "track_name": name,
                "kind": str(input_kinds[name]) if name in input_kinds else "",
                # What the model receives, the client sends.
                "direction": str(ClientTrackDirection.SENDONLY),
                "metrics": _finite(metrics),
            }
            for name, metrics in (model_input or {}).items()
        ],
        "model_output": [
            {
                "track_name": name,
                "kind": str(output_kinds[name]) if name in output_kinds else "",
                # What the model sends, the client receives.
                "direction": str(ClientTrackDirection.RECVONLY),
                "metrics": _finite(metrics),
            }
            for name, metrics in model_output.items()
        ],
        "connection_stats": [
            {
                "conn_id": int(cid),
                "metrics": _finite(reading.metrics),
                "track_stats": [_track(track) for track in reading.tracks],
            }
            for cid, reading in readings
        ],
    }


def _track(track: TransportTrackReading) -> dict[str, Any]:
    return {
        "track_name": track.track_name,
        "kind": str(track.kind) if track.kind is not None else "",
        "direction": str(track.direction),
        "codec": track.codec,
        "metrics": _finite(track.metrics),
        "frame_stages": frame_stages_detail(track.frame_stages),
    }


def _finite(metrics: Mapping[str, float]) -> dict[str, float]:
    """Keep the values JSON can carry: finite numbers."""
    return {name: float(value) for name, value in metrics.items() if math.isfinite(value)}
