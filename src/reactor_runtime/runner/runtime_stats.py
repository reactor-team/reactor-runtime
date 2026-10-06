"""Count what the model emits during a session, for the runtime's own stats.

The runtime reports its own view of a session alongside what clients report
about theirs. The model's output is the part of that view no transport sees:
how many frames it produced on each output track and how evenly it produced
them. :class:`ModelOutput` keeps those counts for the session that is running.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable


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
