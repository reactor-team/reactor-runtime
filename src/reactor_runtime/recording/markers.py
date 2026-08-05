"""Media-time marker bookkeeping for the recorder.

Markers are seconds on the recorded media timeline, not wall-clock: the timeline
advances by the play-out duration of the frames actually fed to the encoder
(``n_frames / fps`` per chunk), so a clip's marker range and the recorded media
share one origin regardless of how fast or slow the model produced the frames.
The timeline starts at zero and only moves once the first real frame is fed.
"""

from __future__ import annotations

import threading


class MarkerBookkeeper:
    """Tracks how much media time has been recorded.

    Thread-safe: the clip-range math runs on the event loop while the timeline
    advances on the model thread as chunks are fed.
    """

    def __init__(self) -> None:
        """Start the timeline at zero, before any frame is fed."""
        self._media_time = 0.0
        self._started = False
        self._lock = threading.Lock()

    def now_marker(self) -> float:
        """Return the recorded media time in seconds."""
        with self._lock:
            return self._media_time

    def advance(self, seconds: float) -> None:
        """Add *seconds* of recorded media to the timeline and latch the start."""
        if seconds <= 0:
            return
        with self._lock:
            self._media_time += seconds
            self._started = True

    @property
    def first_real_frame_marker(self) -> float | None:
        """Return ``0.0`` once a frame has been fed, else ``None``.

        The media timeline is anchored at the first real frame, so the first
        frame is the origin.
        """
        with self._lock:
            return 0.0 if self._started else None

    @property
    def recording_started(self) -> bool:
        """Return whether any real frame has been fed."""
        with self._lock:
            return self._started

    def compute_clip_range(self, duration_seconds: float) -> tuple[float, float]:
        """Return ``(start, end)`` for a snap clip of *duration_seconds* ending now."""
        end_marker = self.now_marker()
        start_marker = max(0.0, end_marker - duration_seconds)
        return start_marker, end_marker

    def compute_recording_range(self) -> tuple[float, float]:
        """Return ``(0, now)`` for a full-session recording."""
        return 0.0, self.now_marker()
