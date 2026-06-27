"""Wall-clock marker bookkeeping for the recorder.

Markers are wall-clock seconds aligned with the encoder's
``-use_wallclock_as_timestamps`` input. When ``anchor_at_first_frame`` is set
(the config's ``skip_leading_black``), ``t=0`` is the first real frame fed to the
encoder rather than recorder-arm time, so a clip's marker range and the recorded
media share one origin.
"""

from __future__ import annotations

import threading
import time


class MarkerBookkeeper:
    """Tracks the recording timeline origin and the first real frame.

    Thread-safe: the clip-range math runs on the event loop while the
    first-frame latch runs on the emission thread.
    """

    def __init__(self, *, anchor_at_first_frame: bool = False) -> None:
        """Start the timeline at construction, to be re-anchored on the first frame."""
        self._anchor_at_first_frame = anchor_at_first_frame
        self._session_start = time.monotonic()
        self._recording_start: float | None = None
        self._first_real_frame_marker: float | None = None
        self._lock = threading.Lock()

    def now_marker(self) -> float:
        """Return seconds since the active timeline origin."""
        with self._lock:
            origin = (
                self._recording_start if self._recording_start is not None else self._session_start
            )
        return time.monotonic() - origin

    @property
    def first_real_frame_marker(self) -> float | None:
        """Return the session-relative time of the first real frame, or ``None``."""
        with self._lock:
            return self._first_real_frame_marker

    @property
    def recording_started(self) -> bool:
        """Return whether a real frame has been seen.

        Always ``True`` when not anchoring at the first frame, since the timeline
        then runs from recorder-arm time.
        """
        if not self._anchor_at_first_frame:
            return True
        with self._lock:
            return self._recording_start is not None

    def mark_first_real_frame(self) -> None:
        """Latch the first real frame, re-anchoring the timeline when configured."""
        with self._lock:
            if self._first_real_frame_marker is None:
                self._first_real_frame_marker = time.monotonic() - self._session_start
            if self._anchor_at_first_frame and self._recording_start is None:
                self._recording_start = time.monotonic()

    def compute_clip_range(self, duration_seconds: float) -> tuple[float, float]:
        """Return ``(start, end)`` for a snap clip of *duration_seconds* ending now."""
        end_marker = self.now_marker()
        start_marker = max(0.0, end_marker - duration_seconds)
        return start_marker, end_marker

    def compute_recording_range(self) -> tuple[float, float]:
        """Return ``(0, now)`` for a full-session recording."""
        return 0.0, self.now_marker()
