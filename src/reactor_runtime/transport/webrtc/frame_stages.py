"""Add up where a connection's frames spend their time, stage by stage.

A frame crosses several stages on this side of the wire. Some the runtime times
itself, frame by frame: ``delivery`` (decoder to the runner), ``output_pacing``
(the output pacer's queue) and ``output_queue`` (the pacer to libwebrtc).
libwebrtc times others as running totals, which a stats sample turns into a
window's worth: ``encode`` on an outbound track, ``jitter_buffer`` and
``decode`` on an inbound one. On an inbound video track, libwebrtc also
carries the sender's ``encode_wait``, ``packetize`` and ``pacer`` inside the
video, as timing frames, so the client's own stages show up here as samples.

:class:`FrameStageWindow` holds all of them until a report takes them.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping

from reactor_runtime.core import FrameStage, TrackDirection
from reactor_runtime.transport.webrtc.stats import PeerStats, TrackStat

STAGES = (
    "submit",
    "encode_wait",
    "encode",
    "packetize",
    "pacer",
    "jitter_buffer",
    "decode",
    "delivery",
    "output_pacing",
    "output_queue",
)
"""Every stage, in the order a frame goes through them."""

_ORDER = {name: index for index, name in enumerate(STAGES)}


class FrameStageWindow:
    """The time each track's frames spent in each stage, since the last take.

    Frames are timed from the threads that move them, and a report takes the
    window from the event loop, so every access holds a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[tuple[str, str], tuple[float, int]] = {}
        # The last timing frame folded in, per track, so one that a later
        # sample repeats is counted once.
        self._timing_frames: dict[str, int] = {}

    def add(self, track: str, stage: str, ms: float, frames: int = 1) -> None:
        """Add *frames* frames that spent *ms* milliseconds in *stage*, together.

        A count below one or a time that is negative or not finite is no
        measurement, and is left out.
        """
        if frames < 1 or not math.isfinite(ms) or ms < 0:
            return
        key = (track, stage)
        with self._lock:
            total, count = self._totals.get(key, (0.0, 0))
            self._totals[key] = (total + ms, count + frames)

    def fold(self, sample: PeerStats, previous: PeerStats | None) -> None:
        """Add the stages libwebrtc timed between *previous* and *sample*.

        libwebrtc keeps running totals, so a window's worth is the difference
        between two samples. A track with no previous sample adds nothing,
        and neither does a total that went backwards, which is a restarted
        stream rather than a measurement.
        """
        before = {(t.name, t.direction): t for t in previous.tracks} if previous else {}
        for track in sample.tracks:
            earlier = before.get((track.name, track.direction))
            if track.direction is TrackDirection.OUT:
                self._add_delta(
                    track.name, "encode", track, earlier, "encode_seconds", "frames_encoded"
                )
            else:
                self._add_delta(
                    track.name,
                    "jitter_buffer",
                    track,
                    earlier,
                    "jitter_buffer_seconds",
                    "jitter_buffer_frames",
                )
                self._add_delta(
                    track.name, "decode", track, earlier, "decode_seconds", "frames_decoded"
                )
                self._add_timing_frame(track)

    def take(self) -> dict[str, tuple[FrameStage, ...]]:
        """Return each track's stages, in trip order, and start a new window."""
        with self._lock:
            totals, self._totals = self._totals, {}
        stages: dict[str, list[FrameStage]] = {}
        for (track, stage), (total, count) in totals.items():
            stages.setdefault(track, []).append(FrameStage(stage, total_ms=total, frames=count))
        return {
            track: tuple(sorted(found, key=lambda s: _ORDER.get(s.name, len(_ORDER))))
            for track, found in stages.items()
        }

    def _add_delta(
        self,
        track: str,
        stage: str,
        sample: TrackStat,
        earlier: TrackStat | None,
        seconds_field: str,
        frames_field: str,
    ) -> None:
        if earlier is None:
            return
        seconds, frames = getattr(sample, seconds_field), getattr(sample, frames_field)
        seconds_before = getattr(earlier, seconds_field)
        frames_before = getattr(earlier, frames_field)
        if None in (seconds, frames, seconds_before, frames_before):
            return
        self.add(track, stage, (seconds - seconds_before) * 1000.0, frames - frames_before)

    def _add_timing_frame(self, track: TrackStat) -> None:
        timing = track.timing_frame
        if timing is None:
            return
        with self._lock:
            if self._timing_frames.get(track.name) == timing.rtp_timestamp:
                return
            self._timing_frames[track.name] = timing.rtp_timestamp
        spans: Mapping[str, float] = {
            "encode_wait": timing.encode_wait_ms,
            "packetize": timing.packetize_ms,
            "pacer": timing.pacer_ms,
        }
        for stage, ms in spans.items():
            self.add(track.name, stage, ms)
