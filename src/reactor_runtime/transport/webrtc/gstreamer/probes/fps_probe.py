
"""
Buffer-count FPS probes on GStreamer pads.
Each :class:`FpsProbe` instance is independent: attach one probe per pad /
measurement site (e.g. after ``videoconvert``, after an encoder, on a queue src pad).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger

logger = get_logger(__name__)


class FpsProbe:
    """
    Install a ``Gst.PadProbeType.BUFFER`` probe that reports buffers per second
    over a rolling wall-clock window (default 1s of monotonic time).
    Instances do not share state. Typical usage::
        probe_videoconvert = FpsProbe("video_after_videoconvert")
        probe_videoconvert.attach(videoconvert.get_static_pad("src"))
        probe_encoder = FpsProbe("video_after_encoder")
        probe_encoder.attach(encoder_element.get_static_pad("src"))
    ``last_fps`` is updated at the end of each completed window. Each completed
    window is logged at INFO.
    """

    def __init__(
        self,
        name: str,
        *,
        window_ns: Optional[int] = None,
    ) -> None:
        self._name = name
        self._window_ns = int(Gst.SECOND if window_ns is None else window_ns)
        if self._window_ns <= 0:
            raise ValueError("window_ns must be positive")

        self._pad: Optional[Gst.Pad] = None
        self._probe_id: Optional[int] = None
        self._window_start_ns: Optional[int] = None
        self._count = 0
        self._last_fps: Optional[float] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_fps(self) -> Optional[float]:
        """Buffers per second from the last completed ``window_ns`` interval."""
        return self._last_fps

    def attach(self, pad: Gst.Pad) -> None:
        """
        Add this probe on *pad* (usually a src pad, downstream of the element under test).
        Raises:
            RuntimeError: if this instance already has a probe installed.
        """
        if self._probe_id:
            raise RuntimeError(
                f"FpsProbe({self._name!r}) is already attached; call detach() first"
            )
        self._pad = pad
        self._window_start_ns = None
        self._count = 0
        self._probe_id = pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._buffer_probe,
            None,  # user_data (PyGObject passes this as the 3rd callback arg)
        )
        if not self._probe_id:
            self._pad = None
            raise RuntimeError(f"FpsProbe({self._name!r}): pad.add_probe() failed")

    def detach(self) -> None:
        """Remove the probe from the pad, if installed."""
        if self._pad is not None and self._probe_id is not None:
            self._pad.remove_probe(self._probe_id)
        self._pad = None
        self._probe_id = None
        self._window_start_ns = None
        self._count = 0
        self._last_fps = None

    def _buffer_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        _user_data: Any,
    ) -> Gst.PadProbeReturn:
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        now = time.monotonic_ns()
        window_start = self._window_start_ns
        if window_start is None:
            self._window_start_ns = now
            return Gst.PadProbeReturn.OK

        self._count += 1
        elapsed = now - window_start
        if elapsed >= self._window_ns:
            # buffers / second for the window that just ended
            fps = self._count * 1_000_000_000 / elapsed
            self._last_fps = fps
            self._window_start_ns = now
            self._count = 0

        return Gst.PadProbeReturn.OK
