
"""
Audio sender: appsrc ! audioconvert ! audiorate ! encoder.

Audio is paced to the pipeline clock via appsrc need-data: push_buffer() enqueues
20ms chunks, and we push one chunk per need-data so the receiver gets real-time
rate (avoids insertedSamplesForDeceleration and discontinuous sound).
"""

from collections import deque
from typing import Optional, Union

import numpy as np

from reactor_runtime.transport.webrtc.gstreamer.encoders import EncoderFactory
from reactor_runtime.transport.webrtc.gstreamer.sdp.codec import CodecEntry
from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    add_many,
    link_many,
    link_pads,
    make_element,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger

from .base import _SenderStreamBase

logger = get_logger(__name__)


class AudioSender(_SenderStreamBase):
    """
    A Gst.Bin for audio sender: appsrc ! audioconvert ! audiorate ! encoder.

    The encoder is created via :class:`EncoderFactory`. The bin exposes
    a ghost pad on the encoder's src so it can be linked downstream
    (e.g. to webrtcbin).
    """

    def __init__(
        self,
        codec_entry: CodecEntry,
        name: str = "audio_sender",
        ssrc: Optional[int] = None,
        rate: int = 48000,
        channels: int = 1,
    ):
        super().__init__(name=name)

        encoding_name = codec_entry.get("codec")
        pt = codec_entry.get("payload_type")
        if not encoding_name or pt is None:
            raise ValueError("codec_entry must contain 'codec' and 'payload_type'")
        format_params = codec_entry.get("parameters") or {}

        self._codec_name: str = encoding_name
        self._rate = int(rate)
        self._channels = int(channels)

        # 20ms chunk sizing (S16LE: 2 bytes per sample per channel)
        self._samples_per_20ms = int(self._rate * 0.02)
        self._bytes_per_sample = 2 * self._channels
        self._bytes_per_20ms = self._samples_per_20ms * self._bytes_per_sample
        self._duration_20ms_ns = int((self._samples_per_20ms / self._rate) * Gst.SECOND)
        # Queue of 20ms chunk bytes; pushed to appsrc only in need-data (paced to pipeline)
        self._chunk_queue: deque = deque(maxlen=50)  # cap ~1s to avoid latency buildup

        self._appsrc = make_element("appsrc", "appsrc")
        try_set_property(self._appsrc, "format", Gst.Format.TIME)
        try_set_property(self._appsrc, "is-live", True)
        try_set_property(self._appsrc, "do-timestamp", True)
        try_set_property(self._appsrc, "block", False)
        try_set_property(self._appsrc, "min-percent", 0)
        try_set_property(
            self._appsrc, "max-bytes", self._bytes_per_20ms * 5
        )  # ~100ms queue
        self._appsrc.connect("need-data", self._on_need_data)
        # Set caps before pipeline runs so pad negotiation can complete (required for audio to flow).
        # layout=interleaved is required by audioconvert (GStreamer audio-info expects a layout).
        appsrc_caps = Gst.Caps.from_string(
            f"audio/x-raw,format=S16LE,rate={self._rate},channels={self._channels},layout=interleaved"
        )
        self._appsrc.set_property("caps", appsrc_caps)

        audioconvert = make_element("audioconvert", "audioconvert")
        audiorate = make_element("audiorate", "audiorate")
        self._encoder = EncoderFactory.create(
            encoding_name, pt, format_params, ssrc=ssrc
        )

        add_many(
            self,
            self._appsrc,
            audioconvert,
            audiorate,
            self._encoder,
            sync_with_parent=True,
        )
        link_many(self._appsrc, audioconvert, audiorate)
        link_pads(audiorate.get_static_pad("src"), self._encoder.pad_sink())

        encoder_src = self._encoder.pad_src()
        if not encoder_src:
            raise RuntimeError("encoder has no src pad")
        ghost_src = Gst.GhostPad.new("src", encoder_src)
        if not ghost_src or not self.add_pad(ghost_src):
            raise RuntimeError("Failed to add ghost src pad to AudioSender")
        self._ghost_src: Gst.GhostPad = ghost_src

    def _on_need_data(self, _src: Gst.Element, _length: int) -> None:
        """Push one 20ms chunk when the pipeline asks for data (keeps real-time pace)."""
        if self._chunk_queue:
            chunk_bytes = self._chunk_queue.popleft()
            chunk_samples = len(chunk_bytes) // self._bytes_per_sample
            duration_ns = int((chunk_samples / self._rate) * Gst.SECOND)
            buf = Gst.Buffer.new_wrapped(bytes(chunk_bytes))
            buf.duration = duration_ns
        else:
            # Push silence to avoid underrun and keep stream continuous
            buf = Gst.Buffer.new_wrapped(bytes(self._bytes_per_20ms))
            buf.duration = self._duration_20ms_ns
        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            logger.warning(f"AudioSender._on_need_data: push-buffer returned {ret}")

    @property
    def codec_name(self) -> str:
        return self._codec_name

    @property
    def current_bitrate_kbps(self) -> int:
        """Configured encoder bitrate in kbps."""
        return self._encoder.get_bitrate_kbps()

    def push_buffer(self, data: Union[bytes, np.ndarray]) -> bool:
        """
        Enqueue raw S16LE audio as 20ms chunks; they are pushed to appsrc in
        need-data so playback is paced to the pipeline clock (real-time).

        Caps are set in __init__ (rate/channels must match). Must be called
        from the thread that runs the GStreamer main loop.

        Args:
            data: Raw S16LE audio as bytes, or a numpy array of dtype int16
                  (shape (n_samples,) or (n_samples, channels)).

        Returns:
            True if the buffer was pushed successfully (Gst.FlowReturn.OK),
            False otherwise or if data cannot be interpreted as S16LE.
        """
        try:
            if isinstance(data, np.ndarray):
                if data.dtype != np.int16:
                    logger.debug(
                        "AudioSender.push_buffer: rejected non-int16 array",
                        dtype=getattr(data.dtype, "name", data.dtype),
                    )
                    return False
                raw_bytes = data.tobytes()
            else:
                raw_bytes = bytes(data)
        except Exception as e:
            logger.debug("AudioSender.push_buffer: failed to get bytes", error=str(e))
            return False

        if not raw_bytes:
            logger.debug("AudioSender.push_buffer: empty data")
            return False

        # S16LE: 2 bytes per sample per channel
        num_samples = len(raw_bytes) // 2 // self._channels
        if num_samples == 0:
            logger.debug(
                "AudioSender.push_buffer: no full samples",
                length=len(raw_bytes),
                channels=self._channels,
            )
            return False

        # Enqueue 20ms chunks; they are pushed to appsrc in _on_need_data (paced to pipeline clock)
        offset = 0
        while offset < len(raw_bytes):
            chunk_bytes = raw_bytes[offset : offset + self._bytes_per_20ms]
            offset += len(chunk_bytes)
            if len(chunk_bytes) < self._bytes_per_sample:
                continue

            if len(self._chunk_queue) == self._chunk_queue.maxlen:
                logger.warning(
                    "AudioSender: chunk queue full, the oldest chunk will be dropped"
                )

            self._chunk_queue.append(bytes(chunk_bytes))
        return True
