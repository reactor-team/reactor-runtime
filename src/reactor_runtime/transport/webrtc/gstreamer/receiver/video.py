
"""
Video receiver: video_decoder ! videoconvert ! capsfilter ! appsink.
"""

from typing import Optional

from reactor_runtime.transport.webrtc.gstreamer.decoders import DecoderFactory
from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    add_many,
    link_many,
    link_pads,
    make_element,
)
from reactor_runtime.transport.webrtc.gstreamer.probes.fps_probe import FpsProbe

from .base import _ReceiverStreamBase


class VideoReceiver(_ReceiverStreamBase):
    """
    A Gst.Bin for video receiver: video_decoder ! videoconvert ! capsfilter ! appsink.

    The decoder is created via :class:`DecoderFactory` from the given
    encoding name. The bin exposes a ghost sink pad (decoder RTP input)
    so it can be linked from a source pad (e.g. webrtcbin src). Use
    :meth:`set_on_new_sample` to set a callback when a new decoded frame
    is available.
    """

    def __init__(
        self,
        encoding_name: str,
        name: str = "video_receiver",
    ):
        super().__init__(name=name)

        decoder = DecoderFactory.create(encoding_name)
        videoconvert = make_element("videoconvert", "consumer_videoconvert")
        capsfilter = make_element("capsfilter", "rgb_caps")
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))
        self._appsink = make_element("appsink", "appsink")
        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("sync", False)
        self._appsink.set_property("max-buffers", 1)
        self._appsink.set_property("wait-on-eos", False)
        self._appsink.connect("new-sample", self._on_new_sample)

        add_many(
            self,
            decoder,
            videoconvert,
            capsfilter,
            self._appsink,
            sync_with_parent=True,
        )
        link_many(videoconvert, capsfilter, self._appsink)
        link_pads(decoder.pad_src(), videoconvert.get_static_pad("sink"))

        decoder_sink = decoder.pad_sink()
        if not decoder_sink:
            raise RuntimeError("decoder has no sink pad")
        ghost_sink = Gst.GhostPad.new("sink", decoder_sink)
        if not ghost_sink or not self.add_pad(ghost_sink):
            raise RuntimeError("Failed to add ghost sink pad to VideoReceiver")
        self._ghost_sink: Gst.GhostPad = ghost_sink

        self._fps_probe = FpsProbe(f"{name}_decoded")
        self._fps_probe.attach(self._appsink.get_static_pad("sink"))

        self.codec_name: str = encoding_name
        self.last_width: Optional[int] = None
        self.last_height: Optional[int] = None

    @property
    def fps(self) -> Optional[float]:
        """Decoded frames per second from the last completed measurement window."""
        return self._fps_probe.last_fps
