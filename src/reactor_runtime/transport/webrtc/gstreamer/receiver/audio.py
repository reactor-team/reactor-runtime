
"""
Audio receiver: audio_decoder ! audioconvert ! capsfilter ! appsink.

Supports Opus via :class:`OpusDecoderBin` (rtpopusdepay ! opusdec). Other codecs
can be added to :class:`DecoderFactory` later.
"""

from reactor_runtime.transport.webrtc.gstreamer.decoders import DecoderFactory
from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    add_many,
    link_many,
    link_pads,
    make_element,
)

from .base import _ReceiverStreamBase


class AudioReceiver(_ReceiverStreamBase):
    """
    A Gst.Bin for audio receiver: audio_decoder ! audioconvert ! capsfilter ! appsink.

    The decoder is created via :class:`DecoderFactory` from the given
    encoding name. The bin exposes a ghost sink pad (RTP input) so it can be
    linked from a source pad (e.g. webrtcbin src). Use :meth:`set_on_new_sample`
    to set a callback when a new decoded frame is available.
    """

    def __init__(
        self,
        encoding_name: str,
        name: str = "audio_receiver",
    ):
        super().__init__(name=name)

        encoding_upper = encoding_name.upper()
        if encoding_upper == "OPUS":
            decoder = DecoderFactory.create("OPUS")
        else:
            raise ValueError(
                "Unsupported audio encoding-name for receiver: %r" % encoding_name
            )

        audioconvert = make_element("audioconvert", "consumer_audioconvert")
        capsfilter = make_element("capsfilter", "audio_caps")
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string("audio/x-raw,format=S16LE,rate=48000,channels=1"),
        )
        self._appsink = make_element("appsink", "appsink")
        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("sync", False)
        self._appsink.set_property("max-buffers", 1)
        self._appsink.set_property("wait-on-eos", False)
        self._appsink.connect("new-sample", self._on_new_sample)

        add_many(
            self,
            decoder,
            audioconvert,
            capsfilter,
            self._appsink,
            sync_with_parent=True,
        )
        link_many(audioconvert, capsfilter, self._appsink)
        link_pads(decoder.pad_src(), audioconvert.get_static_pad("sink"))

        decoder_sink = decoder.pad_sink()
        if not decoder_sink:
            raise RuntimeError("decoder has no sink pad")
        ghost_sink = Gst.GhostPad.new("sink", decoder_sink)
        if not ghost_sink or not self.add_pad(ghost_sink):
            raise RuntimeError("Failed to add ghost sink pad to AudioReceiver")
        self._ghost_sink: Gst.GhostPad = ghost_sink
