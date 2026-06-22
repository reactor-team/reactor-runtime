from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.settings import HW_CODECS_ENABLED
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger
from .base import BaseRTPDecoderBin

logger = get_logger(__name__)


class VP8DecoderBin(BaseRTPDecoderBin):
    """
    RTP decoder bin implementing:

        sink (RTP VP8) ->
        rtpvp8depay ->
        decoder (HW preferred, SW fallback) ->
        src (raw video)

    Designed for use with:
        - webrtcbin src pads
        - rtpbin receive chains
        - Custom RTP demux pipelines

    Low-latency focus:
        - Prefer NVIDIA hardware decoder when available (nvvp8dec)
        - Otherwise fall back to software decoder (vp8dec)

    VP8 is widely supported and generally robust in WebRTC scenarios.
    """

    def __init__(self, name: str = "vp8_decoder_bin"):
        super().__init__(name=name)

        # ---------------------------------------------------------
        # RTP depayloader
        # ---------------------------------------------------------
        # Converts RTP packets into elementary VP8 bitstream.
        self._depay = make_element("rtpvp8depay", "rtpvp8depay")
        try_set_property(self._depay, "wait-for-keyframe", True)
        try_set_property(self._depay, "request-keyframe", True)

        # ---------------------------------------------------------
        # Select decoder implementation
        # ---------------------------------------------------------
        self._dec_factory = self._pick_decoder_factory()
        self._dec = make_element(self._dec_factory, self._dec_factory)

        # ---------------------------------------------------------
        # Best-effort low-latency tuning
        # ---------------------------------------------------------
        # Software decoder (vp8dec) benefits from threading; HW decoders ignore.
        try_set_property(self._dec, "threads", 4)

        # ---------------------------------------------------------
        # Build internal pipeline
        # ---------------------------------------------------------
        self.add(self._depay)
        self.add(self._dec)

        self._link_or_raise(
            self._depay,
            self._dec,
            "rtpvp8depay -> %s" % self._dec_factory,
        )

        # ---------------------------------------------------------
        # Expose ghost pads
        # ---------------------------------------------------------
        # sink  → RTP input
        # src   → raw decoded video output
        depay_sink = self._depay.get_static_pad("sink")
        dec_src = self._dec.get_static_pad("src")

        if not depay_sink or not dec_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        # Ghost pads allow this bin to behave like:
        #     RTP in → raw video out
        self._create_ghost_pads(depay_sink, dec_src)

    def _pick_decoder_factory(self) -> str:
        """
        Select best available VP8 decoder.

        Priority:
            1) NVIDIA hardware decoder (nvvp8dec)
            2) vp8dec (libvpx software decoder)
        """
        if HW_CODECS_ENABLED and Gst.ElementFactory.find("nvvp8dec") is not None:
            logger.info(
                "VP8DecoderBin: using hardware (HW) decoder: nvvp8dec",
            )
            return "nvvp8dec"
        if Gst.ElementFactory.find("vp8dec") is not None:
            logger.info(
                "VP8DecoderBin: using software (SW) decoder: vp8dec",
            )
            return "vp8dec"
        raise RuntimeError("No VP8 decoder available (tried: nvvp8dec, vp8dec)")
