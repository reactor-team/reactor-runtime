from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.settings import HW_CODECS_ENABLED
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger
from .base import BaseRTPDecoderBin

logger = get_logger(__name__)


class VP9DecoderBin(BaseRTPDecoderBin):
    """
    RTP decoder bin implementing:

        sink (RTP VP9) ->
        rtpvp9depay ->
        decoder (HW preferred, SW fallback) ->
        src (raw video)

    Designed for WebRTC receive pipelines.

    Low-latency focus:
        - Prefer NVIDIA hardware decoder when available (nvvp9dec)
        - Otherwise fall back to software decoder (vp9dec)

    Notes:
        - VP9 profile handling is implicit: depayloader extracts profile
          information from RTP payload and passes it to the decoder.
        - Decoder must tolerate dynamic resolution changes.
    """

    def __init__(self, name: str = "vp9_decoder_bin"):
        super().__init__(name=name)

        # ---------------------------------------------------------
        # RTP depayloader
        # ---------------------------------------------------------
        # Converts RTP VP9 packets into VP9 elementary bitstream.
        # Handles RTP-specific framing and aggregation.
        self._depay = make_element("rtpvp9depay", "rtpvp9depay")
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
        # Software decoder (vp9dec) benefits from threading; HW decoders ignore.
        try_set_property(self._dec, "threads", 8)

        # ---------------------------------------------------------
        # Build internal pipeline
        # ---------------------------------------------------------
        self.add(self._depay)
        self.add(self._dec)

        self._link_or_raise(
            self._depay,
            self._dec,
            "rtpvp9depay -> %s" % self._dec_factory,
        )

        # ---------------------------------------------------------
        # Expose ghost pads
        # ---------------------------------------------------------
        # sink → RTP input
        # src  → raw decoded frames
        depay_sink = self._depay.get_static_pad("sink")
        dec_src = self._dec.get_static_pad("src")

        if not depay_sink or not dec_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        # Ghost pads allow the bin to behave like a single element:
        #     RTP in → raw video out
        self._create_ghost_pads(depay_sink, dec_src)

    def _pick_decoder_factory(self) -> str:
        """
        Select best available VP9 decoder.

        Priority:
            1) NVIDIA hardware decoder (nvvp9dec)
            2) vp9dec (libvpx software decoder)
        """
        if HW_CODECS_ENABLED and Gst.ElementFactory.find("nvvp9dec") is not None:
            logger.info(
                "VP9DecoderBin: using hardware (HW) decoder: nvvp9dec",
            )
            return "nvvp9dec"
        if Gst.ElementFactory.find("vp9dec") is not None:
            logger.info(
                "VP9DecoderBin: using software (SW) decoder: vp9dec",
            )
            return "vp9dec"
        raise RuntimeError("No VP9 decoder available (tried: nvvp9dec, vp9dec)")
