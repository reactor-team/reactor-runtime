from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.settings import HW_CODECS_ENABLED
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger
from .base import BaseRTPDecoderBin

logger = get_logger(__name__)


class AV1DecoderBin(BaseRTPDecoderBin):
    """
    RTP decoder bin implementing:

        sink (RTP AV1) ->
        rtpav1depay ->
        av1parse ->
        decoder (HW or SW) ->
        src (raw video)

    Decoder selection priority:
        1) Hardware decoders (if available)
        2) dav1ddec (fast, optimized software decoder)
        3) avdec_av1 (libav fallback)

    AV1 decoding can be CPU-intensive; hardware acceleration is strongly preferred.
    """

    def __init__(self, name: str = "av1_decoder_bin"):
        super().__init__(name=name)

        # ---------------------------------------------------------
        # RTP depayloader
        # ---------------------------------------------------------
        # Converts RTP AV1 packets into AV1 elementary stream.
        # Handles RTP aggregation and packet boundaries.
        self._depay = make_element("rtpav1depay", "rtpav1depay")
        try_set_property(self._depay, "wait-for-keyframe", True)
        try_set_property(self._depay, "request-keyframe", True)

        # ---------------------------------------------------------
        # AV1 parser
        # ---------------------------------------------------------
        # Ensures proper frame boundary alignment and metadata handling
        # before feeding frames to decoder.
        #
        # Particularly important for:
        #   - Hardware decoders
        #   - Dynamic resolution changes
        self._parse = make_element("av1parse", "av1parse")

        # ---------------------------------------------------------
        # Select decoder implementation
        # ---------------------------------------------------------
        # Hardware acceleration is preferred when available.
        self._dec_factory = self._pick_decoder_factory()

        self._dec = make_element(self._dec_factory, self._dec_factory)

        # ---------------------------------------------------------
        # Best-effort low-latency tuning
        # ---------------------------------------------------------

        # Drop corrupted frames instead of stalling decoder.
        # Important for WebRTC environments with packet loss.
        try_set_property(self._dec, "discard-corrupted-frames", True)

        # Reduce frame reordering / buffering delay (if supported).
        try_set_property(self._dec, "max-frame-delay", 1)

        # Control in-loop filtering (may trade quality for speed).
        try_set_property(self._dec, "inloop-filters", 4)

        # n-threads=0 usually means auto-detect optimal thread count.
        try_set_property(self._dec, "n-threads", 0)

        # ---------------------------------------------------------
        # Build internal pipeline
        # ---------------------------------------------------------
        self.add(self._depay)
        self.add(self._parse)
        self.add(self._dec)

        self._link_or_raise(self._depay, self._parse, "rtpav1depay -> av1parse")
        self._link_or_raise(
            self._parse,
            self._dec,
            "av1parse -> %s" % self._dec_factory,
        )

        # ---------------------------------------------------------
        # Expose ghost pads
        # ---------------------------------------------------------
        # sink → RTP input
        # src  → raw decoded frames
        sink_pad = self._depay.get_static_pad("sink")
        src_pad = self._dec.get_static_pad("src")

        if not sink_pad or not src_pad:
            raise RuntimeError("Failed to fetch sink/src pads")

        # Ghost pads allow the bin to behave like:
        #     RTP in → raw video out
        self._create_ghost_pads(sink_pad, src_pad)

    def _pick_decoder_factory(self) -> str:
        """
        Select best available AV1 decoder implementation.

        Order of preference:

            1) Hardware decoders (GPU acceleration)
            2) dav1ddec (fast multi-threaded software decoder)
            3) avdec_av1 (libav fallback, slower)

        Hardware decoder names vary by platform and plugin build.
        """

        # ---------------------------------------------------------
        # Hardware decoder candidates (platform-dependent)
        # ---------------------------------------------------------
        hw_candidates = [
            # NVIDIA (desktop / datacenter)
            "nvav1dec",
            "nvdec_av1",
            # NVIDIA Jetson / embedded
            "nvv4l2decoder",
            # VA-API (Intel/AMD on Linux)
            "vaav1dec",
            # Windows D3D11
            "d3d11av1dec",
        ]

        if HW_CODECS_ENABLED:
            for f in hw_candidates:
                if Gst.ElementFactory.find(f) is not None:
                    logger.info("AV1DecoderBin: using hardware (HW) decoder", decoder=f)
                    return f

        # ---------------------------------------------------------
        # Preferred software decoder
        # ---------------------------------------------------------
        # dav1d is highly optimized and generally faster than libav.
        if Gst.ElementFactory.find("dav1ddec") is not None:
            logger.info(
                "AV1DecoderBin: using software (SW) decoder: dav1ddec",
            )
            return "dav1ddec"

        # ---------------------------------------------------------
        # Final fallback
        # ---------------------------------------------------------
        if Gst.ElementFactory.find("avdec_av1") is not None:
            logger.info(
                "AV1DecoderBin: using software (SW) decoder: avdec_av1",
            )
            return "avdec_av1"

        raise RuntimeError("No AV1 decoder available")
