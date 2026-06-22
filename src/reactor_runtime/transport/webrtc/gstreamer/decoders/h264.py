from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.settings import HW_CODECS_ENABLED
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger
from .base import BaseRTPDecoderBin

logger = get_logger(__name__)


class H264DecoderBin(BaseRTPDecoderBin):
    """
    RTP decoder bin implementing:

        sink (RTP H264) ->
        rtph264depay ->
        decoder (HW preferred, SW fallback) ->
        src (raw video)

    Low-latency focus:
        - Prefer NVIDIA hardware decoder when available
        - Otherwise fall back to software decoder

    H264 is the most widely supported WebRTC codec
    (especially required for Safari interoperability).
    """

    def __init__(self, name: str = "h264_decoder_bin"):
        super().__init__(name=name)

        # ---------------------------------------------------------
        # RTP depayloader
        # ---------------------------------------------------------
        # Converts RTP H264 packets into H264 elementary stream.
        # Handles STAP-A / FU-A packetization modes internally.
        self._depay = make_element("rtph264depay", "rtph264depay")
        try_set_property(self._depay, "wait-for-keyframe", True)
        try_set_property(self._depay, "request-keyframe", True)

        # ---------------------------------------------------------
        # Select decoder implementation
        # ---------------------------------------------------------
        # Hardware acceleration reduces CPU usage significantly.
        self._dec_factory = self._pick_decoder_factory()

        self._dec = make_element(self._dec_factory, self._dec_factory)

        # ---------------------------------------------------------
        # Best-effort low-latency tuning
        # ---------------------------------------------------------
        # Decoder properties vary by implementation.
        # Try common threading properties if available.

        try_set_property(self._dec, "max-threads", 4)
        try_set_property(self._dec, "threads", 4)

        # ---------------------------------------------------------
        # Build internal pipeline
        # ---------------------------------------------------------
        self.add(self._depay)
        self.add(self._dec)

        # Ensure depayloader output links into decoder
        self._link_or_raise(
            self._depay,
            self._dec,
            "rtph264depay -> %s" % self._dec_factory,
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

        # Ghost pads make this bin behave like:
        #     RTP in → raw video out
        self._create_ghost_pads(depay_sink, dec_src)

    def _pick_decoder_factory(self) -> str:
        """
        Select best available H264 decoder.

        Priority:

            1) NVIDIA hardware decoders (GPU accelerated)
            2) avdec_h264 (libav software decoder)

        Hardware decoder names vary depending on:
            - Distribution
            - Plugin set
            - Driver version
        """

        # ---------------------------------------------------------
        # NVIDIA hardware decoder candidates
        # ---------------------------------------------------------
        if HW_CODECS_ENABLED:
            for f in ("nvh264dec", "nvdec_h264", "nvv4l2decoder"):
                if Gst.ElementFactory.find(f) is not None:
                    logger.info(
                        "H264DecoderBin: using hardware (HW) decoder", decoder=f
                    )
                    return f

        # ---------------------------------------------------------
        # Software fallback (libav)
        # ---------------------------------------------------------
        if Gst.ElementFactory.find("avdec_h264") is not None:
            logger.info(
                "H264DecoderBin: using software (SW) decoder: avdec_h264",
            )
            return "avdec_h264"

        raise RuntimeError(
            "No H264 decoder available (tried: nvh264dec, nvdec_h264, nvv4l2decoder, avdec_h264)"
        )
