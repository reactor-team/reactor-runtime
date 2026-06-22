from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.settings import HW_CODECS_ENABLED
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger
from .base import BaseRTPDecoderBin

logger = get_logger(__name__)


class H265DecoderBin(BaseRTPDecoderBin):
    """
    RTP decoder bin implementing:

        sink (RTP H265 / HEVC) ->
        rtph265depay ->
        decoder (HW preferred, SW fallback) ->
        src (raw video)

    H265/HEVC is less commonly supported in browsers,
    but useful in controlled environments or native clients.

    Hardware acceleration is strongly preferred due to
    higher computational cost compared to H264/VP8.
    """

    def __init__(self, name: str = "h265_decoder_bin"):
        super().__init__(name=name)

        # ---------------------------------------------------------
        # RTP depayloader
        # ---------------------------------------------------------
        # Converts RTP H265 packets into H265 elementary bitstream.
        # Handles aggregation units (AP) and fragmentation units (FU).
        self._depay = make_element("rtph265depay", "rtph265depay")
        try_set_property(self._depay, "wait-for-keyframe", True)
        try_set_property(self._depay, "request-keyframe", True)

        self._h265parse = make_element("h265parse", "h265parse")
        self._h265parse.set_property("config-interval", -1)

        # ---------------------------------------------------------
        # Select decoder implementation
        # ---------------------------------------------------------
        self._dec_factory = self._pick_decoder_factory()
        self._dec = make_element(self._dec_factory, self._dec_factory)

        # ---------------------------------------------------------
        # Build internal pipeline
        #   rtph265depay → h265parse → decoder
        # ---------------------------------------------------------
        self.add(self._depay)
        self.add(self._h265parse)
        self.add(self._dec)

        self._link_or_raise(self._depay, self._h265parse, "rtph265depay -> h265parse")
        self._link_or_raise(
            self._h265parse, self._dec, "h265parse -> %s" % self._dec_factory
        )

        # ---------------------------------------------------------
        # Expose ghost pads
        # ---------------------------------------------------------
        depay_sink = self._depay.get_static_pad("sink")
        dec_src = self._dec.get_static_pad("src")

        if not depay_sink or not dec_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        # Ghost pads allow the bin to behave like:
        #     RTP in → raw video out
        self._create_ghost_pads(depay_sink, dec_src)

    def _pick_decoder_factory(self) -> str:
        """
        Select best available H265 decoder implementation.

        Priority:

            1) NVIDIA hardware decoders
            2) avdec_h265 (libav software decoder)
            3) libde265dec (fallback software decoder)

        Hardware decoder names vary depending on:
            - Distro
            - Plugin set
            - Driver version
        """

        # ---------------------------------------------------------
        # NVIDIA hardware decoders
        # ---------------------------------------------------------
        if HW_CODECS_ENABLED:
            for f in ("nvh265dec", "nvdec_h265", "nvv4l2decoder"):
                if Gst.ElementFactory.find(f) is not None:
                    logger.info(
                        "H265DecoderBin: using hardware (HW) decoder", decoder=f
                    )
                    return f

        # ---------------------------------------------------------
        # Software fallback (libav)
        # ---------------------------------------------------------
        if Gst.ElementFactory.find("avdec_h265") is not None:
            logger.info(
                "H265DecoderBin: using software (SW) decoder: avdec_h265",
            )
            return "avdec_h265"

        raise RuntimeError(
            "No H265 decoder available (tried: nvh265dec, nvdec_h265, "
            "nvv4l2decoder, avdec_h265)"
        )
