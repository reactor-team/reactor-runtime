
from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import make_element
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger
from .base import BaseRTPDecoderBin

logger = get_logger(__name__)


class OpusDecoderBin(BaseRTPDecoderBin):
    """
    RTP decoder bin implementing:

        sink (RTP Opus) ->
        rtpopusdepay ->
        opusdec ->
        src (raw audio)

    Designed for use with:
        - webrtcbin src pads
        - rtpbin receive chains

    Pairs with :class:`OpusEncoderBin` on the send side.
    """

    def __init__(self, name: str = "opus_decoder_bin"):
        super().__init__(name=name)

        if Gst.ElementFactory.find("rtpopusdepay") is None:
            raise RuntimeError("rtpopusdepay element is not available")
        if Gst.ElementFactory.find("opusdec") is None:
            raise RuntimeError("opusdec element is not available")

        self._depay = make_element("rtpopusdepay", "rtpopusdepay")
        self._dec = make_element("opusdec", "opusdec")

        self.add(self._depay)
        self.add(self._dec)

        self._link_or_raise(
            self._depay,
            self._dec,
            "rtpopusdepay -> opusdec",
        )

        depay_sink = self._depay.get_static_pad("sink")
        dec_src = self._dec.get_static_pad("src")

        if not depay_sink or not dec_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        logger.info("OpusDecoderBin: using software decoder: opusdec")
        self._create_ghost_pads(depay_sink, dec_src)
