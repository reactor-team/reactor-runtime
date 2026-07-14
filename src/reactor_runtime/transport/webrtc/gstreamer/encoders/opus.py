
"""
Opus audio encoder bin: sink -> opusenc -> rtpopuspay -> src.

Designed for WebRTC RTP pipelines. Opus is the standard audio codec
for WebRTC (Chrome, Firefox, Safari).

SDP fmtp parameters (RFC 7587) are honored when passed via format:
  - minptime: minimum RTP packet time in ms; encoder frame size is chosen >= minptime.
  - useinbandfec=1: enable in-band FEC in opusenc.
"""

from typing import Dict, Optional

from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer.settings import RTP_PAYLOAD_MTU
from .base import BaseEncoderBin

# opusenc frame-size enum values (ms); must be >= SDP minptime
_OPUS_FRAME_SIZES_MS = (2.5, 5, 10, 20, 40, 60)


def _frame_size_from_minptime(minptime_ms: Optional[int]) -> int:
    """Choose smallest opusenc frame size >= minptime, or default 20 ms.

    Returns the ``GstOpusEncFrameSize`` enum value as an int: the 2.5 ms
    frame maps to enum ``2`` and every other size to its millisecond value,
    so ``int(size)`` yields the correct enum member for all of them
    (notably ``int(2.5) == 2``).
    """
    if minptime_ms is None or minptime_ms <= 0:
        return 20
    for size in _OPUS_FRAME_SIZES_MS:
        if size >= minptime_ms:
            return int(size)
    return 60


def _parse_useinbandfec(format_params: Dict[str, Optional[str]]) -> bool:
    """True if useinbandfec=1 in SDP fmtp."""
    val = format_params.get("useinbandfec")
    if val is None:
        return True  # default for WebRTC
    return str(val).strip() == "1"


class OpusEncoderBin(BaseEncoderBin):
    """
    Encoder bin implementing:

        sink -> opusenc -> rtpopuspay(pt=...) -> src

    Accepts raw audio (e.g. from audioconvert/audiorate); produces RTP Opus.
    """

    def __init__(
        self,
        pt: int,
        name: str = "opus_encoder_bin",
        initial_bitrate_kbps: int = 64,
        bitrate_type: str = "constrained-vbr",
        frame_size_ms: Optional[float] = None,
        inband_fec: Optional[bool] = None,
        dtx: bool = False,
        ssrc: Optional[int] = None,
        format: Optional[Dict[str, Optional[str]]] = None,
    ):
        """
        Args:
            pt:
                RTP payload type negotiated via SDP.

            initial_bitrate_kbps:
                Target bitrate in kbps (e.g. 64).

            bitrate_type:
                "cbr", "vbr", or "constrained-vbr" (default for WebRTC).

            frame_size_ms:
                Frame duration in ms (2.5, 5, 10, 20, 40, 60). If None, derived
                from format minptime (SDP fmtp) or default 20.

            inband_fec:
                Enable in-band FEC. If None, derived from format useinbandfec or True.

            dtx:
                Discontinuous transmission (silence suppression).

            ssrc:
                Optional RTP SSRC for the payloader.

            format:
                SDP fmtp parameters (e.g. minptime=10;useinbandfec=1). Used to set
                frame_size >= minptime and useinbandfec.
        """
        super().__init__(name=name)

        self._pt = int(pt)
        format_params = format or {}

        # SDP minptime: minimum RTP packet time in ms; we use frame size >= minptime
        minptime_val = format_params.get("minptime")
        minptime_ms: Optional[int] = None
        if minptime_val is not None and str(minptime_val).strip().isdigit():
            minptime_ms = int(minptime_val)

        if frame_size_ms is not None:
            frame_size = int(float(frame_size_ms))
        else:
            frame_size = _frame_size_from_minptime(minptime_ms)

        if inband_fec is not None:
            fec = inband_fec
        else:
            fec = _parse_useinbandfec(format_params)

        self._enc = make_element("opusenc", "opusenc")
        self._pay = make_element("rtpopuspay", "rtpopuspay")

        self.set_target_bitrate_kbps(initial_bitrate_kbps)

        try_set_property(self._enc, "bitrate-type", bitrate_type)
        # frame-size is the GstOpusEncFrameSize enum. It must be passed as the
        # int enum value: GStreamer < 1.28 refuses to coerce a float and raises,
        # which would otherwise abort the entire encoder (and WebRTC) setup.
        try_set_property(self._enc, "frame-size", frame_size)
        try_set_property(self._enc, "inband-fec", fec)
        try_set_property(self._enc, "dtx", dtx)

        self._pay.set_property("pt", self._pt)
        try_set_property(self._pay, "mtu", RTP_PAYLOAD_MTU)
        if ssrc is not None:
            try_set_property(self._pay, "ssrc", ssrc)

        self.add(self._enc)
        self.add(self._pay)

        if not self._enc.link(self._pay):
            raise RuntimeError("Failed to link opusenc -> rtpopuspay")

        enc_sink = self._enc.get_static_pad("sink")
        pay_src = self._pay.get_static_pad("src")

        if not enc_sink or not pay_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        self._create_ghost_pads(enc_sink, pay_src)

    def set_target_bitrate_kbps(self, bitrate_kbps: int) -> None:
        """Set ``opusenc`` ``bitrate`` (property unit: bps)."""
        if bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        self._enc.set_property("bitrate", int(bitrate_kbps) * 1000)

    def get_bitrate_kbps(self) -> int:
        """Return configured bitrate in kbps."""
        return int(self._enc.get_property("bitrate")) // 1000
