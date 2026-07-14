from typing import Dict, Optional

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.settings import HW_CODECS_ENABLED
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from .base import BaseEncoderBin


def h265_fmtp_to_x265_option_string(fmtp: Dict[str, Optional[str]]) -> str:
    """
    Map HEVC SDP fmtp (RFC 7798) keys to ``x265enc`` ``option-string`` tokens.

    SDP ``tier-flag`` is mapped to ``high-tier``.  ``profile-id`` is
    intentionally omitted: GStreamer's x265enc enforces profile through caps
    negotiation (calling x265_param_apply_profile internally), not via
    x265_param_parse — passing ``profile=...`` in option-string causes an
    encoder init failure.

    ``level-id`` is intentionally NOT mapped to ``level-idc``. The offered
    level-id advertises the receiver's decode ceiling, and forcing it onto
    the encoder makes x265 refuse to initialize for any frame above that
    level's budget (Safari offers level-id=93 = level 3.1, ~720p), silently
    stalling the sender branch. x265 derives a resolution-valid level and
    writes it into the bitstream; receivers decode based on that, and in
    practice tolerate streams above their advertised ceiling.

    Unrecognized or missing values are omitted so the encoder can use defaults.
    """
    parts: list[str] = []

    def _get(key: str) -> Optional[str]:
        v = fmtp.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    tier = _get("tier-flag")
    if tier == "1":
        parts.append("high-tier=1")
    elif tier == "0":
        parts.append("high-tier=0")

    return ":".join(parts)


class H265EncoderBin(BaseEncoderBin):
    """
    Encoder bin implementing:

        sink -> H265 encoder (NVENC or x265) ->
        rtph265pay(pt=...) -> src

    Designed for RTP/WebRTC pipelines.

    Notes:
        - H265 (HEVC) is not widely supported in browsers,
          but may be used in controlled environments.
        - No parser element is inserted here (unlike H264),
          assuming encoder produces RTP-compatible output.
        - Real-time tuning is critical to prevent latency buildup.
    """

    def __init__(
        self,
        pt: int,
        name: str = "h265_encoder_bin",
        initial_bitrate_kbps: int = 1500,
        ssrc: Optional[int] = None,
        format: Optional[Dict[str, Optional[str]]] = None,
    ):
        """
        Args:
            pt:
                RTP payload type negotiated in SDP.

            ssrc:
                SSRC of the stream.

            format:
                Parsed SDP fmtp parameters; for ``x265enc``, ``profile-id``,
                ``tier-flag``, and ``level-id`` are mapped into ``option-string``.

        """
        super().__init__(name=name)

        self._pt = int(pt)

        # ---------------------------------------------------------
        # Prefer hardware encoder if available (NVENC),
        # fallback to software x265.
        # ---------------------------------------------------------
        self._enc_factory = self._pick_encoder_factory()

        self._enc = make_element(self._enc_factory, self._enc_factory)

        # RTP payloader
        self._pay = make_element("rtph265pay", "rtph265pay")

        # ---------------------------------------------------------
        # Apply real-time tuning
        # ---------------------------------------------------------
        if self._enc_factory == "nvh265enc":
            self._apply_nvenc_realtime_tuning()

            self.set_target_bitrate_kbps(initial_bitrate_kbps)
        else:
            # Software x265 tuning
            self._apply_x265_realtime_tuning(
                format=format,
            )

            self.set_target_bitrate_kbps(initial_bitrate_kbps)

        # Set RTP payload type to match SDP negotiation
        self._pay.set_property("pt", self._pt)

        # config-interval=1 ensures VPS/SPS/PPS are periodically sent.
        # Important for:
        #   - Mid-stream decoder join
        #   - Recovery after packet loss
        try_set_property(self._pay, "config-interval", 1)
        if ssrc is not None:
            try_set_property(self._pay, "ssrc", ssrc)

        # ---------------------------------------------------------
        # Build internal pipeline
        # ---------------------------------------------------------
        self.add(self._enc)
        self.add(self._pay)

        if not self._enc.link(self._pay):
            raise RuntimeError("Failed to link %s -> rtph265pay" % self._enc_factory)

        enc_sink = self._enc.get_static_pad("sink")
        pay_src = self._pay.get_static_pad("src")

        if not enc_sink or not pay_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        # Expose ghost pads so bin behaves like a simple element:
        #   raw video in → RTP H265 out
        self._create_ghost_pads(enc_sink, pay_src)

    def set_target_bitrate_kbps(self, bitrate_kbps: int) -> None:
        """Set encoder ``bitrate`` (``nvh265enc`` / ``x265enc`` use kbps)."""
        if bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        self._enc.set_property("bitrate", int(bitrate_kbps))

    def get_bitrate_kbps(self) -> int:
        """Return configured bitrate in kbps."""
        return int(self._enc.get_property("bitrate"))

    def _pick_encoder_factory(self) -> str:
        """
        Select encoder implementation:

            - nvh265enc (GPU-accelerated, preferred for performance)
            - x265enc (software fallback)

        Hardware encoding significantly reduces CPU load,
        especially at higher resolutions.
        """
        if HW_CODECS_ENABLED and Gst.ElementFactory.find("nvh265enc") is not None:
            return "nvh265enc"
        return "x265enc"

    def _apply_x265_realtime_tuning(
        self,
        speed_preset: str = "ultrafast",
        tune: str = "zerolatency",
        format: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        """
        Configure x265 for low-latency operation.

        Important parameters:
            tune=zerolatency → disables lookahead
            bframes=0        → avoids latency increase
        """
        try_set_property(self._enc, "speed-preset", speed_preset)
        try_set_property(self._enc, "tune", tune)

        fmtp_opts = h265_fmtp_to_x265_option_string(format or {})
        option_chunks = [fmtp_opts, "bframes=0", "repeat-headers=1"]
        option_string = ":".join(c for c in option_chunks if c)

        try_set_property(self._enc, "option-string", option_string)

    def _apply_nvenc_realtime_tuning(self) -> None:
        """
        Configure NVIDIA NVENC for low-latency streaming.

        Key points:
            - CBR rate control
            - No B-frames
            - Frequent IDR frames
            - Repeated VPS/SPS/PPS insertion
        """
        try_set_property(self._enc, "zerolatency", True)
        try_set_property(self._enc, "rc-mode", "cbr")
        try_set_property(self._enc, "rate-control", "cbr")

        # Disable B-frames (adds latency)
        try_set_property(self._enc, "bframes", 0)
        try_set_property(self._enc, "max-bframes", 0)

        # Low-latency presets
        try_set_property(self._enc, "preset", "low-latency-hq")
        try_set_property(self._enc, "preset", "low-latency")
        try_set_property(self._enc, "tuning-info", "low-latency")

        # Ensure parameter sets are repeated in stream
        # Required for RTP interoperability
        try_set_property(self._enc, "repeat-sequence-header", True)
        try_set_property(self._enc, "insert-vps-sps-pps", True)
        try_set_property(self._enc, "insert-sps-pps", True)
