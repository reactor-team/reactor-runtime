from typing import Optional

from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from .base import BaseEncoderBin


class VP8EncoderBin(BaseEncoderBin):
    """
    Encoder bin implementing:

        sink -> vp8enc -> rtpvp8pay(pt=...) -> src

    Designed for WebRTC RTP pipelines.

    VP8 is widely supported across browsers (Chrome, Firefox, Safari),
    making it a safe baseline codec for interoperability.
    """

    def __init__(
        self,
        pt: int,
        name: str = "vp8_encoder_bin",
        initial_bitrate_kbps: int = 6000,
        deadline: int = 1,
        cpu_used: int = 16,
        threads: int = 8,
        error_resilient: bool = True,
        ssrc: Optional[int] = None,
    ):
        """
        Args:
            pt:
                RTP payload type negotiated via SDP.

            initial_bitrate_kbps:
                Initial target bitrate in kbps.

            deadline:
                Controls encoder latency mode.
                1 = real-time mode (low latency, lower quality).
                Higher values improve quality but increase latency.

            cpu_used:
                Speed vs quality tradeoff (libvpx parameter).
                Higher value = faster encoding, lower quality.

            error_resilient:
                Enables resilience features to improve recovery
                from packet loss (important in WebRTC).
        """
        super().__init__(name=name)

        self._pt = int(pt)

        # ---------------------------------------------------------
        # Create VP8 encoder (libvpx based)
        # ---------------------------------------------------------
        self._enc = make_element("vp8enc", "vp8enc")

        # RTP payloader for VP8
        self._pay = make_element("rtpvp8pay", "rtpvp8pay")

        # ---------------------------------------------------------
        # Real-time tuning
        # ---------------------------------------------------------
        # Properties may vary depending on GStreamer build.
        # try_set_property prevents runtime failure if property is unavailable.

        # deadline=1 → real-time mode (disables lookahead)
        try_set_property(self._enc, "deadline", int(deadline))

        # cpu-used controls speed vs compression efficiency.
        # Higher values reduce CPU usage and latency.
        try_set_property(self._enc, "cpu-used", int(cpu_used))

        # Enable multi-threading
        try_set_property(self._enc, "threads", int(threads))

        # Enable error resilience tools (helps with packet loss).
        try_set_property(self._enc, "error-resilient", bool(error_resilient))

        self.set_target_bitrate_kbps(initial_bitrate_kbps)

        # RTP payload type must match SDP negotiation
        self._pay.set_property("pt", self._pt)
        if ssrc is not None:
            try_set_property(self._pay, "ssrc", ssrc)

        # ---------------------------------------------------------
        # Build internal pipeline
        # ---------------------------------------------------------
        self.add(self._enc)
        self.add(self._pay)

        if not self._enc.link(self._pay):
            raise RuntimeError("Failed to link vp8enc -> rtpvp8pay")

        # Fetch pads for ghost pad exposure
        enc_sink = self._enc.get_static_pad("sink")
        pay_src = self._pay.get_static_pad("src")

        if not enc_sink or not pay_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        # Expose ghost pads so bin behaves as:
        #   raw video in → RTP VP8 out
        self._create_ghost_pads(enc_sink, pay_src)

    def set_target_bitrate_kbps(self, bitrate_kbps: int) -> None:
        """Set ``vp8enc`` ``target-bitrate`` (internal unit: bps)."""
        if bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        self._enc.set_property("target-bitrate", int(bitrate_kbps) * 1000)

    def get_bitrate_kbps(self) -> int:
        """Return configured target bitrate in kbps."""
        return int(self._enc.get_property("target-bitrate")) // 1000
