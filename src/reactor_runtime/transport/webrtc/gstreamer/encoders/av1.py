from typing import Optional

from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    make_element,
    try_set_property,
)
from .base import BaseEncoderBin


class AV1EncoderBin(BaseEncoderBin):
    """
    GStreamer bin that encapsulates:

        sink -> AV1 encoder -> av1parse -> rtpav1pay(pt=...) -> src

    Designed for WebRTC usage (e.g., with webrtcbin), where:
        - Encoded stream must be RTP payloaded
        - PT (payload type) must match SDP negotiation
        - Low latency is required (real-time constraints)

    The bin exposes:
        - sink ghost pad  → raw video input
        - src ghost pad   → RTP AV1 output
    """

    def __init__(
        self,
        pt: int,
        name: str = "av1_encoder_bin",
        initial_bitrate_kbps: int = 1500,
        threads: int = 8,
        ssrc: Optional[int] = None,
    ):
        """
        Args:
            pt:
                RTP payload type negotiated in SDP.
                Must match the PT used in rtpav1pay and advertised in answer.

            initial_bitrate_kbps:
                Initial encoder bitrate target (kbps).
                Can later be updated dynamically.

            threads:
                Performance tuning parameters for encoder.
                (Currently partially applied depending on encoder backend.)
        """
        super().__init__(name=name)

        self._pt = int(pt)

        # ---------------------------------------------------------
        # Pick encoder implementation (AOM vs SVT)
        # ---------------------------------------------------------
        self._enc_factory = self._pick_encoder_factory()
        self._enc = make_element(self._enc_factory, self._enc_factory)

        # Parser ensures proper bitstream formatting for RTP payloader.
        # Especially important for AV1 where bitstream framing matters.
        self._parser = make_element("av1parse", "av1parse")

        # RTP payloader converts AV1 bitstream into RTP packets.
        self._pay = make_element("rtpav1pay", "rtpav1pay")

        # ---------------------------------------------------------
        # Apply real-time encoder tuning
        # ---------------------------------------------------------
        self._apply_realtime_tuning(threads=threads)

        self.set_target_bitrate_kbps(initial_bitrate_kbps)

        # Set RTP payload type to match negotiated PT
        self._pay.set_property("pt", self._pt)
        if ssrc is not None:
            try_set_property(self._pay, "ssrc", ssrc)

        # ---------------------------------------------------------
        # Build bin pipeline
        # ---------------------------------------------------------
        self.add(self._enc)
        self.add(self._parser)
        self.add(self._pay)

        if not self._enc.link(self._parser):
            raise RuntimeError("Failed to link %s -> av1parse" % self._enc_factory)

        if not self._parser.link(self._pay):
            raise RuntimeError("Failed to link av1parse -> rtpav1pay")

        # Fetch pads to expose via ghost pads
        enc_sink = self._enc.get_static_pad("sink")
        pay_src = self._pay.get_static_pad("src")
        if not enc_sink or not pay_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        # Create ghost pads so this bin behaves like a simple element:
        #   raw video in → RTP AV1 out
        self._create_ghost_pads(enc_sink, pay_src)

    def _pick_encoder_factory(self) -> str:
        """
        Choose which AV1 encoder implementation to use.

        Currently:
            - Prefers "av1enc" (AOM-based)
            - Avoids "svtav1enc" due to limited exposed tuning knobs

        Note:
            SVT-AV1 is often faster, but if critical properties
            (e.g. usage-profile, latency knobs) are not exposed,
            real-time WebRTC behavior may degrade.
        """
        # In the future, could dynamically detect:
        # if Gst.ElementFactory.find("svtav1enc") is not None:
        #     return "svtav1enc"
        return "av1enc"

    def _apply_realtime_tuning(self, threads: int) -> None:
        """
        Apply low-latency tuning parameters to AV1 encoder.

        These settings aim to:
            - Minimize encoding delay
            - Reduce buffering
            - Favor speed over compression efficiency
            - Drop frames instead of increasing latency

        Important for WebRTC:
            Latency stability is more important than bitrate efficiency.
        """
        if self._enc_factory == "av1enc":
            # Enable row-based multi-threading
            try_set_property(self._enc, "row-mt", True)

            # Use tiling to improve parallelization
            try_set_property(self._enc, "tile-columns", 4)
            try_set_property(self._enc, "tile-rows", 4)

            # Disable lookahead to reduce latency
            try_set_property(self._enc, "lag-in-frames", 0)

            # Control thread count
            try_set_property(self._enc, "threads", threads)

            # Real-time usage profile (critical for WebRTC)
            # Typically:
            #   0 = good quality
            #   1 = real-time
            try_set_property(self._enc, "usage-profile", 1)

            # Reduce encoder buffer size to lower latency
            try_set_property(self._enc, "buf-sz", 200)

            # Drop frames if encoder falls behind
            # Prevents pipeline backpressure explosion
            try_set_property(self._enc, "drop-frame", True)

            # Constrain quantizer range for predictable behavior
            # Lower quantizer = better quality but slower
            try_set_property(self._enc, "min-quantizer", 8)
            try_set_property(self._enc, "max-quantizer", 16)

    def set_target_bitrate_kbps(self, bitrate_kbps: int) -> None:
        """Set ``target-bitrate`` on the AV1 encoder (GStreamer uses kbps)."""
        if bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        self._enc.set_property("target-bitrate", int(bitrate_kbps))

    def get_bitrate_kbps(self) -> int:
        """Return configured ``target-bitrate`` (kbps)."""
        return int(self._enc.get_property("target-bitrate"))
