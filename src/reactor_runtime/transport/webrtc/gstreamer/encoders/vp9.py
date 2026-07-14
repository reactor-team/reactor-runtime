from typing import Optional

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    link_many,
    make_element,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer.settings import RTP_PAYLOAD_MTU
from .base import BaseEncoderBin


def vp9_profile_to_caps(profile: int) -> tuple[str, str]:
    """
    Map VP9 profile ID (from SDP fmtp profile-id) to:

        1) Raw video caps required BEFORE vp9enc
        2) Encoded caps required AFTER vp9enc

    Why this is important:
        VP9 profiles define:
            - Bit depth (8-bit vs 10/12-bit)
            - Chroma subsampling (4:2:0 vs 4:4:4)

        If raw format does not match profile,
        vp9enc may fail negotiation or produce incompatible output.

    Profiles:

        0: 8-bit 4:2:0      → I420
        1: 8-bit 4:4:4      → Y444
        2: 10/12-bit 4:2:0  → I420_10LE
        3: 10/12-bit 4:4:4  → Y444_10LE
    """

    # Encoded caps must signal profile so RTP side matches SDP
    enc_caps = f"video/x-vp9,profile=(string){profile}"

    if profile == 0:
        return "video/x-raw,format=(string)I420", enc_caps
    elif profile == 1:
        return "video/x-raw,format=(string)Y444", enc_caps
    elif profile == 2:
        return "video/x-raw,format=(string)I420_10LE", enc_caps
    elif profile == 3:
        return "video/x-raw,format=(string)Y444_10LE", enc_caps

    raise ValueError(f"VP9 invalid profile: {profile} (expected 0..3)")


class VP9EncoderBin(BaseEncoderBin):
    """
    Encoder bin implementing:

        sink -> capsfilter(raw format) ->
        vp9enc ->
        capsfilter(profile enforcement) ->
        rtpvp9pay(pt=...) -> src

    Designed for WebRTC RTP usage.

    Notes:
        - VP9 profile must match SDP fmtp profile-id.
        - Capsfilters enforce negotiation correctness.
        - Real-time tuning is critical for stable latency.
    """

    def __init__(
        self,
        pt: int,
        name: str = "vp9_encoder_bin",
        initial_bitrate_kbps: int = 800,
        deadline: int = 1,
        cpu_used: int = 16,
        threads: int = 8,
        row_mt: bool = True,
        profile_id: str = "0",
        ssrc: Optional[int] = None,
    ):
        """
        Args:
            pt:
                RTP payload type negotiated via SDP.

            deadline:
                Controls libvpx latency mode.
                1 = real-time (low latency, reduced compression).

            cpu_used:
                Speed vs quality tradeoff.
                Higher value = faster encoding, lower quality.

            row_mt:
                Enables row-based multi-threading (improves scalability).

            profile_id:
                VP9 profile negotiated via SDP (fmtp profile-id).
        """
        super().__init__(name=name)

        self._pt = int(pt)

        # ---------------------------------------------------------
        # Capsfilter BEFORE encoder:
        # Enforces raw input format compatible with profile.
        # ---------------------------------------------------------
        self._video_capsfilter = make_element("capsfilter", "video_capsfilter")

        # VP9 encoder (libvpx based)
        self._enc = make_element("vp9enc", "vp9enc")

        # Capsfilter AFTER encoder:
        # Enforces encoded profile matches SDP negotiation.
        self._enc_capsfilter = make_element("capsfilter", "enc_capsfilter")

        # RTP payloader
        self._pay = make_element("rtpvp9pay", "rtpvp9pay")

        # ---------------------------------------------------------
        # Profile enforcement
        # ---------------------------------------------------------
        video_caps, enc_caps = vp9_profile_to_caps(int(profile_id))

        # Apply raw caps
        try_set_property(
            self._video_capsfilter, "caps", Gst.Caps.from_string(video_caps)
        )

        # Apply encoded caps
        try_set_property(self._enc_capsfilter, "caps", Gst.Caps.from_string(enc_caps))

        # ---------------------------------------------------------
        # Real-time tuning (libvpx best-effort)
        # ---------------------------------------------------------

        # Enable row-based multi-threading
        try_set_property(self._enc, "row-mt", bool(row_mt))

        # deadline=1 → real-time mode (no lookahead)
        try_set_property(self._enc, "deadline", int(deadline))

        # Speed vs compression tradeoff
        try_set_property(self._enc, "cpu-used", int(cpu_used))

        # Thread parallelism
        try_set_property(self._enc, "threads", int(threads))

        # Force constant bitrate (CBR) mode
        try_set_property(self._enc, "end-usage", "cbr")

        # Drop frames if encoder falls behind
        try_set_property(self._enc, "dropframe-threshold", 50)

        # Disable lookahead for low latency
        try_set_property(self._enc, "lag-in-frames", 0)

        # Reduce encoder buffering
        try_set_property(self._enc, "buffer-size", 200)
        try_set_property(self._enc, "buffer-initial-size", 200)
        try_set_property(self._enc, "buffer-optimal-size", 200)

        self.set_target_bitrate_kbps(initial_bitrate_kbps)

        # RTP payload type must match SDP negotiation
        self._pay.set_property("pt", self._pt)
        try_set_property(self._pay, "mtu", RTP_PAYLOAD_MTU)
        if ssrc is not None:
            try_set_property(self._pay, "ssrc", ssrc)

        # ---------------------------------------------------------
        # Build internal bin pipeline
        # ---------------------------------------------------------
        self.add(self._video_capsfilter)
        self.add(self._enc)
        self.add(self._enc_capsfilter)
        self.add(self._pay)

        link_many(
            self._video_capsfilter,
            self._enc,
            self._enc_capsfilter,
            self._pay,
        )

        enc_sink = self._video_capsfilter.get_static_pad("sink")
        pay_src = self._pay.get_static_pad("src")

        if not enc_sink or not pay_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        # Expose ghost pads so bin behaves as:
        #   raw video in → RTP VP9 out
        self._create_ghost_pads(enc_sink, pay_src)

    def set_target_bitrate_kbps(self, bitrate_kbps: int) -> None:
        """Set ``vp9enc`` ``target-bitrate`` (internal unit: bps)."""
        if bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        self._enc.set_property("target-bitrate", int(bitrate_kbps) * 1000)

    def get_bitrate_kbps(self) -> int:
        """Return configured target bitrate in kbps."""
        return int(self._enc.get_property("target-bitrate")) // 1000
