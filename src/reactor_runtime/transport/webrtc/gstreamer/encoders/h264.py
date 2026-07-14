from typing import Optional

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.settings import (
    HW_CODECS_ENABLED,
    RTP_PAYLOAD_MTU,
)
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    link_many,
    make_element,
    try_set_property,
)
from .base import BaseEncoderBin


def h264_plid_to_gst_profile_level(profile_level_id: str) -> tuple[str, str]:
    """
    Convert SDP profile-level-id (e.g. "42e01f") into:

        (profile_string, level_string)

    for use in GStreamer caps.

    Why this is needed:
        - SDP negotiates H264 profile-level-id in hex format.
        - GStreamer caps require profile + level as strings.
        - Mismatch between negotiated profile and encoder output
          can cause decoder failures (especially Safari).

    Example mappings:
        "42e01f" → ("constrained-baseline", "3.1")
        "4d0032" → ("main", "5.0")
        "640029" → ("high", "4.1")
    """

    if len(profile_level_id) != 6:
        raise ValueError(f"Invalid profile-level-id: {profile_level_id}")

    # profile-level-id is 3 bytes encoded in hex:
    #   byte 0 → profile_idc
    #   byte 1 → constraint flags
    #   byte 2 → level_idc
    try:
        data = bytes.fromhex(profile_level_id)
    except ValueError as e:
        raise ValueError(f"Invalid hex in profile-level-id '{profile_level_id}': {e}")

    profile_idc = data[0]
    compat = data[1]
    level_idc = data[2]

    # ---------- Profile detection ----------
    # See RFC 6184 (H264 RTP payload format)

    if profile_idc == 0x42:  # 66 → Baseline family
        # Constrained Baseline is indicated by constraint_set1_flag (bit 6)
        if compat & 0x40:
            profile = "constrained-baseline"
        else:
            profile = "baseline"

    elif profile_idc == 0x4D:  # 77 → Main profile
        profile = "main"

    elif profile_idc == 0x64:  # 100 → High profile
        profile = "high"

    else:
        # Unknown profile
        raise ValueError(f"Unsupported H264 profile_idc: 0x{profile_idc:02x}")

    # ---------- Level ----------
    # level_idc is expressed as:
    #   30 → Level 3.0
    #   31 → Level 3.1
    #   40 → Level 4.0
    major = level_idc // 10
    minor = level_idc % 10
    level = f"{major}.{minor}"

    return profile, level


class H264EncoderBin(BaseEncoderBin):
    """
    Encoder bin implementing:

        sink -> H264 encoder (NVENC or x264) ->
        h264parse -> capsfilter(profile/level) ->
        rtph264pay(pt=...) -> src

    Designed for WebRTC usage with:
        - Strict profile-level negotiation
        - Low-latency configuration
        - Periodic SPS/PPS insertion (critical for WebRTC)
    """

    def __init__(
        self,
        pt: int,
        name: str = "h264_encoder_bin",
        initial_bitrate_kbps: int = 6000,
        key_int_max: int = 60,
        threads: int = 4,
        x264_speed_preset: str = "ultrafast",
        profile_level_id: str = "42e01f",
        ssrc: Optional[int] = None,
    ):
        """
        Args:
            pt:
                RTP payload type negotiated in SDP.

            key_int_max:
                Maximum GOP size (keyframe interval).
                In WebRTC, frequent keyframes improve recovery
                after packet loss or PLI requests.

            profile_level_id:
                Hex string negotiated in SDP.
                Must match encoder output caps to avoid
                decoder incompatibility (Safari sensitive).
        """
        super().__init__(name=name)

        self._pt = int(pt)

        # ---------------------------------------------------------
        # Select hardware encoder if available (NVENC),
        # otherwise fallback to software x264.
        # ---------------------------------------------------------
        self._enc_factory = self._pick_encoder_factory()

        self._enc = make_element(self._enc_factory, self._enc_factory)

        # h264parse ensures proper alignment and SPS/PPS handling.
        self._parse = make_element("h264parse", "h264parse")

        # capsfilter enforces negotiated profile and level.
        # Critical to ensure output matches SDP.
        self._capsfilter = make_element("capsfilter", "capsfilter")

        # RTP payloader
        self._pay = make_element("rtph264pay", "rtph264pay")

        # ---------------------------------------------------------
        # Apply real-time tuning
        # ---------------------------------------------------------
        if self._enc_factory == "nvh264enc":
            self._apply_nvenc_realtime_tuning(key_int_max=key_int_max)
            self.set_target_bitrate_kbps(initial_bitrate_kbps)
        else:
            self._apply_x264_realtime_tuning(
                key_int_max=key_int_max,
                threads=threads,
                speed_preset=x264_speed_preset,
            )
            self.set_target_bitrate_kbps(initial_bitrate_kbps)

        # Set RTP payload type
        self._pay.set_property("pt", self._pt)
        try_set_property(self._pay, "mtu", RTP_PAYLOAD_MTU)

        # config-interval=1 forces SPS/PPS to be sent periodically.
        # Important for WebRTC because:
        #   - Decoders may join mid-stream
        #   - Recovery after packet loss requires SPS/PPS
        try_set_property(self._pay, "config-interval", 1)
        if ssrc is not None:
            try_set_property(self._pay, "ssrc", ssrc)

        # ---------------------------------------------------------
        # Enforce negotiated profile only — NOT level.
        # ---------------------------------------------------------
        # The SDP profile-level-id encodes a level (e.g. 42e01f -> 3.1) that is
        # only valid up to ~720p. Pinning that level in the capsfilter caps the
        # encoder below the actual resolution: at 2K the encoded stream is
        # ~level 5.x, the level-3.1 capsfilter fails to negotiate, and the whole
        # send branch stalls with `not-negotiated` — zero RTP ever leaves. VP8
        # (no level) is unaffected, which is why it works over the same path.
        #
        # Pin the profile only. x264/NVENC pick a level valid for the resolution,
        # and receivers (incl. Safari) match on the profile, not the exact level.
        # (`h264_plid_to_gst_profile_level` still validates the negotiated id.)
        profile, _level = h264_plid_to_gst_profile_level(profile_level_id)

        caps = Gst.Caps.from_string(f"video/x-h264,profile=(string){profile}")

        try_set_property(self._capsfilter, "caps", caps)

        # ---------------------------------------------------------
        # Build pipeline inside bin
        # ---------------------------------------------------------
        self.add(self._enc)
        self.add(self._parse)
        self.add(self._capsfilter)
        self.add(self._pay)

        link_many(self._enc, self._parse, self._capsfilter, self._pay)

        enc_sink = self._enc.get_static_pad("sink")
        pay_src = self._pay.get_static_pad("src")

        if not enc_sink or not pay_src:
            raise RuntimeError("Failed to fetch sink/src pads")

        self._create_ghost_pads(enc_sink, pay_src)

    def set_target_bitrate_kbps(self, bitrate_kbps: int) -> None:
        """Set encoder ``bitrate`` (``nvh264enc`` / ``x264enc`` use kbps)."""
        if bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        self._enc.set_property("bitrate", int(bitrate_kbps))

    def get_bitrate_kbps(self) -> int:
        """Return configured bitrate in kbps."""
        return int(self._enc.get_property("bitrate"))

    def _pick_encoder_factory(self) -> str:
        """
        Prefer NVENC if available (GPU acceleration),
        otherwise fallback to x264 software encoder.
        """
        if HW_CODECS_ENABLED and Gst.ElementFactory.find("nvh264enc") is not None:
            return "nvh264enc"
        return "x264enc"

    def _apply_x264_realtime_tuning(
        self, key_int_max: int, threads: int, speed_preset: str
    ) -> None:
        """
        Configure x264 for low-latency WebRTC usage.

        Key settings:
            tune=zerolatency → disables lookahead
            key-int-max      → limits GOP size
            sliced-threads   → improves parallelism
            byte-stream=False → produce AVCC format (required for RTP)
        """
        try_set_property(self._enc, "tune", "zerolatency")
        try_set_property(self._enc, "speed-preset", speed_preset)
        try_set_property(self._enc, "key-int-max", int(key_int_max))
        try_set_property(self._enc, "threads", int(threads))
        try_set_property(self._enc, "sliced-threads", True)
        try_set_property(self._enc, "byte-stream", False)
        try_set_property(self._enc, "aud", False)

    def _apply_nvenc_realtime_tuning(self, key_int_max: int) -> None:
        """
        Configure NVIDIA NVENC for low-latency CBR streaming.

        Important for WebRTC:
            - No B-frames (adds latency)
            - CBR rate control
            - Frequent IDR frames
            - Repeated SPS/PPS
        """
        try_set_property(self._enc, "zerolatency", True)
        try_set_property(self._enc, "rc-mode", "cbr")
        try_set_property(self._enc, "rate-control", "cbr")

        try_set_property(self._enc, "gop-size", int(key_int_max))
        try_set_property(self._enc, "iframeinterval", int(key_int_max))

        try_set_property(self._enc, "bframes", 0)
        try_set_property(self._enc, "max-bframes", 0)

        try_set_property(self._enc, "preset", "low-latency-hq")
        try_set_property(self._enc, "preset", "low-latency")
        try_set_property(self._enc, "tuning-info", "low-latency")

        try_set_property(self._enc, "repeat-sequence-header", True)
        try_set_property(self._enc, "insert-sps-pps", True)
