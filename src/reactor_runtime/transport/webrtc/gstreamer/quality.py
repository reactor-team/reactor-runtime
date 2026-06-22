
"""
Video quality scoring utilities.

Provides a formula-based estimate of the *expected* bitrate for a given
resolution, frame-rate, and codec, and a 0–10 quality score comparing
a measured bitrate against that expectation.
"""

from typing import Optional

# =========================================================================
# Reference tables
# =========================================================================

# Bits-per-pixel for H.264 at "medium" quality (general content, no motion
# extremes).  Other codecs scale from this baseline via _CODEC_EFFICIENCY.
_BPP_H264_MEDIUM = 0.08

# Compression efficiency relative to H.264 (lower → fewer bits needed for
# equal perceived quality).  Keys are RTP encoding-name strings (uppercase),
# as returned by GStreamer and SDP negotiation.
_CODEC_EFFICIENCY: dict[str, float] = {
    "H264": 1.00,
    "H265": 0.55,
    "VP8": 1.00,
    "VP9": 0.65,
    "AV1": 0.45,
}

# =========================================================================
# Per-track helpers
# =========================================================================


def expected_video_bitrate_bps(
    width: int, height: int, fps: float, codec_name: str
) -> int:
    """Return the target bitrate in bps for a good-quality encode."""
    efficiency = _CODEC_EFFICIENCY.get(codec_name.upper(), 1.0)
    return int(width * height * fps * _BPP_H264_MEDIUM * efficiency)


def video_qos_score(
    current_bps: int, width: int, height: int, fps: float, codec_name: str
) -> float:
    """Return a 0–10 quality score for a single video track.

    10.0 means at or above the expected bitrate for good quality.
    Score scales linearly down to 0 as bitrate approaches zero.
    """
    expected = expected_video_bitrate_bps(width, height, fps, codec_name)
    if expected <= 0:
        return 10.0
    return round(min(10.0, current_bps / expected * 10.0), 1)


# =========================================================================
# Aggregate score across multiple senders
# =========================================================================


def aggregate_qos_score(scores: list[float]) -> Optional[float]:
    """Return the average 0–10 quality score across all video senders.

    Returns None when the list is empty (no ready senders).
    """
    if not scores:
        return None
    return round(sum(scores) / len(scores), 1)
