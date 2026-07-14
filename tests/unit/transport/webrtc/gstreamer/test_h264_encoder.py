
"""Tests for the H264 encoder bin's profile/level caps behavior.

The level inside an offered profile-level-id (e.g. 3.1 in Chrome's default
``42e01f``) is the offerer's decode ceiling, not a bound on what the answerer
may send: browsers offer ``level-asymmetry-allowed=1`` (RFC 6184), which
permits answering with a different level. Pinning the offered level into the
encoder caps makes x264enc reject any frame above the level's macroblock
budget (level 3.1 tops out around 720p30), silently stalling the sender
branch — the "zero RTP at 2K" bug. These tests lock in the fixed caps shape:
profile is enforced, level is not. The end-to-end 2K data-flow regression
lives in ``test_h264_highres.py``.
"""

import pytest

from reactor_runtime.transport.webrtc.gstreamer.gst import Gst
from reactor_runtime.transport.webrtc.gstreamer.encoders.h264 import (
    H264EncoderBin,
    h264_plid_to_gst_profile_level,
)


class TestH264PlidToGstProfileLevel:
    def test_constrained_baseline_31(self) -> None:
        assert h264_plid_to_gst_profile_level("42e01f") == (
            "constrained-baseline",
            "3.1",
        )

    def test_main_50(self) -> None:
        assert h264_plid_to_gst_profile_level("4d0032") == ("main", "5.0")

    def test_high_41(self) -> None:
        assert h264_plid_to_gst_profile_level("640029") == ("high", "4.1")

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError):
            h264_plid_to_gst_profile_level("42e0")

    def test_invalid_hex_raises(self) -> None:
        with pytest.raises(ValueError):
            h264_plid_to_gst_profile_level("zze01f")


def _capsfilter_structure(profile_level_id: str) -> Gst.Structure:
    enc_bin = H264EncoderBin(pt=109, profile_level_id=profile_level_id)
    capsfilter = enc_bin.get_by_name("capsfilter")
    assert capsfilter is not None
    caps = capsfilter.get_property("caps")
    assert caps is not None
    assert caps.get_size() == 1
    return caps.get_structure(0)


class TestH264EncoderBinCaps:
    def test_profile_is_enforced(self) -> None:
        struct = _capsfilter_structure("42e01f")
        assert struct.get_name() == "video/x-h264"
        assert struct.get_string("profile") == "constrained-baseline"

    def test_level_is_not_pinned(self) -> None:
        # Pinning the offered level caps output at that level's frame-size
        # budget and breaks every resolution above it.
        struct = _capsfilter_structure("42e01f")
        assert not struct.has_field("level")

    def test_profile_is_enforced_for_high(self) -> None:
        struct = _capsfilter_structure("640029")
        assert struct.get_string("profile") == "high"
        assert not struct.has_field("level")
