
"""Tests for H265 encoder fmtp → x265 option-string mapping."""

from reactor_runtime.transport.webrtc.gstreamer.encoders.h265 import (
    h265_fmtp_to_x265_option_string,
)


class TestH265FmtpToX265OptionString:
    def test_empty(self) -> None:
        assert h265_fmtp_to_x265_option_string({}) == ""

    def test_tier_and_level(self) -> None:
        # profile-id is ignored (set via GStreamer caps, not option-string)
        s = h265_fmtp_to_x265_option_string(
            {"profile-id": "1", "tier-flag": "0", "level-id": "93"}
        )
        assert s == "high-tier=0:level-idc=3.1"

    def test_high_tier_level_4(self) -> None:
        s = h265_fmtp_to_x265_option_string(
            {"profile-id": "2", "tier-flag": "1", "level-id": "120"}
        )
        assert s == "high-tier=1:level-idc=4"

    def test_profile_id_only_produces_empty(self) -> None:
        # profile-id alone generates no option-string tokens
        assert h265_fmtp_to_x265_option_string({"profile-id": "3"}) == ""

    def test_level_only(self) -> None:
        s = h265_fmtp_to_x265_option_string({"level-id": "90"})
        assert s == "level-idc=3"

    def test_level_6(self) -> None:
        s = h265_fmtp_to_x265_option_string({"level-id": "180"})
        assert s == "level-idc=6"

    def test_invalid_level_skipped(self) -> None:
        assert h265_fmtp_to_x265_option_string({"level-id": "999"}) == ""
