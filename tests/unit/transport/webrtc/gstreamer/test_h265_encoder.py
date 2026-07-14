
"""Tests for H265 encoder fmtp → x265 option-string mapping.

The offered ``level-id`` is deliberately not mapped into the option-string:
it advertises the receiver's decode ceiling, and forcing it onto x265 makes
the encoder refuse to initialize for frames above that level's budget
(Safari offers level-id=93 = level 3.1, ~720p), silently stalling the
sender branch. Only ``tier-flag`` is mapped; x265 derives the level from
the actual resolution/framerate.
"""

from reactor_runtime.transport.webrtc.gstreamer.encoders.h265 import (
    h265_fmtp_to_x265_option_string,
)


class TestH265FmtpToX265OptionString:
    def test_empty(self) -> None:
        assert h265_fmtp_to_x265_option_string({}) == ""

    def test_tier_mapped_level_ignored(self) -> None:
        # profile-id is ignored (set via GStreamer caps, not option-string);
        # level-id is ignored (receiver decode ceiling, not an encoder bound).
        s = h265_fmtp_to_x265_option_string(
            {"profile-id": "1", "tier-flag": "0", "level-id": "93"}
        )
        assert s == "high-tier=0"

    def test_high_tier(self) -> None:
        s = h265_fmtp_to_x265_option_string(
            {"profile-id": "2", "tier-flag": "1", "level-id": "120"}
        )
        assert s == "high-tier=1"

    def test_profile_id_only_produces_empty(self) -> None:
        assert h265_fmtp_to_x265_option_string({"profile-id": "3"}) == ""

    def test_level_only_produces_empty(self) -> None:
        assert h265_fmtp_to_x265_option_string({"level-id": "90"}) == ""

    def test_level_180_produces_empty(self) -> None:
        assert h265_fmtp_to_x265_option_string({"level-id": "180"}) == ""
