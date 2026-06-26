
import pytest

from reactor_runtime.transport.webrtc.gstreamer.quality import (
    _CODEC_EFFICIENCY,
    _BPP_H264_MEDIUM,
    aggregate_qos_score,
    expected_video_bitrate_bps,
    video_qos_score,
)


class TestExpectedVideoBitrateBps:
    def test_h264_1080p30(self):
        assert expected_video_bitrate_bps(1920, 1080, 30, "H264") == int(
            1920 * 1080 * 30 * _BPP_H264_MEDIUM * _CODEC_EFFICIENCY["H264"]
        )

    def test_h265_is_more_efficient_than_h264(self):
        h264 = expected_video_bitrate_bps(1920, 1080, 30, "H264")
        h265 = expected_video_bitrate_bps(1920, 1080, 30, "H265")
        assert h265 < h264

    def test_vp8_equals_h264(self):
        assert expected_video_bitrate_bps(1280, 720, 30, "VP8") == expected_video_bitrate_bps(
            1280, 720, 30, "H264"
        )

    def test_all_codecs_below_h264_except_vp8(self):
        base = expected_video_bitrate_bps(1920, 1080, 30, "H264")
        assert expected_video_bitrate_bps(1920, 1080, 30, "H265") < base
        assert expected_video_bitrate_bps(1920, 1080, 30, "VP9") < base
        assert expected_video_bitrate_bps(1920, 1080, 30, "AV1") < base

    def test_av1_is_most_efficient(self):
        codecs = ["H264", "VP8", "VP9", "H265", "AV1"]
        bitrates = {c: expected_video_bitrate_bps(1920, 1080, 30, c) for c in codecs}
        assert bitrates["AV1"] == min(bitrates.values())

    def test_scales_linearly_with_fps(self):
        at_30 = expected_video_bitrate_bps(1280, 720, 30, "H264")
        at_60 = expected_video_bitrate_bps(1280, 720, 60, "H264")
        assert at_60 == at_30 * 2

    def test_scales_linearly_with_resolution(self):
        hd = expected_video_bitrate_bps(1280, 720, 30, "H264")
        fhd = expected_video_bitrate_bps(1920, 1080, 30, "H264")
        ratio = (1920 * 1080) / (1280 * 720)
        assert fhd == pytest.approx(hd * ratio, rel=1e-6)

    def test_case_insensitive_codec(self):
        assert expected_video_bitrate_bps(1920, 1080, 30, "h264") == expected_video_bitrate_bps(
            1920, 1080, 30, "H264"
        )
        assert expected_video_bitrate_bps(1920, 1080, 30, "vp9") == expected_video_bitrate_bps(
            1920, 1080, 30, "VP9"
        )

    def test_unknown_codec_falls_back_to_h264_efficiency(self):
        unknown = expected_video_bitrate_bps(1920, 1080, 30, "UNKNOWN")
        h264 = expected_video_bitrate_bps(1920, 1080, 30, "H264")
        assert unknown == h264

    def test_returns_int(self):
        result = expected_video_bitrate_bps(1920, 1080, 30, "H264")
        assert isinstance(result, int)


class TestVideoQualityScore:
    def test_at_expected_bitrate_returns_10(self):
        bps = expected_video_bitrate_bps(1920, 1080, 30, "H264")
        assert video_qos_score(bps, 1920, 1080, 30, "H264") == 10.0

    def test_above_expected_capped_at_10(self):
        bps = expected_video_bitrate_bps(1920, 1080, 30, "H264") * 2
        assert video_qos_score(bps, 1920, 1080, 30, "H264") == 10.0

    def test_half_expected_bitrate_returns_5(self):
        bps = expected_video_bitrate_bps(1920, 1080, 30, "H264") // 2
        assert video_qos_score(bps, 1920, 1080, 30, "H264") == 5.0

    def test_zero_bitrate_returns_0(self):
        assert video_qos_score(0, 1920, 1080, 30, "H264") == 0.0

    def test_score_increases_with_bitrate(self):
        base = expected_video_bitrate_bps(1920, 1080, 30, "H264")
        scores = [
            video_qos_score(base * p // 100, 1920, 1080, 30, "H264")
            for p in [0, 25, 50, 75, 100]
        ]
        assert scores == sorted(scores)

    def test_score_range_is_0_to_10(self):
        for bps in [0, 1_000_000, 5_000_000, 100_000_000]:
            score = video_qos_score(bps, 1920, 1080, 30, "H264")
            assert 0.0 <= score <= 10.0

    def test_returns_float_rounded_to_one_decimal(self):
        # 1/3 of expected → 3.3 (not 3.333...)
        bps = expected_video_bitrate_bps(1920, 1080, 30, "H264") // 3
        score = video_qos_score(bps, 1920, 1080, 30, "H264")
        assert score == round(score, 1)

    def test_codec_efficiency_reflected_in_score(self):
        # Same bitrate: H265 scores higher because its expected target is lower.
        bps = 3_000_000
        h264_score = video_qos_score(bps, 1920, 1080, 30, "H264")
        h265_score = video_qos_score(bps, 1920, 1080, 30, "H265")
        assert h265_score > h264_score

    def test_case_insensitive_codec(self):
        bps = 3_000_000
        assert video_qos_score(bps, 1920, 1080, 30, "h264") == video_qos_score(
            bps, 1920, 1080, 30, "H264"
        )

    def test_unknown_codec_uses_h264_efficiency(self):
        bps = 3_000_000
        assert video_qos_score(bps, 1920, 1080, 30, "UNKNOWN") == video_qos_score(
            bps, 1920, 1080, 30, "H264"
        )


class TestAggregateQualityScore:
    def test_empty_list_returns_none(self):
        assert aggregate_qos_score([]) is None

    def test_single_score_returned_as_is(self):
        assert aggregate_qos_score([8.0]) == 8.0

    def test_perfect_scores_return_10(self):
        assert aggregate_qos_score([10.0, 10.0, 10.0]) == 10.0

    def test_zero_scores_return_0(self):
        assert aggregate_qos_score([0.0, 0.0]) == 0.0

    def test_average_of_two_scores(self):
        assert aggregate_qos_score([10.0, 5.0]) == 7.5

    def test_average_of_three_scores(self):
        assert aggregate_qos_score([9.0, 6.0, 3.0]) == 6.0

    def test_result_rounded_to_one_decimal(self):
        # 10 + 5 + 2 = 17 / 3 = 5.666... → 5.7
        score = aggregate_qos_score([10.0, 5.0, 2.0])
        assert score == round(score, 1)
        assert score == 5.7

    def test_mixed_low_and_high_scores(self):
        score = aggregate_qos_score([0.0, 10.0])
        assert score == 5.0
