"""Unit tests for the libwebrtc SDP transforms.

Pure string transforms with no native dependency: they exercise the module
directly, without the ``reactor_webrtc`` wheel.
"""

from reactor_runtime.transport.webrtc.libwebrtc.sdp import (
    deduplicate_bundle_pts,
    embed_ice_candidates,
)

_TWO_SECTION_ANSWER = (
    "v=0\r\n"
    "o=- 0 0 IN IP4 0.0.0.0\r\n"
    "s=-\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
    "a=mid:0\r\n"
    "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
    "a=mid:1\r\n"
)


def test_embed_ice_candidates_returns_input_when_none() -> None:
    assert embed_ice_candidates(_TWO_SECTION_ANSWER, []) == _TWO_SECTION_ANSWER


def test_embed_ice_candidates_places_each_at_its_m_section() -> None:
    embedded = embed_ice_candidates(
        _TWO_SECTION_ANSWER,
        [
            (0, "candidate:aud 1 udp 1 1.1.1.1 5000 typ host"),
            (1, "candidate:vid 1 udp 1 1.1.1.1 6000 typ host"),
        ],
    )
    lines = embedded.split("\r\n")
    audio_at = lines.index("a=candidate:aud 1 udp 1 1.1.1.1 5000 typ host")
    video_at = lines.index("a=candidate:vid 1 udp 1 1.1.1.1 6000 typ host")
    # The audio candidate sits inside section 0 (before the video m= line); the
    # video candidate sits inside section 1 (after it).
    video_mline = lines.index("m=video 9 UDP/TLS/RTP/SAVPF 96")
    assert audio_at < video_mline < video_at
    assert embedded.count("a=end-of-candidates") == 2


def test_embed_ice_candidates_skips_untagged_index() -> None:
    embedded = embed_ice_candidates(
        _TWO_SECTION_ANSWER, [(None, "candidate:x 1 udp 1 1.1.1.1 5000 typ host")]
    )
    assert "a=candidate" not in embedded
    assert "a=end-of-candidates" not in embedded


def test_deduplicate_bundle_pts_without_bundle_is_unchanged() -> None:
    sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:0\r\n"
    assert deduplicate_bundle_pts(sdp) == sdp


def test_deduplicate_bundle_pts_drops_conflicting_rtx_payload_type() -> None:
    sdp = (
        "v=0\r\n"
        "a=group:BUNDLE 0 1\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96 97\r\n"
        "a=mid:0\r\n"
        "a=rtpmap:96 VP8/90000\r\n"
        "a=rtpmap:97 rtx/90000\r\n"
        "a=fmtp:97 apt=96\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 98 97\r\n"
        "a=mid:1\r\n"
        "a=rtpmap:98 VP9/90000\r\n"
        "a=rtpmap:97 rtx/90000\r\n"
        "a=fmtp:97 apt=98\r\n"
    )
    out = deduplicate_bundle_pts(sdp)
    sections = out.split("m=video")
    first, second = sections[1], sections[2]
    # The first section keeps rtx PT 97 (apt=96, the winner).
    assert "a=rtpmap:97 rtx/90000" in first
    assert "a=fmtp:97 apt=96" in first
    # The second section's conflicting PT 97 and its scoped lines are removed.
    assert "97" not in second.splitlines()[0].split()
    assert "a=rtpmap:97" not in second
    assert "a=fmtp:97" not in second
    # The non-conflicting primary codec is untouched.
    assert "a=rtpmap:98 VP9/90000" in second


def test_deduplicate_bundle_pts_keeps_agreeing_rtx() -> None:
    sdp = (
        "v=0\r\n"
        "a=group:BUNDLE 0 1\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 96 97\r\n"
        "a=mid:0\r\n"
        "a=rtpmap:97 rtx/90000\r\n"
        "a=fmtp:97 apt=96\r\n"
        "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        "a=mid:1\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )
    assert deduplicate_bundle_pts(sdp) == sdp
