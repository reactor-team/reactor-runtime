"""Unit tests for the libwebrtc SDP transforms.

Pure string transforms with no native dependency: they exercise the module
directly, without the ``reactor_webrtc`` wheel.
"""

import pytest

from reactor_runtime.transport.webrtc.sdp import (
    bump_session_version,
    deduplicate_bundle_pts,
    embed_ice_candidates,
    set_media_direction,
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


# ── set_media_direction ──────────────────────────────────────────────────────

_TWO_SECTIONS = (
    "v=0\r\n"
    "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
    "a=mid:0\r\n"
    "a=sendonly\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
    "a=mid:1\r\n"
    "a=sendonly\r\n"
)


def test_set_media_direction_changes_only_the_named_section() -> None:
    out = set_media_direction(_TWO_SECTIONS, "1", "inactive")
    assert out.split("\r\n") == [
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 96",
        "a=mid:0",
        "a=sendonly",  # untouched
        "m=audio 9 UDP/TLS/RTP/SAVPF 111",
        "a=mid:1",
        "a=inactive",
        "",
    ]


def test_set_media_direction_round_trips() -> None:
    paused = set_media_direction(_TWO_SECTIONS, "1", "inactive")
    assert set_media_direction(paused, "1", "sendonly") == _TWO_SECTIONS


def test_set_media_direction_adds_one_to_a_section_that_has_none() -> None:
    """A section with no direction line is sendrecv, so the flip needs one."""
    sdp = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:1\r\na=rtcp-mux\r\n"
    out = set_media_direction(sdp, "1", "inactive")
    assert (
        out == "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:1\r\na=inactive\r\na=rtcp-mux\r\n"
    )


def test_set_media_direction_leaves_an_unknown_mid_alone() -> None:
    assert set_media_direction(_TWO_SECTIONS, "9", "inactive") == _TWO_SECTIONS


def test_set_media_direction_rejects_a_value_that_is_not_a_direction() -> None:
    with pytest.raises(ValueError, match="not an SDP direction"):
        set_media_direction(_TWO_SECTIONS, "1", "sendonly-ish")


# ── bump_session_version ─────────────────────────────────────────────────────

_WITH_ORIGIN = "v=0\r\no=- 7503906533368660784 2 IN IP4 127.0.0.1\r\ns=-\r\n"


def test_bump_session_version_advances_only_the_version_field() -> None:
    out = bump_session_version(_WITH_ORIGIN, 1)
    assert out.split("\r\n")[1] == "o=- 7503906533368660784 3 IN IP4 127.0.0.1"


def test_bump_session_version_counts_from_the_negotiated_one() -> None:
    """Each pass starts from the base, so it carries how far past it is."""
    assert "7503906533368660784 5 " in bump_session_version(_WITH_ORIGIN, 3)


def test_bump_session_version_leaves_a_description_with_no_origin_alone() -> None:
    assert bump_session_version("v=0\r\ns=-\r\n", 1) == "v=0\r\ns=-\r\n"


def test_bump_session_version_leaves_an_unparsable_origin_alone() -> None:
    sdp = "v=0\r\no=- session two IN IP4 127.0.0.1\r\n"
    assert bump_session_version(sdp, 1) == sdp
