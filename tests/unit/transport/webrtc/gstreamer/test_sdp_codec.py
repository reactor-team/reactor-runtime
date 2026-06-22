"""
tests/test_sdp_codec.py

Covered Scenarios (Public API Only)
===================================

Purpose
-------
Validate SDP video codec selection and fmtp extraction behavior using ONLY
the public functions:

    - get_video_codec_from_sdp(sdp: str) -> CodecEntry
        (reads list from transports/gstreamer/settings.SUPPORTED_VIDEO_CODECS;
         returns selected CodecEntry with codec, payload_type, parameters)

    - get_codec_from_sdp_by_mid(sdp: str, mid: str) -> CodecEntry
        (finds media section by a=mid; works for video or audio; uses
         SUPPORTED_VIDEO_CODECS or SUPPORTED_AUDIO_CODECS per section type)

Private helpers such as:
    - _normalize_encoding_name
    - parse_fmtp_params

are NOT tested directly. They are exercised indirectly through the observable
behavior of get_video_codec_from_sdp().

---------------------------------------------------------------------

Scenarios Covered
-----------------

1) Input validation and malformed SDP handling
   - Empty / non-string SDP -> ValueError
   - SDP without any m=video line -> ValueError
   - Malformed m=video line (no PTs) -> ValueError
   - m=video with PTs but no matching a=rtpmap for prioritized codecs -> ValueError
     and error message includes PT diagnostics (e.g., "96:<no-rtpmap>")

2) Codec selection rules (observable behavior)
   - Only the FIRST m=video section is considered (ignores subsequent m=video)
   - Payload type (PT) order from m=video is used as browser preference order
   - Priority list is applied first; PT order breaks ties among available codecs

3) Priority list normalization (indirectly tests _normalize_encoding_name)
   - Accepts codec names with punctuation/aliases:
       * "H.264" and "H-264" should match "H264" signaled via rtpmap
   - Case-insensitivity is handled (e.g., "vp8" vs "VP8")

4) RTPMAP parsing robustness
   - Supports rtpmap with optional channels (e.g., /90000/2) by regex
   - Ignores non-matching rtpmap lines safely

5) FMTP parsing behavior (indirectly tests parse_fmtp_params)
   - Parses key=value pairs split by ';' with optional whitespace
   - Parses flag-only params (no '=') as key -> None
   - Missing fmtp yields empty dict {}
   - Multiple fmtp lines for the same PT are merged (later keys override)

6) Same codec at multiple PTs with different parameters
   - When entry requires specific parameters (e.g. profile-level-id), the loop
     skips PTs whose fmtp does not match and continues to the next PT; the first
     PT that matches is selected (exercises parameter-filtering loop).

---------------------------------------------------------------------

Notes
-----
- These tests validate behavior at the public API boundary.
- They are intentionally resilient to newline style differences.
- SDPs are minimal and focused on m=video, a=rtpmap, and a=fmtp semantics.
"""

import pytest

from reactor_runtime.transport.webrtc.gstreamer.errors import WebRTCNoVideoError
from reactor_runtime.transport.webrtc.gstreamer.sdp.codec import (
    NoSupportedCodecsError,
    get_codec_from_sdp_by_mid,
    get_rtx_payload_type_by_mid,
    get_video_codec_from_sdp,
    normalize_sdp_for_supported_codecs,
)


def _p(codec: str, parameters=None):
    """Build a priority entry: {"codec": codec, "parameters": {...}}."""
    return {"codec": codec, "parameters": parameters or {}}

# ---------------------------------------------------------------------
# SDP fixtures (minimal but representative)
# ---------------------------------------------------------------------

SDP_NO_VIDEO = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=rtpmap:111 opus/48000/2
"""

SDP_VIDEO_NO_PTS = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF
a=rtpmap:96 VP8/90000
"""

SDP_VIDEO_PTS_BUT_NO_RTPMAP = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96 102
a=fmtp:102 profile-level-id=42e01f;packetization-mode=1
"""

SDP_VIDEO_MULTI_CODECS = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96 102 98 104
a=rtpmap:96 VP8/90000
a=rtpmap:102 H264/90000
a=fmtp:102 level-asymmetry-allowed=1; packetization-mode=1; profile-level-id=42e01f
a=rtpmap:98 VP9/90000
a=rtpmap:104 AV1/90000
a=fmtp:104 level-idx=5;profile=0;tier=0
"""

# Two m=video lines: only first should be considered
SDP_TWO_VIDEO_SECTIONS = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 VP8/90000
m=video 9 UDP/TLS/RTP/SAVPF 104
a=rtpmap:104 AV1/90000
"""

# rtpmap with channels (should still parse codec)
SDP_RTPMAP_WITH_CHANNELS = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 102
a=rtpmap:102 H264/90000/2
"""

# FMTP flags and multiple fmtp lines for same PT (merge behavior)
SDP_FMTP_FLAGS_AND_MERGE = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 102
a=rtpmap:102 H264/90000
a=fmtp:102 usedtx; stereo=1
a=fmtp:102 packetization-mode=1; stereo=0
"""

# Same codec (H264) at multiple PTs with different profile-level-id; exercises
# parameter-filtering loop (skip non-matching PT, continue to next).
SDP_SAME_CODEC_MULTIPLE_PTS_DIFFERENT_FMTP = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 98 102
a=rtpmap:98 H264/90000
a=rtpmap:102 H264/90000
a=fmtp:98 profile-level-id=4d002a;packetization-mode=1
a=fmtp:102 profile-level-id=42e01f;packetization-mode=1
"""

# SDP with a=mid for get_codec_from_sdp_by_mid tests: video mid=0, audio mid=1
SDP_WITH_MIDS_VIDEO_AUDIO = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96 102
a=mid:0
a=rtpmap:96 VP8/90000
a=rtpmap:102 H264/90000
a=fmtp:102 profile-level-id=42e01f;packetization-mode=1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:1
a=rtpmap:111 opus/48000/2
"""

# Same but with named mids
SDP_WITH_NAMED_MIDS = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:video
a=rtpmap:96 VP8/90000
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:audio
a=rtpmap:111 opus/48000/2
"""

# SDP with mid but no video/audio section for that mid (application section)
SDP_WITH_APPLICATION_MID = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=application 9 UDP/DTLS/SCTP webrtc-datachannel
a=mid:2
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:0
a=rtpmap:96 VP8/90000
"""


# ---------------------------------------------------------------------
# Tests: input validation / errors
# ---------------------------------------------------------------------

@pytest.mark.parametrize("bad_sdp", [None, "", 123, [], {}])
def test_get_video_codec_from_sdp_rejects_empty_or_non_string(bad_sdp):
    with pytest.raises(ValueError):
        get_video_codec_from_sdp(bad_sdp)  # type: ignore[arg-type]


def test_get_video_codec_from_sdp_raises_when_no_m_video():
    with pytest.raises(WebRTCNoVideoError, match="No m=video"):
        get_video_codec_from_sdp(SDP_NO_VIDEO)


def test_get_video_codec_from_sdp_raises_on_malformed_m_video_line():
    # m=video exists but no payload types
    with pytest.raises(ValueError, match="Malformed m=video|No payload types"):
        get_video_codec_from_sdp(SDP_VIDEO_NO_PTS)


def test_get_video_codec_from_sdp_raises_when_no_supported_codec_found_and_includes_diagnostics():
    # PTs exist but no rtpmap: should fail and include "<no-rtpmap>" diagnostic
    with pytest.raises(ValueError) as e:
        get_video_codec_from_sdp(SDP_VIDEO_PTS_BUT_NO_RTPMAP)

    msg = str(e.value)
    assert "None of the supported codecs" in msg
    assert "96:<no-rtpmap>" in msg
    assert "102:<no-rtpmap>" in msg


# ---------------------------------------------------------------------
# Tests: selection rules (priority + PT order)
# ---------------------------------------------------------------------

def test_selects_using_default_priority_and_pt_order_as_tiebreaker(monkeypatch):
    # Use VP8 first so we select VP8 when SDP offers multiple codecs
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("VP8"), _p("H264"), _p("VP9"), _p("AV1")],
    )
    selected = get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)

    assert selected["codec"] == "VP8"
    assert selected["payload_type"] == 96
    assert selected["parameters"] == {}  # no fmtp for VP8 in fixture


def test_custom_priority_overrides_default_and_uses_pt_order_within_codec_choice(
    monkeypatch,
):
    # Patch to prefer AV1 first -> should pick AV1 104
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("AV1"), _p("H264")],
    )
    selected = get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)
    assert selected["codec"] == "AV1"
    assert selected["payload_type"] == 104
    assert selected["parameters"].get("profile") == "0"
    assert selected["parameters"].get("tier") == "0"


def test_pt_order_breaks_ties_when_priority_contains_multiple_available_codecs(
    monkeypatch,
):
    # Patch to [H264, VP8] -> should pick H264 first
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264"), _p("VP8")],
    )
    selected = get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102

    # Now invert: VP8 first
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("VP8"), _p("H264")],
    )
    selected = get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)
    assert selected["codec"] == "VP8"
    assert selected["payload_type"] == 96


def test_only_first_video_section_is_considered(monkeypatch):
    # First m=video is VP8 only; second is AV1 only. Patch to prefer AV1 first.
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("AV1"), _p("VP8")],
    )
    selected = get_video_codec_from_sdp(SDP_TWO_VIDEO_SECTIONS)
    # Even though we prefer AV1, first m=video doesn't offer it.
    assert selected["codec"] == "VP8"
    assert selected["payload_type"] == 96


# ---------------------------------------------------------------------
# Tests: normalization behavior (indirect)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "priority_entry",
    ["H264", "H.264", "h264", "H-264", "H_264"],
)
def test_priority_name_normalization_matches_rtpmap_codec(priority_entry, monkeypatch):
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p(priority_entry)],
    )
    selected = get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102


def test_rtpmap_with_channels_is_parsed(monkeypatch):
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264")],
    )
    selected = get_video_codec_from_sdp(SDP_RTPMAP_WITH_CHANNELS)
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102


# ---------------------------------------------------------------------
# Tests: fmtp parsing/merging behavior (indirect)
# ---------------------------------------------------------------------

def test_fmtp_parsing_splits_kv_pairs_and_strips_spaces(monkeypatch):
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264")],
    )
    selected = get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102
    fmtp = selected["parameters"]
    assert fmtp["level-asymmetry-allowed"] == "1"
    assert fmtp["packetization-mode"] == "1"
    assert fmtp["profile-level-id"] == "42e01f"


def test_fmtp_flag_params_are_mapped_to_none_and_multiple_lines_merge_last_wins(
    monkeypatch,
):
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264")],
    )
    selected = get_video_codec_from_sdp(SDP_FMTP_FLAGS_AND_MERGE)
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102
    fmtp = selected["parameters"]
    # parameters is Dict[str, str]: only keys with a value are included; flag-style (None) omitted
    assert "usedtx" not in fmtp
    # First line sets stereo=1, second line sets stereo=0 -> last wins
    assert fmtp["stereo"] == "0"
    assert fmtp["packetization-mode"] == "1"


def test_priority_with_parametrization_merges_fmtp_priority_wins(monkeypatch):
    # SDP has H264 with profile-level-id=42e01f, packetization-mode=1.
    # Entry parameters must match SDP to select; then returned params are merged (entry wins).
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264", parameters={"profile-level-id": "42e01f", "packetization-mode": "1"})],
    )
    selected = get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102
    fmtp = selected["parameters"]
    assert fmtp["profile-level-id"] == "42e01f"
    assert fmtp["packetization-mode"] == "1"
    assert fmtp.get("level-asymmetry-allowed") == "1"


def test_payload_type_from_offer_in_return(monkeypatch):
    """Returned payload_type is the PT from the SDP offer where the codec was found."""
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264")],
    )
    selected = get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102  # H264 is at PT 102 in SDP_VIDEO_MULTI_CODECS


def test_codec_present_but_parameters_not_in_sdp_raises_no_codec_selected(monkeypatch):
    # SDP offers H264 with profile-level-id=42e01f (from SDP_VIDEO_MULTI_CODECS).
    # We only support H264 with profile-level-id=4d002a (different value).
    # We must NOT select H264; no codec matches -> raise.
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264", parameters={"profile-level-id": "4d002a"})],
    )
    with pytest.raises(ValueError) as e:
        get_video_codec_from_sdp(SDP_VIDEO_MULTI_CODECS)
    assert "None of the supported codecs were found" in str(e.value)


def test_same_codec_multiple_pts_skips_non_matching_pt_selects_matching(monkeypatch):
    # SDP offers H264 at PT 98 (profile-level-id=4d002a) and PT 102 (profile-level-id=42e01f).
    # We support H264 with profile-level-id=42e01f only. The loop must skip PT 98 and
    # select PT 102 (first PT in order that matches).
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264", parameters={"profile-level-id": "42e01f", "packetization-mode": "1"})],
    )
    selected = get_video_codec_from_sdp(SDP_SAME_CODEC_MULTIPLE_PTS_DIFFERENT_FMTP)
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102
    assert selected["parameters"]["profile-level-id"] == "42e01f"
    assert selected["parameters"]["packetization-mode"] == "1"


SDP_VIDEO_VP9_WITH_RTX = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 98 99
a=mid:0
a=rtpmap:98 VP9/90000
a=rtpmap:99 rtx/90000
a=fmtp:99 apt=98
"""


def test_get_rtx_payload_type_by_mid_returns_rtx_pt_matching_apt():
    assert get_rtx_payload_type_by_mid(SDP_VIDEO_VP9_WITH_RTX, "0", 98) == 99


def test_get_rtx_payload_type_by_mid_returns_none_when_no_rtx_apt():
    assert get_rtx_payload_type_by_mid(SDP_VIDEO_VP9_WITH_RTX, "0", 96) is None


# ---------------------------------------------------------------------
# get_codec_from_sdp_by_mid
# ---------------------------------------------------------------------

@pytest.mark.parametrize("bad_sdp", [None, "", 123])
def test_get_codec_from_sdp_by_mid_rejects_invalid_sdp(bad_sdp):
    with pytest.raises(ValueError, match="sdp must be"):
        get_codec_from_sdp_by_mid(bad_sdp, "0")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_mid", [None, "", 123])
def test_get_codec_from_sdp_by_mid_rejects_invalid_mid(bad_mid):
    with pytest.raises(ValueError, match="mid must be"):
        get_codec_from_sdp_by_mid(SDP_WITH_MIDS_VIDEO_AUDIO, bad_mid)  # type: ignore[arg-type]


def test_get_codec_from_sdp_by_mid_raises_when_no_section_has_mid():
    with pytest.raises(ValueError, match="No video or audio section with a=mid:nonexistent"):
        get_codec_from_sdp_by_mid(SDP_WITH_MIDS_VIDEO_AUDIO, "nonexistent")


def test_get_codec_from_sdp_by_mid_raises_when_mid_is_application_only():
    # mid 2 is application, not video/audio
    with pytest.raises(ValueError, match="No video or audio section with a=mid:2"):
        get_codec_from_sdp_by_mid(SDP_WITH_APPLICATION_MID, "2")


def test_get_codec_from_sdp_by_mid_returns_video_codec_for_video_mid(monkeypatch):
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264"), _p("VP8")],
    )
    selected = get_codec_from_sdp_by_mid(SDP_WITH_MIDS_VIDEO_AUDIO, "0")
    assert selected["codec"] == "H264"
    assert selected["payload_type"] == 102
    assert selected["parameters"].get("profile-level-id") == "42e01f"


def test_get_codec_from_sdp_by_mid_returns_audio_codec_for_audio_mid(monkeypatch):
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_AUDIO_CODECS",
        [_p("Opus")],
    )
    selected = get_codec_from_sdp_by_mid(SDP_WITH_MIDS_VIDEO_AUDIO, "1")
    assert selected["codec"] == "OPUS"
    assert selected["payload_type"] == 111


def test_get_codec_from_sdp_by_mid_works_with_named_mids(monkeypatch):
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("VP8")],
    )
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_AUDIO_CODECS",
        [_p("Opus")],
    )
    video_entry = get_codec_from_sdp_by_mid(SDP_WITH_NAMED_MIDS, "video")
    assert video_entry["codec"] == "VP8"
    assert video_entry["payload_type"] == 96

    audio_entry = get_codec_from_sdp_by_mid(SDP_WITH_NAMED_MIDS, "audio")
    assert audio_entry["codec"] == "OPUS"
    assert audio_entry["payload_type"] == 111


def test_get_codec_from_sdp_by_mid_video_section_same_logic_as_get_video_codec(monkeypatch):
    """Selecting by mid=0 on SDP with one video section should match get_video_codec_from_sdp."""
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("VP8"), _p("H264")],
    )
    by_mid = get_codec_from_sdp_by_mid(SDP_WITH_MIDS_VIDEO_AUDIO, "0")
    # First video section only (same as get_video_codec_from_sdp) -> VP8 at 96
    assert by_mid["codec"] == "VP8"
    assert by_mid["payload_type"] == 96


def test_get_codec_from_sdp_by_mid_raises_when_section_has_no_supported_codec(monkeypatch):
    """Section exists for mid but no codec in priority list -> ValueError."""
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("AV1")],  # SDP only offers VP8 and H264
    )
    with pytest.raises(ValueError, match="None of the supported video codecs"):
        get_codec_from_sdp_by_mid(SDP_WITH_MIDS_VIDEO_AUDIO, "0")


# ---------------------------------------------------------------------
# normalize_sdp_for_supported_codecs
# ---------------------------------------------------------------------

def test_normalize_sdp_for_supported_codecs_keeps_only_supported_video_codecs(monkeypatch):
    """Video section is reduced to PTs whose codec is in SUPPORTED_VIDEO_CODECS; RTX kept for kept primaries."""
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264"), _p("VP9"), _p("VP8"), _p("AV1")],
    )
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_AUDIO_CODECS",
        [_p("Opus")],
    )
    sdp = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96 97 98 99 102 103
a=rtpmap:96 VP8/90000
a=rtpmap:97 rtx/90000
a=fmtp:97 apt=96
a=rtpmap:98 foo/90000
a=rtpmap:99 bar/90000
a=rtpmap:102 H264/90000
a=fmtp:102 profile-level-id=42e01f;packetization-mode=1
a=rtpmap:103 VP9/90000
a=fmtp:103 profile-id=0
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=rtpmap:111 opus/48000/2
"""
    out = normalize_sdp_for_supported_codecs(sdp)
    lines = out.splitlines()
    m_video = [l for l in lines if l.startswith("m=video")]
    assert len(m_video) == 1
    pts = m_video[0].split()[4:]
    # Unsupported (98, 99) must be gone; only supported PTs (96 VP8, 97 RTX, 102 H264, 103 VP9) may appear
    assert "98" not in pts and "99" not in pts
    assert set(pts) <= {"96", "97", "102", "103"}
    assert len(pts) >= 3
    assert "a=rtpmap:98" not in out and "a=rtpmap:99" not in out
    assert "a=rtpmap:102 H264/90000" in out
    assert "a=rtpmap:103 VP9/90000" in out
    assert "m=audio" in out
    assert "a=rtpmap:111 opus" in out or "a=rtpmap:111 OPUS" in out


def test_normalize_sdp_for_supported_codecs_raises_when_video_has_no_supported_codecs():
    """Raises NoSupportedCodecsError when video section has no codec in SUPPORTED_VIDEO_CODECS."""
    sdp = """v=0
o=- 1 2 IN IP4 127.0.0.1
m=video 9 UDP/TLS/RTP/SAVPF 99
a=rtpmap:99 foo/90000
"""
    with pytest.raises(NoSupportedCodecsError) as exc_info:
        normalize_sdp_for_supported_codecs(sdp)
    assert "video" in str(exc_info.value).lower()


def test_normalize_sdp_for_supported_codecs_raises_when_audio_has_no_pts():
    """Raises NoSupportedCodecsError when audio section has no payload types."""
    sdp = """v=0
o=- 1 2 IN IP4 127.0.0.1
m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 VP8/90000
m=audio 9 UDP/TLS/RTP/SAVPF
"""
    with pytest.raises(NoSupportedCodecsError) as exc_info:
        normalize_sdp_for_supported_codecs(sdp)
    assert "audio" in str(exc_info.value).lower()


def test_normalize_sdp_for_supported_codecs_keeps_only_supported_audio_codecs(monkeypatch):
    """Audio section is reduced to PTs whose codec is in SUPPORTED_AUDIO_CODECS (Opus only)."""
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("VP8")],
    )
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_AUDIO_CODECS",
        [_p("Opus")],
    )
    sdp = """v=0
o=- 1 2 IN IP4 127.0.0.1
m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 VP8/90000
m=audio 9 UDP/TLS/RTP/SAVPF 111 8
a=rtpmap:111 opus/48000/2
a=rtpmap:8 PCMA/8000
"""
    out = normalize_sdp_for_supported_codecs(sdp)
    # PCMA (8) must be dropped; Opus (111) must be kept
    assert "a=rtpmap:111 opus" in out or "a=rtpmap:111 OPUS" in out
    assert "a=rtpmap:8 " not in out
    m_audio = [l for l in out.splitlines() if l.startswith("m=audio")]
    assert len(m_audio) >= 1
    pts = m_audio[0].split()[4:]
    assert "8" not in pts
    if pts:
        assert "111" in pts


def test_normalize_sdp_for_supported_codecs_raises_when_audio_has_no_supported_codecs():
    """Raises NoSupportedCodecsError when audio section has no codec in SUPPORTED_AUDIO_CODECS."""
    sdp = """v=0
o=- 1 2 IN IP4 127.0.0.1
m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 VP8/90000
m=audio 9 UDP/TLS/RTP/SAVPF 8
a=rtpmap:8 PCMA/8000
"""
    with pytest.raises(NoSupportedCodecsError) as exc_info:
        normalize_sdp_for_supported_codecs(sdp)
    assert "audio" in str(exc_info.value).lower()


def test_normalize_sdp_for_supported_codecs_filters_by_parameters(monkeypatch):
    """When an entry has parameters, only SDP lines with matching fmtp are kept."""
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS",
        [_p("H264", parameters={"profile-level-id": "42e01f", "packetization-mode": "1"})],
    )
    monkeypatch.setattr(
        "reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_AUDIO_CODECS",
        [_p("Opus")],
    )
    sdp = """v=0
o=- 1 2 IN IP4 127.0.0.1
m=video 9 UDP/TLS/RTP/SAVPF 98 102
a=rtpmap:98 H264/90000
a=fmtp:98 profile-level-id=4d002a;packetization-mode=1
a=rtpmap:102 H264/90000
a=fmtp:102 profile-level-id=42e01f;packetization-mode=1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=rtpmap:111 opus/48000/2
"""
    out = normalize_sdp_for_supported_codecs(sdp)
    # Only H264 with profile-level-id=42e01f (102) is kept; 98 is dropped (params don't match)
    assert "a=rtpmap:102 H264/90000" in out
    assert "a=fmtp:102 profile-level-id=42e01f" in out
    assert "a=rtpmap:98 " not in out
