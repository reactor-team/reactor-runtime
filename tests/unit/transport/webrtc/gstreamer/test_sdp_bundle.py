"""
tests/transports/gstreamer/test_sdp_bundle.py

Covered Scenarios (Public API Only)
===================================

Purpose
-------
Validate BUNDLE policy detection and correction behavior in SDP
using ONLY public APIs:

    - detect_bundle_policy_from_sdp(...)
    - fix_sdp_to_max_compat_if_bundle_invalid(...)
    - Public methods of BundleCheckResult

Private helpers such as:
    - _split_sdp_sections
    - _get_attr_value
    - _get_session_bundle_mids

are NOT tested directly. They are exercised indirectly through
public API behavior.

---------------------------------------------------------------------

Scenarios Covered
-----------------

1) SDP Without a=group:BUNDLE
   - Classified as "max-compat"
   - Diagnostic maps (ufrag/pwd/fingerprint) are still populated
   - Semantic helpers:
       * is_max_compat()
       * is_valid()

2) Valid BUNDLE SDP (max-bundle)
   - a=group:BUNDLE present
   - All referenced mids exist
   - All bundled mids share identical:
       * ice-ufrag
       * ice-pwd
   - When strict=True:
       * fingerprints must match if present per m-line
   - Consistency helpers verified:
       * has_consistent_ice_credentials()
       * has_consistent_fingerprint()
       * unique_ufrags()
       * unique_pwds()
       * unique_fingerprints()

3) Invalid BUNDLE Due to ICE Inconsistency
   - Different ice-ufrag values -> bundle-invalid
   - Different ice-pwd values   -> bundle-invalid
   - Result.reason contains diagnostic explanation

4) Invalid BUNDLE Due to Missing MID
   - BUNDLE references a mid not defined in any m-line
   - That mid appears in diagnostic maps with None values
   - Classified as bundle-invalid

5) Fingerprint Validation in Strict Mode
   - strict=True:
       * Mismatched fingerprints across bundled m-lines -> bundle-invalid
       * Fingerprint present in only one bundled m-line -> bundle-invalid
   - strict=False:
       * Fingerprint mismatch does NOT invalidate SDP
         (as long as ICE credentials are consistent)

6) fix_sdp_to_max_compat_if_bundle_invalid(...)
   - When SDP is bundle-invalid:
       * Removes only session-level "a=group:BUNDLE ..." lines
       * Preserves other "a=group:*" lines (e.g., LS, FID)
       * Re-analyzes and results in max-compat
       * Output uses CRLF and ends with CRLF
   - When SDP is already valid (max-bundle or max-compat):
       * Function is a no-op
       * SDP is returned unchanged

---------------------------------------------------------------------

Notes
-----
- Test SDPs are minimal and focused on attributes relevant to BUNDLE.
- Newlines are normalized in comparisons to avoid fragility between LF and CRLF.
- These tests validate observable behavior, not internal implementation.
"""

import pytest


from reactor_runtime.transport.webrtc.gstreamer.sdp.bundle import (
    add_answer_webrtc_attributes,
    detect_bundle_policy_from_sdp,
    fix_sdp_to_max_compat_if_bundle_invalid,
    tracks_from_sdp_offer,
)
from reactor_runtime.core import TrackDirection, TrackInfo, TrackKind


def _normalize_newlines(s: str) -> str:
    # Helps compare SDPs without being fragile to \r\n vs \n
    return s.replace("\r\n", "\n").replace("\r", "\n")


SDP_MAX_COMPAT_NO_BUNDLE = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=msid-semantic: WMS
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:ua
a=ice-pwd:pa
a=fingerprint:sha-256 AA:BB
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=ice-ufrag:uv
a=ice-pwd:pv
a=fingerprint:sha-256 CC:DD
"""

SDP_MAX_BUNDLE_OK = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:u
a=ice-pwd:p
a=fingerprint:sha-256 AA:BB
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=ice-ufrag:u
a=ice-pwd:p
a=fingerprint:sha-256 AA:BB
"""

SDP_BUNDLE_BAD_UFRAG = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:u1
a=ice-pwd:p
a=fingerprint:sha-256 AA:BB
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=ice-ufrag:u2
a=ice-pwd:p
a=fingerprint:sha-256 AA:BB
"""

SDP_BUNDLE_BAD_PWD = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:u
a=ice-pwd:p1
a=fingerprint:sha-256 AA:BB
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=ice-ufrag:u
a=ice-pwd:p2
a=fingerprint:sha-256 AA:BB
"""

SDP_BUNDLE_MISSING_MID = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 2
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:u
a=ice-pwd:p
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=ice-ufrag:u
a=ice-pwd:p
"""

SDP_BUNDLE_FP_MISMATCH = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:u
a=ice-pwd:p
a=fingerprint:sha-256 AA:BB
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=ice-ufrag:u
a=ice-pwd:p
a=fingerprint:sha-256 CC:DD
"""

SDP_BUNDLE_FP_PRESENT_ONLY_ONE_MLINE = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:u
a=ice-pwd:p
a=fingerprint:sha-256 AA:BB
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=ice-ufrag:u
a=ice-pwd:p
"""


def test_detect_no_bundle_is_max_compat_and_extracts_diagnostics():
    r = detect_bundle_policy_from_sdp(SDP_MAX_COMPAT_NO_BUNDLE)

    assert r.is_max_compat()
    assert r.is_valid()
    assert not r.has_bundle_group
    assert r.bundle_mids == []
    assert "No a=group:BUNDLE" in (r.reason or "")

    # Even in max-compat, it extracts diagnostics by mid.
    assert r.ufrag_by_mid["0"] == "ua"
    assert r.pwd_by_mid["0"] == "pa"
    assert r.fingerprint_by_mid["0"] == "sha-256 AA:BB"

    assert r.ufrag_by_mid["1"] == "uv"
    assert r.pwd_by_mid["1"] == "pv"
    assert r.fingerprint_by_mid["1"] == "sha-256 CC:DD"


def test_detect_bundle_ok_is_max_bundle_and_consistent_helpers_true():
    r = detect_bundle_policy_from_sdp(SDP_MAX_BUNDLE_OK, strict=True)

    assert r.is_max_bundle()
    assert r.has_bundle_group
    assert r.bundle_mids == ["0", "1"]
    assert r.has_consistent_ice_credentials() is True
    assert r.has_consistent_fingerprint() is True

    assert r.unique_ufrags() == {"u"}
    assert r.unique_pwds() == {"p"}
    assert r.unique_fingerprints() == {"sha-256 AA:BB"}


@pytest.mark.parametrize(
    "sdp, expected_reason_substr",
    [
        (SDP_BUNDLE_BAD_UFRAG, "ice-ufrag"),
        (SDP_BUNDLE_BAD_PWD, "ice-pwd"),
    ],
)
def test_detect_bundle_inconsistent_ice_is_invalid(sdp: str, expected_reason_substr: str):
    r = detect_bundle_policy_from_sdp(sdp, strict=True)

    assert r.is_invalid()
    assert r.has_bundle_group
    assert r.bundle_mids == ["0", "1"]
    assert expected_reason_substr in (r.reason or "")


def test_detect_bundle_referencing_missing_mid_is_invalid_and_sets_none_values():
    r = detect_bundle_policy_from_sdp(SDP_BUNDLE_MISSING_MID, strict=True)

    assert r.is_invalid()
    assert r.has_bundle_group
    assert r.bundle_mids == ["0", "2"]
    assert "references mids not found" in (r.reason or "")

    # mid "2" doesn't exist, it should appear as None in the maps
    assert r.ufrag_by_mid["2"] is None
    assert r.pwd_by_mid["2"] is None
    assert r.fingerprint_by_mid["2"] is None


def test_detect_bundle_fingerprint_mismatch_is_invalid_when_strict():
    r = detect_bundle_policy_from_sdp(SDP_BUNDLE_FP_MISMATCH, strict=True)

    assert r.is_invalid()
    assert "fingerprint" in (r.reason or "")


def test_detect_bundle_fingerprint_mismatch_is_ignored_when_not_strict():
    r = detect_bundle_policy_from_sdp(SDP_BUNDLE_FP_MISMATCH, strict=False)

    assert r.is_max_bundle()
    assert r.has_consistent_ice_credentials() is True


def test_detect_bundle_fingerprint_present_only_one_mline_invalid_in_strict():
    r = detect_bundle_policy_from_sdp(SDP_BUNDLE_FP_PRESENT_ONLY_ONE_MLINE, strict=True)

    assert r.is_invalid()
    assert "fingerprint" in (r.reason or "")


def test_fix_sdp_to_max_compat_removes_bundle_group_and_returns_max_compat():
    fixed_sdp, fixed_result = fix_sdp_to_max_compat_if_bundle_invalid(
        SDP_BUNDLE_BAD_UFRAG, strict=True
    )

    assert fixed_result.is_max_compat()
    assert fixed_result.has_bundle_group is False
    assert fixed_result.bundle_mids == []

    norm = _normalize_newlines(fixed_sdp)
    assert "a=group:BUNDLE" not in norm

    # Function rewrites with CRLF and also ends with CRLF
    assert fixed_sdp.endswith("\r\n")


def test_fix_sdp_noop_when_already_valid():
    fixed_sdp, fixed_result = fix_sdp_to_max_compat_if_bundle_invalid(
        SDP_MAX_BUNDLE_OK, strict=True
    )

    assert fixed_result.is_max_bundle()
    assert _normalize_newlines(fixed_sdp) == _normalize_newlines(SDP_MAX_BUNDLE_OK)


def test_fix_preserves_other_group_lines_but_removes_bundle_only():
    sdp = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:LS 0 1
a=group:BUNDLE 0 1
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:u1
a=ice-pwd:p
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=ice-ufrag:u2
a=ice-pwd:p
"""
    fixed_sdp, fixed_result = fix_sdp_to_max_compat_if_bundle_invalid(sdp, strict=True)

    assert fixed_result.is_max_compat()

    norm = _normalize_newlines(fixed_sdp)
    assert "a=group:BUNDLE" not in norm
    assert "a=group:LS 0 1" in norm


# ---------------------------------------------------------------------------
# tracks_from_sdp_offer
# ---------------------------------------------------------------------------

SDP_OFFER_SENDONLY_RECVONLY = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:webcam
a=sendonly
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:mic
a=sendonly
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:main_video
a=recvonly
"""

SDP_OFFER_SENDRECV_AND_INACTIVE = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:sendrecv_track
a=sendrecv
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:inactive_track
a=inactive
"""

SDP_OFFER_NO_DIRECTION = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:no_dir
"""

SDP_OFFER_NO_MID = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=sendonly
"""

SDP_OFFER_EMPTY_SESSION = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
"""

SDP_OFFER_UNKNOWN_MEDIA = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=application 9 UDP/DTLS/SCTP webrtc-datachannel
a=mid:data
a=sendonly
"""


def test_tracks_from_sdp_offer_sendonly_recvonly():
    """sendonly -> IN, recvonly -> OUT; name is mid; kind from m= line."""
    tracks = tracks_from_sdp_offer(SDP_OFFER_SENDONLY_RECVONLY)

    assert len(tracks) == 3

    assert tracks[0] == TrackInfo(
        name="webcam", kind=TrackKind.VIDEO, rate=0.0, direction=TrackDirection.IN
    )
    assert tracks[1] == TrackInfo(
        name="mic", kind=TrackKind.AUDIO, rate=0.0, direction=TrackDirection.IN
    )
    assert tracks[2] == TrackInfo(
        name="main_video",
        kind=TrackKind.VIDEO,
        rate=0.0,
        direction=TrackDirection.OUT,
    )


def test_tracks_from_sdp_offer_sendrecv_and_inactive_omitted():
    """Sections with sendrecv or inactive are not included."""
    tracks = tracks_from_sdp_offer(SDP_OFFER_SENDRECV_AND_INACTIVE)

    assert tracks == []


def test_tracks_from_sdp_offer_no_direction_omitted():
    """Section with no a= direction (default sendrecv) is omitted."""
    tracks = tracks_from_sdp_offer(SDP_OFFER_NO_DIRECTION)

    assert tracks == []


def test_tracks_from_sdp_offer_no_mid_omitted():
    """Section without a=mid is omitted."""
    tracks = tracks_from_sdp_offer(SDP_OFFER_NO_MID)

    assert tracks == []


def test_tracks_from_sdp_offer_empty_or_no_media():
    """No media sections -> empty list."""
    assert tracks_from_sdp_offer(SDP_OFFER_EMPTY_SESSION) == []


def test_tracks_from_sdp_offer_unknown_media_kind_omitted():
    """Section with non-video/audio (e.g. application) is omitted."""
    tracks = tracks_from_sdp_offer(SDP_OFFER_UNKNOWN_MEDIA)

    assert tracks == []


# Minimal SDP answer (no extmap-allow-mixed, no msid-semantic, no sdes:mid in media)
SDP_ANSWER_MINIMAL = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:0
a=sendonly
a=rtcp:9 IN IP4 0.0.0.0
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:1
a=sendonly
a=rtcp:9 IN IP4 0.0.0.0
"""


def test_add_answer_webrtc_attributes_adds_ssrc_for_sendonly():
    """Sendonly video/audio sections without a=ssrc get a=ssrc cname only (no msid injection)."""
    out = add_answer_webrtc_attributes(SDP_ANSWER_MINIMAL)

    assert "a=ssrc:" in out
    lines = out.replace("\r\n", "\n").split("\n")
    ssrc_lines = [ln for ln in lines if ln.strip().startswith("a=ssrc:")]
    msid_lines = [ln for ln in lines if "msid:" in ln and ln.strip().startswith("a=ssrc:")]
    assert len(msid_lines) == 0
    assert len(ssrc_lines) == 2  # one cname line per sendonly section (mid 0 and 1)
    cname_lines = [ln for ln in lines if "cname:" in ln and ln.strip().startswith("a=ssrc:")]
    assert len(cname_lines) == 2, "expect one cname line per sendonly section (mid 0 and 1)"
    for ln in cname_lines:
        assert " cname:" in ln


def test_add_answer_webrtc_attributes_uses_ssrc_by_mid_when_provided():
    """When ssrc_by_mid is provided, SDP uses those SSRCs (must match RTP payloader)."""
    ssrc_by_mid = {"0": 12345678, "1": 87654321}
    out = add_answer_webrtc_attributes(SDP_ANSWER_MINIMAL, ssrc_by_mid=ssrc_by_mid)

    assert "a=ssrc:12345678 " in out or "a=ssrc:12345678\r" in out
    assert "a=ssrc:87654321 " in out or "a=ssrc:87654321\r" in out


def test_add_answer_webrtc_attributes_uses_ssrc_by_mid_and_single_cname():
    """When ssrc_by_mid and cname are provided (client pattern), SDP uses those SSRCs and one CNAME."""
    ssrc_by_mid = {"0": 12345678, "1": 87654321}
    cname = "reactor-peer-abc"
    out = add_answer_webrtc_attributes(
        SDP_ANSWER_MINIMAL, ssrc_by_mid=ssrc_by_mid, cname=cname
    )
    assert "a=ssrc:12345678 " in out or "a=ssrc:12345678\r" in out
    assert "a=ssrc:87654321 " in out or "a=ssrc:87654321\r" in out
    cname_lines = [ln for ln in out.split("\r\n") if "cname:" in ln]
    assert len(cname_lines) == 2
    for ln in cname_lines:
        assert f"cname:{cname}" in ln


def test_add_answer_webrtc_attributes_uses_single_cname_when_provided():
    """When cname is provided, all sendonly sections use the same CNAME (RFC 3550)."""
    ssrc_by_mid = {"0": 111, "1": 222}
    shared_cname = "reactor-endpoint-1"
    out = add_answer_webrtc_attributes(
        SDP_ANSWER_MINIMAL, ssrc_by_mid=ssrc_by_mid, cname=shared_cname
    )
    cname_lines = [ln for ln in out.split("\r\n") if "cname:" in ln]
    assert len(cname_lines) == 2
    for ln in cname_lines:
        assert f"cname:{shared_cname}" in ln


SDP_VIDEO_SENDONLY_WITH_FID = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 98 99
a=mid:0
a=sendonly
a=ssrc-group:FID 0 11111111
a=ssrc:11111111 cname:webrtc-rtx
a=ssrc:22222222 cname:webrtc-rtx
a=ssrc:22222222 msid:oldstream oldtrack
a=fingerprint:sha-256 AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA:AA
"""


def test_add_answer_webrtc_attributes_rewrites_fid_with_rtx_ssrc_by_mid():
    """webrtc FID / a=ssrc lines are replaced with payloader + RTX SSRCs from maps."""
    out = add_answer_webrtc_attributes(
        SDP_VIDEO_SENDONLY_WITH_FID,
        ssrc_by_mid={"0": 1663701988},
        rtx_ssrc_by_mid={"0": 728665557},
        cname="reactor-cn",
    )
    norm = out.replace("\r\n", "\n")
    assert "a=ssrc-group:FID 1663701988 728665557" in norm
    assert "a=ssrc-group:FID 0 " not in norm
    assert "11111111" not in norm
    assert "22222222" not in norm
    assert "a=ssrc:728665557 cname:reactor-cn" in norm
    assert "a=ssrc:1663701988 cname:reactor-cn" in norm
    msid_lines = [
        ln for ln in norm.split("\n") if "msid:" in ln and ln.strip().startswith("a=ssrc:")
    ]
    assert len(msid_lines) == 0
