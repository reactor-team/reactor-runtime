"""
tests/transports/gstreamer/test_sdp_ice.py

Covered Scenarios (Public API Only)
===================================

Purpose
-------
Validate ICE candidate stripping and extraction behavior using ONLY
the public function:

    - strip_ice_candidates_from_sdp(sdp: str) -> (sanitized_sdp: str, candidates: List[IceCandidate])

No tests call or assert directly on any private helpers (there are none here),
and no tests depend on internal implementation details beyond the documented
public behavior and dataclass fields.

---------------------------------------------------------------------

Scenarios Covered
-----------------

1) Stripping ICE candidate lines
   - Removes all "a=candidate:..." lines from the returned SDP
   - Removes all "a=end-of-candidates" lines from the returned SDP
   - Preserves other SDP lines exactly (including "a=ice-ufrag", "a=ice-pwd", and unrelated attrs)

2) Candidate extraction correctness
   - Extracts candidate lines as IceCandidate objects with:
       * candidate: full attribute value without "a=" prefix (starts with "candidate:")
       * mline_index: correct zero-based media section index
       * mid: the last seen a=mid:<X> for that media section, or None if not present

3) mline_index behavior and edge cases
   - Candidates before the first "m=" section are assigned mline_index=0 (defensive behavior)
   - Candidates in subsequent sections increment mline_index appropriately

4) MID handling
   - MID resets to None at each new "m=" section
   - If a candidate appears before "a=mid:" within the same section, its mid is None
   - If a=mid is present, candidates after it use that mid

5) Newline normalization and output formatting
   - Input SDPs with mixed newline styles (\n, \r\n, \r) are parsed consistently
   - Output SDP is reassembled using CRLF and always ends with CRLF

6) Empty line handling
   - Fully empty lines are dropped during parsing (per implementation)
   - Output does not contain blank lines from the input

7) Idempotence (behavioral)
   - Running strip_ice_candidates_from_sdp() on an SDP that already has no candidates
     produces an empty candidates list and a semantically equivalent SDP (CRLF-normalized)

---------------------------------------------------------------------

Notes
-----
- SDP fixtures are minimal and focused on ICE-related attributes.
- Assertions normalize newlines where appropriate to avoid brittle comparisons.
"""

import pytest

from reactor_runtime.transport.webrtc.gstreamer.sdp.ice import strip_ice_candidates_from_sdp


def _norm(s: str) -> str:
    """Normalize newlines for stable assertions."""
    return s.replace("\r\n", "\n").replace("\r", "\n")


SDP_WITH_TWO_MLINES_AND_MIDS = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=ice-ufrag:abc
a=ice-pwd:def
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=candidate:1 1 udp 2122260223 192.168.0.2 54321 typ host
a=end-of-candidates
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:1
a=candidate:2 1 udp 2122260223 10.0.0.5 60000 typ host
a=candidate:3 1 udp 2122260223 10.0.0.6 60001 typ host
"""

SDP_WITH_CANDIDATE_BEFORE_MID = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=candidate:1 1 udp 2122260223 192.168.0.2 54321 typ host
a=mid:audio-mid
a=candidate:2 1 udp 2122260223 192.168.0.3 54322 typ host
"""

SDP_CANDIDATE_BEFORE_FIRST_MLINE = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
a=candidate:99 1 udp 2122260223 192.168.0.9 55555 typ host
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
"""

SDP_WITH_MIXED_NEWLINES = "v=0\r\no=- 1 2 IN IP4 127.0.0.1\rs=-\n" \
                          "t=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n" \
                          "a=mid:0\n" \
                          "a=candidate:1 1 udp 1 1.1.1.1 1111 typ host\r" \
                          "a=end-of-candidates\r\n" \
                          "a=ice-ufrag:abc\r\n" \
                          "a=ice-pwd:def\r\n"


SDP_WITHOUT_CANDIDATES = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=mid:0
a=ice-ufrag:abc
a=ice-pwd:def
"""


def test_strips_candidate_and_end_of_candidates_lines_and_preserves_others():
    sanitized, cands = strip_ice_candidates_from_sdp(SDP_WITH_TWO_MLINES_AND_MIDS)

    # Candidate lines removed from SDP
    assert "a=candidate:" not in sanitized
    assert "a=end-of-candidates" not in sanitized

    # ICE creds preserved
    assert "a=ice-ufrag:abc" in sanitized
    assert "a=ice-pwd:def" in sanitized

    # Media lines preserved
    assert "m=audio" in sanitized
    assert "m=video" in sanitized
    assert "a=mid:0" in sanitized
    assert "a=mid:1" in sanitized

    # Extracted candidates count
    assert len(cands) == 3


def test_extracts_candidate_objects_with_correct_fields():
    sanitized, cands = strip_ice_candidates_from_sdp(SDP_WITH_TWO_MLINES_AND_MIDS)

    assert [c.mline_index for c in cands] == [0, 1, 1]
    assert [c.mid for c in cands] == ["0", "1", "1"]

    assert cands[0].candidate.startswith("candidate:1 ")
    assert cands[1].candidate.startswith("candidate:2 ")
    assert cands[2].candidate.startswith("candidate:3 ")

    # Must NOT include "a=" prefix in stored candidate
    assert not cands[0].candidate.startswith("a=")


def test_mid_is_none_for_candidates_before_mid_in_same_section_then_set_after_mid():
    _, cands = strip_ice_candidates_from_sdp(SDP_WITH_CANDIDATE_BEFORE_MID)

    assert len(cands) == 2

    # First candidate appears before a=mid => mid None
    assert cands[0].mline_index == 0
    assert cands[0].mid is None

    # Second candidate appears after a=mid => mid set
    assert cands[1].mline_index == 0
    assert cands[1].mid == "audio-mid"


def test_candidate_before_first_mline_gets_defensive_mline_index_zero():
    _, cands = strip_ice_candidates_from_sdp(SDP_CANDIDATE_BEFORE_FIRST_MLINE)

    assert len(cands) == 1
    assert cands[0].mline_index == 0  # defensive max(mline_index, 0)
    assert cands[0].mid is None
    assert cands[0].candidate.startswith("candidate:99 ")


def test_output_is_crlf_and_always_ends_with_crlf():
    sanitized, _ = strip_ice_candidates_from_sdp(SDP_WITH_TWO_MLINES_AND_MIDS)

    assert "\r\n" in sanitized
    assert sanitized.endswith("\r\n")


def test_mixed_newlines_are_parsed_and_output_is_normalized_crlf():
    sanitized, cands = strip_ice_candidates_from_sdp(SDP_WITH_MIXED_NEWLINES)

    # Candidate removed and extracted
    assert len(cands) == 1
    assert "a=candidate:" not in sanitized
    assert "a=end-of-candidates" not in sanitized

    # Output CRLF normalized
    assert "\r\n" in sanitized
    assert sanitized.endswith("\r\n")

    # Preserves ice creds
    assert "a=ice-ufrag:abc" in sanitized
    assert "a=ice-pwd:def" in sanitized


def test_empty_lines_are_dropped():
    sdp = """v=0

o=- 1 2 IN IP4 127.0.0.1

m=audio 9 UDP/TLS/RTP/SAVPF 111

a=mid:0

a=candidate:1 1 udp 1 1.1.1.1 1111 typ host

"""
    sanitized, _ = strip_ice_candidates_from_sdp(sdp)

    # Implementation drops fully empty lines; output should not contain blank lines.
    # We assert no double-newline once normalized (a conservative check).
    assert "\n\n" not in _norm(sanitized)


def test_idempotent_when_no_candidates_present_candidates_list_empty_and_sdp_crlf_normalized():
    sanitized1, cands1 = strip_ice_candidates_from_sdp(SDP_WITHOUT_CANDIDATES)
    sanitized2, cands2 = strip_ice_candidates_from_sdp(sanitized1)

    assert cands1 == []
    assert cands2 == []

    # Function normalizes to CRLF and adds trailing CRLF; idempotence should hold
    assert sanitized1 == sanitized2
