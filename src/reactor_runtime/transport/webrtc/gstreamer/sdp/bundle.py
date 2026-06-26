from __future__ import annotations
import random
import uuid
from typing import Dict, List, Mapping, Optional, Literal, Tuple
from dataclasses import dataclass

from reactor_runtime.core import TrackDirection, TrackInfo, TrackKind

BundleMode = Literal["max-bundle", "max-compat", "bundle-invalid"]


@dataclass(frozen=True)
class BundleCheckResult:
    """
    Immutable result of BUNDLE validation analysis for a given SDP.

    This class provides both:
        - Raw diagnostic data
        - Semantic helpers for decision making in WebRTC pipelines

    Typical usage:

        result = detect_bundle_policy_from_sdp(sdp)

        if result.is_invalid():
            sdp, result = fix_sdp_to_max_compat_if_bundle_invalid(sdp)

        if result.is_max_bundle():
            # safe to assume shared ICE transport
    """

    mode: BundleMode
    has_bundle_group: bool
    bundle_mids: List[str]
    ufrag_by_mid: Dict[str, Optional[str]]
    pwd_by_mid: Dict[str, Optional[str]]
    fingerprint_by_mid: Dict[str, Optional[str]]
    reason: Optional[str] = None

    # ---------------------------------------------------------------------
    # Semantic helpers
    # ---------------------------------------------------------------------

    def is_max_bundle(self) -> bool:
        """True if SDP is valid and effectively max-bundle."""
        return self.mode == "max-bundle"

    def is_max_compat(self) -> bool:
        """True if SDP behaves as max-compat (no BUNDLE group)."""
        return self.mode == "max-compat"

    def is_invalid(self) -> bool:
        """True if SDP declares BUNDLE but it is invalid/inconsistent."""
        return self.mode == "bundle-invalid"

    def is_valid(self) -> bool:
        """
        True if SDP is in a valid transport mode
        (either max-bundle or max-compat).
        """
        return self.mode in ("max-bundle", "max-compat")

    # ---------------------------------------------------------------------
    # ICE consistency helpers
    # ---------------------------------------------------------------------

    def bundled_mids(self) -> List[str]:
        """
        Return mids that are part of the BUNDLE group.
        Empty list if no BUNDLE group.
        """
        return list(self.bundle_mids)

    def unique_ufrags(self) -> set:
        """Return set of distinct ICE ufrags across bundled mids."""
        return {
            self.ufrag_by_mid.get(mid)
            for mid in self.bundle_mids
            if mid in self.ufrag_by_mid
        }

    def unique_pwds(self) -> set:
        """Return set of distinct ICE passwords across bundled mids."""
        return {
            self.pwd_by_mid.get(mid)
            for mid in self.bundle_mids
            if mid in self.pwd_by_mid
        }

    def unique_fingerprints(self) -> set:
        """Return set of distinct fingerprints across bundled mids."""
        return {
            self.fingerprint_by_mid.get(mid)
            for mid in self.bundle_mids
            if mid in self.fingerprint_by_mid
        }

    def has_consistent_ice_credentials(self) -> bool:
        """
        True if bundled mids share exactly one ICE ufrag and one ICE pwd.
        Does NOT consider fingerprint.
        """
        return (
            len(self.unique_ufrags()) == 1
            and None not in self.unique_ufrags()
            and len(self.unique_pwds()) == 1
            and None not in self.unique_pwds()
        )

    def has_consistent_fingerprint(self) -> bool:
        """
        True if all bundled mids share identical fingerprint
        (ignoring None-only case).
        """
        fps = self.unique_fingerprints()
        if not fps:
            return True  # no fingerprint info present
        return len(fps) == 1 and None not in fps

    # ---------------------------------------------------------------------
    # Debug helpers
    # ---------------------------------------------------------------------

    def summary(self) -> str:
        """
        Compact human-readable summary useful for logs.
        """
        return (
            f"BundleCheckResult(mode={self.mode}, "
            f"bundle_mids={self.bundle_mids}, "
            f"reason={self.reason})"
        )


def _split_sdp_sections(sdp: str) -> Tuple[List[str], List[List[str]]]:
    """
    Split SDP into:
        - session-level lines (before first m=)
        - media sections (each starting with an m= line)

    Why this matters:
        BUNDLE is defined at session-level (a=group:BUNDLE),
        while ICE credentials (ice-ufrag/pwd) are defined per m-line.

    This function preserves ordering and original lines so the SDP
    can later be reconstructed without semantic changes.
    """
    # Normalize line endings to simplify parsing.
    # SDP spec uses CRLF, but implementations may mix.
    lines = sdp.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Remove trailing empty lines only.
    # Internal empty lines are preserved (rare but legal).
    while lines and lines[-1] == "":
        lines.pop()

    session_lines: List[str] = []
    media_sections: List[List[str]] = []
    cur_media: Optional[List[str]] = None

    for ln in lines:
        if ln.startswith("m="):
            # New media section begins
            if cur_media is not None:
                media_sections.append(cur_media)
            cur_media = [ln]
        else:
            # Still session-level until first m=
            if cur_media is None:
                session_lines.append(ln)
            else:
                cur_media.append(ln)

    # Append last media block if present
    if cur_media is not None:
        media_sections.append(cur_media)

    return session_lines, media_sections


def _get_attr_value(lines: List[str], key: str) -> Optional[str]:
    """
    Return the first a=<key>:<value> found in lines.

    Notes:
        - Only the first occurrence is considered.
        - This matches typical WebRTC SDP structure
          (one ice-ufrag, one ice-pwd, one fingerprint per m-line).
        - If attribute exists but value is empty, returns None.
    """
    prefix = f"a={key}:"
    for ln in lines:
        if ln.startswith(prefix):
            # Strip prefix and whitespace.
            # Return None if value becomes empty.
            return ln[len(prefix) :].strip() or None
    return None


def get_mids_by_mline(sdp: str) -> List[Optional[str]]:
    """
    Return the a=mid value for each m= audio or video section in SDP order.

    Used to map webrtcbin pad index (src_0, src_1, ...) to SDP mid.
    """
    _, media_sections = _split_sdp_sections(sdp)
    return [
        _get_attr_value(sec, "mid")
        for sec in media_sections
        if _is_video_or_audio_section(sec)
    ]


def _get_media_direction(lines: List[str]) -> Optional[str]:
    """
    Return the first a= direction found in the media section.

    SDP direction is a=sendonly, a=recvonly, a=sendrecv, or a=inactive
    (no value). Returns None if none of these appear (default is sendrecv).
    """
    for ln in lines:
        if ln in ("a=sendonly", "a=recvonly", "a=sendrecv", "a=inactive"):
            return ln[2:]  # strip "a="
    return None


def _get_media_kind(lines: List[str]) -> Optional[TrackKind]:
    """Return TrackKind from the m= line of a media section, or None."""
    if not lines or not lines[0].startswith("m="):
        return None
    # m=video ... or m=audio ...
    media_type = lines[0].split()[0][2:].lower()
    if media_type == "video":
        return TrackKind.VIDEO
    if media_type == "audio":
        return TrackKind.AUDIO
    return None


def tracks_from_sdp_offer(sdp: str) -> List[TrackInfo]:
    """
    From an SDP offer, return a list of TrackInfo for each media section
    that has a mid and a direction of sendonly or recvonly.

    - name: the a=mid value for the section.
    - direction: IN for sendonly (offerer sends → we receive), OUT for
      recvonly (offerer receives → we send).
    - sendrecv and inactive (and sections with no direction) are omitted.
    """
    _, media_sections = _split_sdp_sections(sdp)
    result: List[TrackInfo] = []
    for sec in media_sections:
        mid = _get_attr_value(sec, "mid")
        if not mid:
            continue
        direction = _get_media_direction(sec)
        if direction == "sendrecv" or direction == "inactive" or direction is None:
            continue
        if direction == "sendonly":
            track_direction = TrackDirection.IN
        elif direction == "recvonly":
            track_direction = TrackDirection.OUT
        else:
            continue
        kind = _get_media_kind(sec)
        if kind is None:
            continue
        result.append(
            TrackInfo(name=mid, kind=kind, rate=0.0, direction=track_direction)
        )
    return result


def _get_session_bundle_mids(session_lines: List[str]) -> List[str]:
    """
    Parse session-level:
        a=group:BUNDLE <mid1> <mid2> ...

    Returns:
        List of mids in the FIRST BUNDLE group found.

    Why only first?
        SDP allows multiple a=group lines (e.g. LS, FID),
        but WebRTC BUNDLE semantics use a single BUNDLE group.
    """
    for ln in session_lines:
        if ln.startswith("a=group:BUNDLE"):
            parts = ln.split()
            # parts[0] == "a=group:BUNDLE"
            # Remaining tokens are mids.
            return parts[1:]
    return []


def detect_bundle_policy_from_sdp(
    sdp: str, *, strict: bool = True
) -> BundleCheckResult:
    """
    Determine whether SDP behaves as:
        - 'max-bundle'
        - 'max-compat'
        - 'bundle-invalid'

    WebRTC context:
        max-bundle → single ICE/DTLS transport shared across m-lines.
        max-compat → separate transports per m-line.
        bundle-invalid → SDP declares BUNDLE but credentials inconsistent.

    Validation rules:
        - If no a=group:BUNDLE → max-compat.
        - If present:
            * All mids must exist.
            * All bundled mids must share identical:
                - ice-ufrag
                - ice-pwd
            * If strict=True:
                - fingerprint must also match (if present per m-line).

    Why fingerprint check is optional:
        Some stacks place fingerprint at session-level only.
        Some duplicate per m-line.
        strict=True enforces per-m-line consistency when present.
    """
    session_lines, media_sections = _split_sdp_sections(sdp)

    bundle_mids = _get_session_bundle_mids(session_lines)
    has_bundle_group = len(bundle_mids) > 0

    # Build map: mid -> media section
    # Required to inspect ICE creds per m-line.
    mid_to_section: Dict[str, List[str]] = {}
    for sec in media_sections:
        mid = _get_attr_value(sec, "mid")
        if mid:
            mid_to_section[mid] = sec

    ufrag_by_mid: Dict[str, Optional[str]] = {}
    pwd_by_mid: Dict[str, Optional[str]] = {}
    fp_by_mid: Dict[str, Optional[str]] = {}

    # -------------------------------------------------------------
    # Case 1: No BUNDLE group → implicit max-compat
    # -------------------------------------------------------------
    if not has_bundle_group:
        # Still extract diagnostics for debugging purposes.
        for mid, sec in mid_to_section.items():
            ufrag_by_mid[mid] = _get_attr_value(sec, "ice-ufrag")
            pwd_by_mid[mid] = _get_attr_value(sec, "ice-pwd")
            fp_by_mid[mid] = _get_attr_value(sec, "fingerprint")

        return BundleCheckResult(
            mode="max-compat",
            has_bundle_group=False,
            bundle_mids=[],
            ufrag_by_mid=ufrag_by_mid,
            pwd_by_mid=pwd_by_mid,
            fingerprint_by_mid=fp_by_mid,
            reason="No a=group:BUNDLE present",
        )

    # -------------------------------------------------------------
    # Case 2: BUNDLE group exists → validate consistency
    # -------------------------------------------------------------
    missing_mids: list[str] = []

    for mid in bundle_mids:
        if mid not in mid_to_section:
            # BUNDLE references mid that does not exist.
            missing_mids.append(mid)
            ufrag_by_mid[mid] = None
            pwd_by_mid[mid] = None
            fp_by_mid[mid] = None
            continue

        sec = mid_to_section[mid]

        ufrag_by_mid[mid] = _get_attr_value(sec, "ice-ufrag")
        pwd_by_mid[mid] = _get_attr_value(sec, "ice-pwd")
        fp_by_mid[mid] = _get_attr_value(sec, "fingerprint")

    # Invalid if BUNDLE references unknown mids.
    if missing_mids:
        return BundleCheckResult(
            mode="bundle-invalid",
            has_bundle_group=True,
            bundle_mids=bundle_mids,
            ufrag_by_mid=ufrag_by_mid,
            pwd_by_mid=pwd_by_mid,
            fingerprint_by_mid=fp_by_mid,
            reason=f"BUNDLE group references mids not found in SDP: {missing_mids}",
        )

    # Validate ufrag/pwd consistency
    ufrags = [ufrag_by_mid[m] for m in bundle_mids]
    pwds = [pwd_by_mid[m] for m in bundle_mids]

    if any(v is None for v in ufrags) or len(set(ufrags)) != 1:
        return BundleCheckResult(
            mode="bundle-invalid",
            has_bundle_group=True,
            bundle_mids=bundle_mids,
            ufrag_by_mid=ufrag_by_mid,
            pwd_by_mid=pwd_by_mid,
            fingerprint_by_mid=fp_by_mid,
            reason=f"Different or missing ice-ufrag among bundled m-lines: {ufrag_by_mid}",
        )

    if any(v is None for v in pwds) or len(set(pwds)) != 1:
        return BundleCheckResult(
            mode="bundle-invalid",
            has_bundle_group=True,
            bundle_mids=bundle_mids,
            ufrag_by_mid=ufrag_by_mid,
            pwd_by_mid=pwd_by_mid,
            fingerprint_by_mid=fp_by_mid,
            reason=f"Different or missing ice-pwd among bundled m-lines: {pwd_by_mid}",
        )

    if strict:
        fps = [fp_by_mid[m] for m in bundle_mids]
        # Some SDPs repeat fingerprint per m-line; if present, require consistency.
        # If it's missing in all, we ignore.
        if any(v is not None for v in fps) and (
            any(v is None for v in fps) or len(set(fps)) != 1
        ):
            return BundleCheckResult(
                mode="bundle-invalid",
                has_bundle_group=True,
                bundle_mids=bundle_mids,
                ufrag_by_mid=ufrag_by_mid,
                pwd_by_mid=pwd_by_mid,
                fingerprint_by_mid=fp_by_mid,
                reason=f"Different or missing fingerprint among bundled m-lines: {fp_by_mid}",
            )

    return BundleCheckResult(
        mode="max-bundle",
        has_bundle_group=True,
        bundle_mids=bundle_mids,
        ufrag_by_mid=ufrag_by_mid,
        pwd_by_mid=pwd_by_mid,
        fingerprint_by_mid=fp_by_mid,
        reason="BUNDLE present and ICE credentials consistent across bundled mids",
    )


def fix_sdp_to_max_compat_if_bundle_invalid(
    sdp: str, *, strict: bool = True
) -> Tuple[str, BundleCheckResult]:
    """
    If SDP has an invalid BUNDLE group (e.g., different ice-ufrag/pwd across bundled mids),
    rewrite SDP to max-compat by removing session-level 'a=group:BUNDLE ...' lines.

    Returns (possibly_modified_sdp, analysis_result).
    """
    analysis = detect_bundle_policy_from_sdp(sdp, strict=strict)
    if analysis.mode != "bundle-invalid":
        return sdp, analysis

    # Remove all session-level a=group:BUNDLE lines (keep any other a=group lines)
    session_lines, media_sections = _split_sdp_sections(sdp)

    new_session_lines: List[str] = []
    for ln in session_lines:
        if ln.startswith("a=group:BUNDLE"):
            continue
        new_session_lines.append(ln)

    # Reassemble with CRLF as per SDP convention
    out_lines: List[str] = []
    out_lines.extend(new_session_lines)
    for sec in media_sections:
        out_lines.extend(sec)

    fixed = "\r\n".join(out_lines) + "\r\n"

    # Re-run analysis on fixed SDP (should become max-compat)
    fixed_analysis = detect_bundle_policy_from_sdp(fixed, strict=strict)
    return fixed, fixed_analysis


def _is_video_or_audio_section(media_section: List[str]) -> bool:
    """True if the media section is m=video or m=audio."""
    if not media_section or not media_section[0].startswith("m="):
        return False
    media_type = media_section[0].split()[0][2:].lower()
    return media_type in ("video", "audio")


def _section_is_sendonly(media_section: List[str]) -> bool:
    """True if the media section has a=sendonly (answerer sends to offerer)."""
    return "a=sendonly" in (ln.strip() for ln in media_section)


def _section_has_ssrc(media_section: List[str]) -> bool:
    """True if the media section has at least one a=ssrc line."""
    return any(ln.strip().startswith("a=ssrc:") for ln in media_section)


def _ssrc_for_mid(ssrc_by_mid: Mapping[str, int], mid: str) -> Optional[int]:
    """Resolve SSRC for ``a=mid`` (``mid`` may be normalized via ``str()`` on keys)."""
    if mid in ssrc_by_mid:
        return int(ssrc_by_mid[mid])
    for k, v in ssrc_by_mid.items():
        if str(k) == mid:
            return int(v)
    return None


def _rewrite_sendonly_ssrc_signaling(
    media_section: List[str],
    _mid: str,
    master_ssrc: int,
    rtx_ssrc: Optional[int],
    shared_cname: Optional[str],
) -> List[str]:
    """
    Remove webrtc ``a=ssrc`` / ``a=ssrc-group`` lines, then append canonical signaling.

    When ``rtx_ssrc`` is set, emits ``a=ssrc-group:FID <master> <rtx>`` (RFC 5576),
    ``a=ssrc:<rtx> cname:`` (same CNAME as media), then media ``cname`` only.
    Otherwise only media ``cname`` (e.g. audio).
    """
    filtered: List[str] = []
    for ln in media_section:
        st = ln.strip()
        if st.startswith("a=ssrc-group:") or st.startswith("a=ssrc:"):
            continue
        filtered.append(ln)

    cname_val = shared_cname if shared_cname is not None else uuid.uuid4().hex[:16]
    inject: List[str] = []
    m = int(master_ssrc)
    if rtx_ssrc is not None:
        r = int(rtx_ssrc)
        inject.append(f"a=ssrc-group:FID {m} {r}")
        inject.append(f"a=ssrc:{m} cname:{cname_val}")
        inject.append(f"a=ssrc:{r} cname:{cname_val}")
    else:
        inject.append(f"a=ssrc:{m} cname:{cname_val}")
    insert_idx = len(filtered)
    for i, ln in enumerate(filtered):
        if ln.strip().startswith("a=fingerprint:"):
            insert_idx = i
            break
    return filtered[:insert_idx] + inject + filtered[insert_idx:]


def _get_mid_from_section(media_section: List[str]) -> Optional[str]:
    """Return the a=mid value from the media section, or None."""
    return _get_attr_value(media_section, "mid")


def add_answer_webrtc_attributes(
    sdp: str,
    ssrc_by_mid: Optional[Dict[str, int]] = None,
    cname: Optional[str] = None,
    rtx_ssrc_by_mid: Optional[Dict[str, int]] = None,
) -> str:
    """
    Add SSRC/cname to sendonly SDP answer sections for WebRTC compatibility.

    When ``ssrc_by_mid`` contains the section's ``a=mid``, existing ``a=ssrc`` and
    ``a=ssrc-group`` lines are removed and replaced with canonical signaling: optional
    ``a=ssrc-group:FID <master> <rtx>`` and ``a=ssrc:<rtx> cname:`` when
    ``rtx_ssrc_by_mid`` provides an RTX SSRC for that mid, then ``a=ssrc:<master>``
    ``cname`` (RFC 5576 FID; no ``a=ssrc`` ``msid`` lines are injected).

    For sendonly sections **not** listed in ``ssrc_by_mid`` that still have no
    ``a=ssrc`` lines, the legacy behavior applies: append random SSRC lines (or use
    ``ssrc_by_mid`` when the mid key exists).

    Callers must keep ``ssrc_by_mid`` / ``rtx_ssrc_by_mid`` aligned with the RTP
    payloader and ``rtprtxsend`` maps.

    Line endings are normalized to CRLF.
    """
    session_lines, media_sections = _split_sdp_sections(sdp)

    # Single CNAME for all sendonly sections when provided (RFC 3550).
    shared_cname = cname
    ssrc_map = ssrc_by_mid or {}
    rtx_map = rtx_ssrc_by_mid or {}

    ssrc_sections: List[List[str]] = []
    for sec in media_sections:
        if _is_video_or_audio_section(sec) and _section_is_sendonly(sec):
            mid = _get_mid_from_section(sec) or "track"
            master = _ssrc_for_mid(ssrc_map, mid)
            if master is not None:
                rtx = _ssrc_for_mid(rtx_map, mid)
                sec = _rewrite_sendonly_ssrc_signaling(
                    sec, mid, master, rtx, shared_cname
                )
            elif not _section_has_ssrc(sec):
                ssrc = _ssrc_for_mid(ssrc_map, mid)
                if ssrc is None:
                    ssrc = random.randint(1, 0x7FFFFFFF)
                cname_val = (
                    shared_cname if shared_cname is not None else uuid.uuid4().hex[:16]
                )
                ssrc_lines = [f"a=ssrc:{ssrc} cname:{cname_val}"]
                insert_idx = len(sec)
                for i, ln in enumerate(sec):
                    if ln.strip().startswith("a=fingerprint:"):
                        insert_idx = i
                        break
                sec = sec[:insert_idx] + ssrc_lines + sec[insert_idx:]
        ssrc_sections.append(sec)

    out_lines: List[str] = []
    out_lines.extend(session_lines)
    for sec in ssrc_sections:
        out_lines.extend(sec)

    return "\r\n".join(out_lines) + "\r\n"
