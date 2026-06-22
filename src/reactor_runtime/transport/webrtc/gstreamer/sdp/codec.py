import re
from typing import Dict, List, Optional, Set, Tuple, TypedDict

from typing import NotRequired  # NotRequired in typing from 3.11

from reactor_runtime.transport.webrtc.gstreamer.errors import WebRTCNoVideoError


class NoSupportedCodecsError(ValueError):
    """Raised when a video or audio media section ends with no codec options."""


class CodecEntry(TypedDict):
    """Canonical shape for a supported codec (video or audio).

    Used in settings.SUPPORTED_VIDEO_CODECS, SUPPORTED_AUDIO_CODECS, and returned
    by get_video_codec_from_sdp. When parameters is defined, every key MUST have
    a string value (no None). When returned from get_video_codec_from_sdp,
    payload_type is the PT from the SDP offer where the codec was found.
    """

    codec: str
    parameters: NotRequired[Dict[str, str]]
    payload_type: NotRequired[Optional[int]]


# =========================================================================
# SDP: Codecs and payload types extraction
# =========================================================================


def _normalize_encoding_name(name: str) -> str:
    """
    Normalize codec names coming from SDP or priority list so comparisons
    become deterministic.

    Examples:
        H.264  -> H264
        H-265  -> H265
        HEVC   -> H265
    """
    # Defensive: accept None/empty and normalize to upper case
    n = (name or "").strip().upper()

    # Remove common punctuation differences between implementations
    # Chrome / Safari / FF may vary in formatting.
    n = n.replace(".", "")  # H.264/H.265 -> H264/H265
    n = n.replace("-", "")  # H-264 -> H264
    n = n.replace("_", "")

    # Handle known aliases
    if n in ("HEVC",):
        return "H265"

    # Defensive mapping for malformed or unexpected values
    if n in ("H26", "H2651"):
        return "H265"

    return n


def parse_fmtp_params(fmtp_value: str) -> Dict[str, Optional[str]]:
    """
    Parse an a=fmtp parameter string into a dictionary.

    Example input:
        "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"
        "usedtx;stereo=1"

    Output:
        {
            "level-asymmetry-allowed": "1",
            "packetization-mode": "1",
            "profile-level-id": "42e01f"
        }

    Flags without '=' (e.g. "usedtx") are mapped to None.

    This function does NOT validate semantics — it only tokenizes.
    """
    out: Dict[str, Optional[str]] = {}

    if not fmtp_value:
        # Gracefully handle empty or missing fmtp
        return out

    # SDP uses ';' as separator but implementations may add spaces.
    parts = [p.strip() for p in fmtp_value.strip().split(";") if p.strip()]

    for part in parts:
        if "=" in part:
            # Split only once: values themselves might theoretically contain '='
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                out[k] = v
        else:
            # Flag-style parameter (no value)
            out[part] = None

    return out


def _parse_priority_entry(entry: CodecEntry) -> Tuple[str, Dict[str, str]]:
    """
    Parse a priority entry into (normalized_codec_name, parameters).

    Entry must be a dict with:
      - "codec": codec name (e.g. "H264", "VP8")
      - "parameters": optional dict of fmtp params; when set, every key MUST have a string value
    """
    codec = entry.get("codec")
    if not codec or not isinstance(codec, str):
        return ("", {})
    name = _normalize_encoding_name(codec)
    params = entry.get("parameters")
    if not isinstance(params, dict):
        params = {}
    # Only include key-value pairs with a string value (Dict[str, str])
    params = {k: v for k, v in params.items() if v is not None and isinstance(v, str)}
    return (name, params)


def get_video_codec_from_sdp(sdp: str) -> CodecEntry:
    """
    Parse an SDP offer and return the selected supported codec entry.

    The list of supported codecs is read from
    :data:`reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS`.
    The first entry in that list whose codec name appears in the SDP offer is
    chosen. The returned entry has:
      - "codec": normalized codec name
      - "payload_type": PT from the SDP offer where the codec was found
      - "parameters": SDP fmtp merged with entry's parameters (entry wins)

    Args:
        sdp: The SDP offer string.

    Returns:
        The selected CodecEntry with resolved payload type and
        merged parameters.

    Raises:
        ValueError if:
            - SDP malformed
            - No m=video
            - No matching codec found in settings list
    """
    if not sdp or not isinstance(sdp, str):
        raise ValueError("sdp must be a non-empty string")

    # Lazy import to avoid circular import (settings imports sdp)
    from reactor_runtime.transport.webrtc.gstreamer.settings import SUPPORTED_VIDEO_CODECS

    priority = SUPPORTED_VIDEO_CODECS
    prio: List[str] = []
    fmtp_overrides: Dict[str, Dict[str, str]] = {}
    for entry in priority:
        name, params = _parse_priority_entry(entry)
        if name:
            prio.append(name)
            fmtp_overrides[name] = params or {}

    # ---------------------------------------------------------------------
    # Step 1: Extract first m=video section
    # ---------------------------------------------------------------------
    # We only consider the first video m-line.
    # In WebRTC typical SDP there is only one video m-line,
    # but spec allows multiple.
    m = re.search(r"(?m)^m=video[^\n]*\n", sdp)
    if not m:
        raise WebRTCNoVideoError("No m=video line in SDP")

    start = m.start()

    # Find next m= section to delimit this media block
    next_m = re.search(r"(?m)^m=[^\n]*\n", sdp[m.end() :])
    end = (m.end() + next_m.start()) if next_m else len(sdp)

    video_section = sdp[start:end]

    # ---------------------------------------------------------------------
    # Step 2: Parse PT list from m=video line
    # ---------------------------------------------------------------------
    mline = re.search(r"(?m)^m=video\s+\d+\s+\S+\s+(.+)$", video_section)
    if not mline:
        raise ValueError("Malformed m=video line in SDP")

    # PTs appear after transport (e.g., UDP/TLS/RTP/SAVPF 96 97 98 ...)
    pt_order = [int(t) for t in mline.group(1).strip().split() if t.isdigit()]
    if not pt_order:
        raise ValueError("No payload types found in m=video line")

    # ---------------------------------------------------------------------
    # Step 3: Build PT -> codec map from a=rtpmap
    # ---------------------------------------------------------------------
    pt_to_codec: Dict[int, str] = {}

    for line in video_section.splitlines():
        line = line.strip()
        if not line.startswith("a=rtpmap:"):
            continue

        # Example:
        #   a=rtpmap:96 VP8/90000
        #   a=rtpmap:98 H264/90000
        mm = re.match(r"a=rtpmap:(\d+)\s+([A-Za-z0-9\-\._]+)\/\d+(?:\/\d+)?", line)
        if not mm:
            continue

        pt = int(mm.group(1))
        codec = _normalize_encoding_name(mm.group(2))
        pt_to_codec[pt] = codec

    # ---------------------------------------------------------------------
    # Step 4: Build PT -> fmtp map
    # ---------------------------------------------------------------------
    pt_to_fmtp: Dict[int, Dict[str, Optional[str]]] = {}

    for line in video_section.splitlines():
        line = line.strip()
        if not line.startswith("a=fmtp:"):
            continue

        # Example:
        #   a=fmtp:98 profile-level-id=42e01f;packetization-mode=1
        mm = re.match(r"a=fmtp:(\d+)\s+(.+)$", line)
        if not mm:
            continue

        pt = int(mm.group(1))
        params_str = mm.group(2).strip()
        parsed = parse_fmtp_params(params_str)

        # Merge if multiple fmtp lines exist for same PT
        # (last occurrence wins for duplicate keys)
        existing = pt_to_fmtp.get(pt)
        if existing is None:
            pt_to_fmtp[pt] = parsed
        else:
            existing.update(parsed)

    # ---------------------------------------------------------------------
    # Step 5: Select codec using priority + PT order; require SDP params to match entry
    # ---------------------------------------------------------------------
    # When an entry has parameters, the SDP must offer the same values for that codec;
    # otherwise we do not select that codec (try next in list or raise).
    for wanted in prio:
        for pt in pt_order:
            if pt_to_codec.get(pt) != wanted:
                continue
            sdp_fmtp = pt_to_fmtp.get(pt, {})
            overrides = fmtp_overrides.get(wanted, {})
            # If our entry requires specific parameters, SDP must have exactly those values
            if overrides:
                sdp_matches = True
                for k, v in overrides.items():
                    sdp_val = sdp_fmtp.get(k) if sdp_fmtp else None
                    if sdp_val != v:
                        sdp_matches = False
                        break
                if not sdp_matches:
                    continue  # Codec present but parameters don't match; try next
            # Merge SDP fmtp with priority entry parameters (entry wins)
            merged_fmtp = {**sdp_fmtp, **overrides}
            # parameters: only keys with a string value (no None); compatible with Dict[str, str]
            parameters: Dict[str, str] = {
                k: v for k, v in merged_fmtp.items() if v is not None
            }
            return {
                "codec": wanted,
                "payload_type": pt,
                "parameters": parameters,
            }

    # If we reach here, no supported codec was found in the offer
    available = []
    for pt in pt_order:
        c = pt_to_codec.get(pt)
        available.append(f"{pt}:{c or '<no-rtpmap>'}")

    raise ValueError(
        "None of the supported codecs were found in m=video. "
        "PTs seen: " + ", ".join(available)
    )


def _get_mid_from_section(section_text: str) -> Optional[str]:
    """Return the a=mid value from a media section, or None if absent."""
    for line in section_text.splitlines():
        line = line.strip()
        if line.startswith("a=mid:"):
            return line[6:].strip() or None
    return None


def _get_codec_from_media_section(section_text: str, media_type: str) -> CodecEntry:
    """
    Parse a single media section (video or audio) and return the selected
    supported codec entry. Uses SUPPORTED_VIDEO_CODECS or SUPPORTED_AUDIO_CODECS
    according to media_type.
    """
    if not section_text or not isinstance(section_text, str):
        raise ValueError("section_text must be a non-empty string")
    if media_type not in ("video", "audio"):
        raise ValueError("media_type must be 'video' or 'audio'")

    from reactor_runtime.transport.webrtc.gstreamer.settings import (
        SUPPORTED_AUDIO_CODECS,
        SUPPORTED_VIDEO_CODECS,
    )

    priority = (
        SUPPORTED_VIDEO_CODECS if media_type == "video" else SUPPORTED_AUDIO_CODECS
    )
    prio: List[str] = []
    fmtp_overrides: Dict[str, Dict[str, str]] = {}
    for entry in priority:
        name, params = _parse_priority_entry(entry)
        if name:
            prio.append(name)
            fmtp_overrides[name] = params or {}

    m = re.search(rf"(?m)^m={re.escape(media_type)}[^\n]*\n", section_text)
    if not m:
        raise ValueError(f"No m={media_type} line in section")

    start = m.start()
    next_m = re.search(r"(?m)^m=[^\n]*\n", section_text[m.end() :])
    end = (m.end() + next_m.start()) if next_m else len(section_text)
    section = section_text[start:end]

    mline = re.search(rf"(?m)^m={re.escape(media_type)}\s+\d+\s+\S+\s+(.+)$", section)
    if not mline:
        raise ValueError(f"Malformed m={media_type} line in SDP")

    pt_order = [int(t) for t in mline.group(1).strip().split() if t.isdigit()]
    if not pt_order:
        raise ValueError(f"No payload types found in m={media_type} line")

    pt_to_codec: Dict[int, str] = {}
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("a=rtpmap:"):
            continue
        mm = re.match(r"a=rtpmap:(\d+)\s+([A-Za-z0-9\-\._]+)\/\d+(?:\/\d+)?", line)
        if not mm:
            continue
        pt = int(mm.group(1))
        codec = _normalize_encoding_name(mm.group(2))
        pt_to_codec[pt] = codec

    pt_to_fmtp: Dict[int, Dict[str, Optional[str]]] = {}
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("a=fmtp:"):
            continue
        mm = re.match(r"a=fmtp:(\d+)\s+(.+)$", line)
        if not mm:
            continue
        pt = int(mm.group(1))
        params_str = mm.group(2).strip()
        parsed = parse_fmtp_params(params_str)
        existing = pt_to_fmtp.get(pt)
        if existing is None:
            pt_to_fmtp[pt] = parsed
        else:
            existing.update(parsed)

    for wanted in prio:
        for pt in pt_order:
            if pt_to_codec.get(pt) != wanted:
                continue
            sdp_fmtp = pt_to_fmtp.get(pt, {})
            overrides = fmtp_overrides.get(wanted, {})
            if overrides:
                sdp_matches = True
                for k, v in overrides.items():
                    sdp_val = sdp_fmtp.get(k) if sdp_fmtp else None
                    if sdp_val != v:
                        sdp_matches = False
                        break
                if not sdp_matches:
                    continue
            merged_fmtp = {**sdp_fmtp, **overrides}
            parameters: Dict[str, str] = {
                k: v for k, v in merged_fmtp.items() if v is not None
            }
            return {
                "codec": wanted,
                "payload_type": pt,
                "parameters": parameters,
            }

    available = []
    for pt in pt_order:
        c = pt_to_codec.get(pt)
        available.append(f"{pt}:{c or '<no-rtpmap>'}")
    raise ValueError(
        f"None of the supported {media_type} codecs were found in section. "
        "PTs seen: " + ", ".join(available)
    )


def get_codec_from_sdp_by_mid(sdp: str, mid: str) -> CodecEntry:
    """
    Parse an SDP offer and return the selected supported codec for the media
    section with the given ``mid`` (a=mid value).

    Works for both video and audio: the media type is determined from the
    m= line of the section that has the matching a=mid. The first supported
    codec from :data:`reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_VIDEO_CODECS`
    or :data:`reactor_runtime.transport.webrtc.gstreamer.settings.SUPPORTED_AUDIO_CODECS`
    (according to media type) that appears in that section is returned.

    Args:
        sdp: The SDP offer string.
        mid: The a=mid value of the media section (e.g. "0", "1", "video", "audio").

    Returns:
        The selected CodecEntry with codec, payload_type, and parameters.

    Raises:
        ValueError: If sdp/mid are invalid, no section has the given mid,
            the section is not video or audio, or no supported codec is found.
    """
    if not sdp or not isinstance(sdp, str):
        raise ValueError("sdp must be a non-empty string")
    if not mid or not isinstance(mid, str):
        raise ValueError("mid must be a non-empty string")

    _session_part, sections = _split_sdp_into_sections(sdp)
    for media_type, section_text in sections:
        if media_type not in ("video", "audio"):
            continue
        section_mid = _get_mid_from_section(section_text)
        if section_mid != mid:
            continue
        return _get_codec_from_media_section(section_text, media_type)

    raise ValueError(f"No video or audio section with a=mid:{mid} found in SDP")


def get_rtx_payload_type_by_mid(
    sdp: str, mid: str, primary_payload_type: int
) -> Optional[int]:
    """
    Return the RTX payload type in the video section for ``mid`` whose ``apt=``
    references ``primary_payload_type``, or ``None`` if none.
    """
    if not sdp or not isinstance(sdp, str):
        raise ValueError("sdp must be a non-empty string")
    if not mid or not isinstance(mid, str):
        raise ValueError("mid must be a non-empty string")

    _session_part, sections = _split_sdp_into_sections(sdp)
    for media_type, section_text in sections:
        if media_type != "video":
            continue
        if _get_mid_from_section(section_text) != mid:
            continue
        _pt_order, pt_to_codec, pt_to_apt, _pt_to_fmtp = _parse_media_section_pts(
            section_text, "video"
        )
        for pt, codec in pt_to_codec.items():
            if codec != "RTX":
                continue
            if pt_to_apt.get(pt) == primary_payload_type:
                return pt
        return None
    return None


# =========================================================================
# SDP: Filter by supported codecs
# =========================================================================

# Attributes keyed by payload type; when filtering/remapping we update these.
_PT_SCOPED_ATTRS = ("rtpmap", "fmtp", "rtcp-fb")


def _split_sdp_into_sections(sdp: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Split SDP into session-level part and a list of (media_type, section_text).

    media_type is the first token after m= (e.g. "video", "audio", "application").
    """
    session_lines: List[str] = []
    sections: List[Tuple[str, str]] = []
    current_media: Optional[str] = None
    current_lines: List[str] = []

    for line in sdp.splitlines():
        if line.startswith("m="):
            if current_media is not None:
                sections.append((current_media, "\n".join(current_lines)))
            # m=video 9 UDP/TLS/RTP/SAVPF 96 97 ...
            parts = line.split()
            current_media = parts[0][2:]  # strip "m="
            current_lines = [line]
        else:
            if current_media is not None:
                current_lines.append(line)
            else:
                session_lines.append(line)

    if current_media is not None:
        sections.append((current_media, "\n".join(current_lines)))

    session_part = "\n".join(session_lines)
    return session_part, sections


def _parse_media_section_pts(
    section_text: str, media_type: str
) -> Tuple[
    List[int],
    Dict[int, str],
    Dict[int, Optional[int]],
    Dict[int, Dict[str, Optional[str]]],
]:
    """
    Parse a media section: PT order from m= line, pt->codec from rtpmap, pt->apt from fmtp (RTX),
    pt->full fmtp params for parameter matching.

    Returns:
        (pt_order, pt_to_codec, pt_to_apt, pt_to_fmtp)
    """
    pt_order: List[int] = []
    pt_to_codec: Dict[int, str] = {}
    pt_to_apt: Dict[int, Optional[int]] = {}
    pt_to_fmtp: Dict[int, Dict[str, Optional[str]]] = {}

    m_match = re.search(
        rf"^m={re.escape(media_type)}\s+\d+\s+\S+\s+(.+)$",
        section_text,
        flags=re.MULTILINE,
    )
    if m_match:
        for t in m_match.group(1).strip().split():
            if t.isdigit():
                pt_order.append(int(t))

    for line in section_text.splitlines():
        line = line.strip()
        if line.startswith("a=rtpmap:"):
            mm = re.match(r"a=rtpmap:(\d+)\s+([A-Za-z0-9\-\._]+)\/\d+(?:\/\d+)?", line)
            if mm:
                pt = int(mm.group(1))
                codec = _normalize_encoding_name(mm.group(2))
                pt_to_codec[pt] = codec
        elif line.startswith("a=fmtp:"):
            mm = re.match(r"a=fmtp:(\d+)\s+(.+)$", line)
            if mm:
                pt = int(mm.group(1))
                params = parse_fmtp_params(mm.group(2))
                pt_to_fmtp[pt] = params
                apt = params.get("apt")
                if apt is not None and isinstance(apt, str) and apt.isdigit():
                    pt_to_apt[pt] = int(apt)
                else:
                    pt_to_apt[pt] = None

    return pt_order, pt_to_codec, pt_to_apt, pt_to_fmtp


def _rewrite_pt_in_section(
    section_text: str,
    media_type: str,
    kept_pt_order: List[int],
    remap: Dict[int, int],
) -> str:
    """
    Rewrite a media section: keep only lines for kept PTs, apply remap (old_pt -> new_pt).

    Updates m= line payload list, a=rtpmap, a=fmtp, a=rtcp-fb, and apt= in fmtp.
    """
    kept_set = set(kept_pt_order)
    # Build new m= line: PTs in kept_pt_order, replaced by remap
    m_match = re.search(
        rf"^(m={re.escape(media_type)}\s+\d+\s+\S+)\s+(.+)$",
        section_text,
        flags=re.MULTILINE,
    )
    if m_match:
        prefix = m_match.group(1)
        new_pts = [str(remap.get(pt, pt)) for pt in kept_pt_order]
        new_m_line = f"{prefix} " + " ".join(new_pts)
        section_text = re.sub(
            rf"^m={re.escape(media_type)}\s+[^\n]+",
            new_m_line,
            section_text,
            count=1,
            flags=re.MULTILINE,
        )

    # Drop lines for PTs not in kept_set (pt-scoped attributes)
    def keep_line(line: str) -> bool:
        line = line.strip()
        for attr in _PT_SCOPED_ATTRS:
            mm = re.match(rf"^a={re.escape(attr)}:(\d+)\b", line)
            if mm:
                return int(mm.group(1)) in kept_set
        return True

    lines = section_text.splitlines()
    filtered = [line for line in lines if keep_line(line)]

    # Apply remap per line: each line's PT is replaced by remap.get(pt, pt) so chains (96->102, 102->98) don't overwrite
    result_lines: List[str] = []
    for line in filtered:
        out_line = line
        stripped = line.strip()
        for attr in _PT_SCOPED_ATTRS:
            mm = re.match(rf"^a={re.escape(attr)}:(\d+)\b", stripped)
            if mm:
                old_pt = int(mm.group(1))
                new_pt = remap.get(old_pt, old_pt)
                if old_pt != new_pt:
                    out_line = re.sub(
                        rf"^(\s*)a={re.escape(attr)}:{old_pt}\b",
                        rf"\1a={attr}:{new_pt}",
                        out_line,
                    )
                break
        for old_pt, new_pt in remap.items():
            if old_pt != new_pt:
                out_line = re.sub(rf"\bapt={old_pt}\b", f"apt={new_pt}", out_line)
        result_lines.append(out_line)

    return "\n".join(result_lines)


_SupportedEntry = Tuple[str, Dict[str, str]]


def _build_supported_entries(entries: List[CodecEntry]) -> List[_SupportedEntry]:
    """Build ordered list of (codec_name, params) from CodecEntry list.
    When an entry has parameters, only SDP lines with matching fmtp are considered supported.
    """
    result: List[_SupportedEntry] = []
    for entry in entries:
        name, params = _parse_priority_entry(entry)
        if name:
            result.append((name, params))
    return result


def _entry_matches_sdp_line(
    entry_codec: str,
    entry_params: Dict[str, str],
    sdp_codec: Optional[str],
    sdp_fmtp: Dict[str, Optional[str]],
) -> bool:
    """True if SDP line (codec + fmtp) matches this supported entry (codec + params)."""
    if sdp_codec is None or _normalize_encoding_name(sdp_codec) != entry_codec:
        return False
    if not entry_params:
        return True
    for k, v in entry_params.items():
        if sdp_fmtp.get(k) != v:
            return False
    return True


def _filter_media_section_by_supported(
    section_text: str,
    media_type: str,
    pt_order: List[int],
    pt_to_codec: Dict[int, str],
    pt_to_apt: Dict[int, Optional[int]],
    pt_to_fmtp: Dict[int, Dict[str, Optional[str]]],
    supported_entries: List[_SupportedEntry],
) -> str:
    """
    Filter one video or audio section to supported codecs (matching parameters when set).
    Keeps primary codecs matching an entry and their RTX (apt=). Payload types are unchanged.
    """
    kept_primary: Set[int] = set()
    for pt in pt_order:
        codec = pt_to_codec.get(pt)
        sdp_fmtp = pt_to_fmtp.get(pt, {})
        for entry_codec, entry_params in supported_entries:
            if _entry_matches_sdp_line(entry_codec, entry_params, codec, sdp_fmtp):
                kept_primary.add(pt)
                break

    kept_rtx = {
        pt
        for pt in pt_order
        if pt_to_apt.get(pt) is not None and pt_to_apt[pt] in kept_primary
    }
    kept_pts_set = kept_primary | kept_rtx
    kept_pt_order = [pt for pt in pt_order if pt in kept_pts_set]

    if not kept_pt_order:
        raise NoSupportedCodecsError(
            f"No supported codecs in {media_type} media section"
        )

    # No PT remapping; identity remap
    remap = {pt: pt for pt in kept_pt_order}
    return _rewrite_pt_in_section(section_text, media_type, kept_pt_order, remap)


def normalize_sdp_for_supported_codecs(sdp: str) -> str:
    """
    Normalize SDP to supported codecs.

    1. Excludes video codec options not in SUPPORTED_VIDEO_CODECS; when an entry has
       ``parameters``, only SDP lines whose fmtp matches those parameters are kept.
    2. Same for audio and SUPPORTED_AUDIO_CODECS.
    3. If any video or audio media section ends with no codec options, raises
       NoSupportedCodecsError.
    4. Returns the new SDP string (payload types unchanged).

    Raises:
        NoSupportedCodecsError: If a video or audio section has no codecs after filtering.
    """
    if not sdp or not isinstance(sdp, str):
        raise ValueError("sdp must be a non-empty string")

    from reactor_runtime.transport.webrtc.gstreamer.settings import (
        SUPPORTED_AUDIO_CODECS,
        SUPPORTED_VIDEO_CODECS,
    )

    supported_video_entries = _build_supported_entries(SUPPORTED_VIDEO_CODECS)
    supported_audio_entries = _build_supported_entries(SUPPORTED_AUDIO_CODECS)
    supported_by_media: Dict[str, List[_SupportedEntry]] = {
        "video": supported_video_entries,
        "audio": supported_audio_entries,
    }

    session_part, sections = _split_sdp_into_sections(sdp)
    result_sections: List[str] = []

    for media_type, section_text in sections:
        if media_type not in supported_by_media:
            result_sections.append(section_text)
            continue

        pt_order, pt_to_codec, pt_to_apt, pt_to_fmtp = _parse_media_section_pts(
            section_text, media_type
        )
        new_section = _filter_media_section_by_supported(
            section_text,
            media_type,
            pt_order,
            pt_to_codec,
            pt_to_apt,
            pt_to_fmtp,
            supported_by_media[media_type],
        )
        result_sections.append(new_section)

    if not result_sections:
        return sdp

    body = "\n".join(result_sections)
    return (session_part + "\n" + body) if session_part else body
