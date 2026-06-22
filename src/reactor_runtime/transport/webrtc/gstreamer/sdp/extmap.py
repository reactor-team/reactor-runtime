
"""
SDP RTP header extension (extmap) helpers.

Parse extension IDs per ``a=mid`` from offers and inject ``a=extmap`` lines into
answers per media section so negotiated IDs match the peer (RFC 5285 / WebRTC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from reactor_runtime.transport.webrtc.gstreamer.gst import GstSdp, GstWebRTC

from .bundle import _get_attr_value, _split_sdp_sections


@dataclass(frozen=True)
class SdpExtmap:
    """One RTP header extension mapping (RFC 5285 / WebRTC)."""

    id: int
    """Extension ID (1–14 for one-byte header, or as negotiated)."""

    uri: str
    """Extension URI (e.g. transport-wide CC draft URI)."""


def _normalize_extmaps(
    extmaps: Sequence[Union[SdpExtmap, Tuple[int, str]]],
) -> List[SdpExtmap]:
    out: List[SdpExtmap] = []
    for item in extmaps:
        if isinstance(item, SdpExtmap):
            out.append(item)
        else:
            eid, uri = item
            out.append(SdpExtmap(id=int(eid), uri=str(uri)))
    return out


def _media_type_from_mline(mline: str) -> Optional[str]:
    if not mline.startswith("m="):
        return None
    parts = mline.split()
    if len(parts) < 1:
        return None
    return parts[0][2:].lower()


def _existing_extmap_ids_in_section(section: List[str]) -> set[int]:
    """Collect numeric extension IDs already present in a media section."""
    ids: set[int] = set()
    for ln in section:
        eid = _parse_extmap_id_from_line(ln.strip())
        if eid is not None:
            ids.add(eid)
    return ids


def _parse_extmap_id_from_line(line: str) -> Optional[int]:
    """
    Parse extension id from ``a=extmap:<id>`` or ``a=extmap:<id>/dir``.

    Returns None if the line is not an extmap or id is not an integer.
    (Does not require a URI token; used to detect occupied ids in a section.)
    """
    s = line.strip()
    if not s.startswith("a=extmap:"):
        return None
    rest = s[len("a=extmap:") :].strip()
    if not rest:
        return None
    id_token = rest.split()[0]
    if "/" in id_token:
        id_token = id_token.split("/")[0]
    try:
        return int(id_token)
    except ValueError:
        return None


def _parse_extmap_id_and_uri_from_line(line: str) -> Optional[Tuple[int, str]]:
    """
    Parse ``a=extmap:<id>`` or ``a=extmap:<id>/<direction>`` and the extension URI.

    Returns None if the line is not a valid one-byte extmap with a URI token.
    """
    s = line.strip()
    if not s.startswith("a=extmap:"):
        return None
    body = s[len("a=extmap:") :].strip()
    if not body:
        return None
    parts = body.split(None, 1)
    if len(parts) < 2:
        return None
    id_token, uri_tail = parts[0], parts[1]
    if "/" in id_token:
        id_token = id_token.split("/")[0]
    try:
        eid = int(id_token)
    except ValueError:
        return None
    # URI is first token; remainder may be extension-specific attributes (RFC 5285).
    uri = uri_tail.split()[0].strip()
    if not uri:
        return None
    return (eid, uri)


def _extmap_id_for_uri_in_lines(lines: Sequence[str], uri: str) -> Optional[int]:
    """First extmap id in ``lines`` whose URI equals ``uri`` (exact match)."""
    for ln in lines:
        parsed = _parse_extmap_id_and_uri_from_line(ln.strip())
        if parsed and parsed[1] == uri:
            return parsed[0]
    return None


def extmap_id_by_mid_for_uri(sdp: str, extension_uri: str) -> Dict[str, Optional[int]]:
    """
    For each media section with ``a=mid:``, return the negotiated extmap id for
    ``extension_uri`` in that section.

    The id is taken from ``a=extmap`` lines in that m-section only. If the URI
    does not appear there, the value for that mid is ``None``. Session-level
    extmaps are not applied (per-m-section ids may differ).

    Args:
        sdp: SDP text (LF or CRLF).
        extension_uri: Full extension URI string (e.g. TWCC draft URI).

    Returns:
        Mapping ``mid -> extension_id_or_none`` for every media section that
        declares ``a=mid:``. Sections without ``a=mid`` are skipped.
    """
    _, media_sections = _split_sdp_sections(sdp)
    out: Dict[str, Optional[int]] = {}
    for sec in media_sections:
        mid = _get_attr_value(sec, "mid")
        if not mid:
            continue
        eid = _extmap_id_for_uri_in_lines(sec, extension_uri)
        out[mid] = eid
    return out


def negotiated_sdp_extmaps_by_mid(
    sdp: str,
    extension_uris: Sequence[str],
) -> Dict[str, List[SdpExtmap]]:
    """
    For each media section with ``a=mid:``, build :class:`SdpExtmap` entries for URIs
    in ``extension_uris`` that appear in that section.

    URIs are processed in order (duplicates in ``extension_uris`` are ignored). Mids
    with no matching extensions are omitted from the result.
    """
    _, media_sections = _split_sdp_sections(sdp)
    unique_uris = tuple(dict.fromkeys(extension_uris))
    out: Dict[str, List[SdpExtmap]] = {}
    for sec in media_sections:
        mid = _get_attr_value(sec, "mid")
        if not mid:
            continue
        found: List[SdpExtmap] = []
        for uri in unique_uris:
            eid = _extmap_id_for_uri_in_lines(sec, uri)
            if eid is not None:
                found.append(SdpExtmap(id=eid, uri=uri))
        if found:
            out[mid] = found
    return out


def _insert_extmap_lines(
    section: List[str],
    extmaps: Sequence[SdpExtmap],
) -> List[str]:
    """
    Insert new ``a=extmap`` lines after ``a=mid:`` if present, else right
    after the ``m=`` line. Skips any ``(id, uri)`` whose id already exists
    in the section.
    """
    lines = list(section)
    existing = _existing_extmap_ids_in_section(lines)
    to_add: List[str] = []
    for em in extmaps:
        if em.id in existing:
            continue
        if em.id <= 0:
            raise ValueError(f"extmap id must be > 0, got {em.id}")
        if not em.uri.strip():
            raise ValueError("extmap uri must be non-empty")
        to_add.append(f"a=extmap:{em.id} {em.uri.strip()}")

    if not to_add:
        return lines

    insert_at = 1
    for i, ln in enumerate(lines):
        if ln.startswith("a=mid:"):
            insert_at = i + 1
            break

    return lines[:insert_at] + to_add + lines[insert_at:]


def add_extmaps_per_mid_to_sdp_text(
    sdp: str,
    extmaps_by_mid: Mapping[str, Sequence[Union[SdpExtmap, Tuple[int, str]]]],
) -> str:
    """
    Insert ``a=extmap`` lines only into media sections whose ``a=mid`` appears
    in ``extmaps_by_mid``, operating on raw SDP text.

    Each mid's list is normalized with the same deduplication rules as
    :func:`add_extmaps_to_sdp_media` (skip ids already present in that section).

    Args:
        sdp: Full SDP document (LF or CRLF).
        extmaps_by_mid: ``mid`` -> sequence of :class:`SdpExtmap` or ``(id, uri)``.

    Returns:
        SDP string with CRLF line endings (RFC 4566 style).
    """
    session_lines, media_sections = _split_sdp_sections(sdp)
    new_sections: List[List[str]] = []
    for sec in media_sections:
        mid = _get_attr_value(sec, "mid")
        if mid and mid in extmaps_by_mid:
            normalized = _normalize_extmaps(extmaps_by_mid[mid])
            new_sections.append(_insert_extmap_lines(sec, normalized))
        else:
            new_sections.append(list(sec))

    out_lines = list(session_lines)
    for sec in new_sections:
        out_lines.extend(sec)

    return "\r\n".join(out_lines) + "\r\n"


def add_extmaps_per_mid_to_sdp(
    desc: GstWebRTC.WebRTCSessionDescription,
    extmaps_by_mid: Mapping[str, Sequence[Union[SdpExtmap, Tuple[int, str]]]],
) -> GstWebRTC.WebRTCSessionDescription:
    """
    Insert ``a=extmap`` lines only into media sections whose ``a=mid`` appears
    in ``extmaps_by_mid``, returning a new :class:`GstWebRTC.WebRTCSessionDescription`.

    ``desc`` is not modified; a new Gst object is allocated. The SDP text is
    updated via :func:`add_extmaps_per_mid_to_sdp_text` and re-parsed.

    Args:
        desc: Session description whose SDP is to be updated.
        extmaps_by_mid: ``mid`` -> sequence of :class:`SdpExtmap` or ``(id, uri)``.

    Returns:
        New session description with the same ``WebRTCSDPType`` as ``desc``.
    """
    new_text = add_extmaps_per_mid_to_sdp_text(desc.sdp.as_text(), extmaps_by_mid)

    res, sdpmsg = GstSdp.SDPMessage.new()
    parse_ret = GstSdp.sdp_message_parse_buffer(bytes(new_text, "utf-8"), sdpmsg)
    if parse_ret != GstSdp.SDPResult.OK:
        raise ValueError(
            f"Failed to parse SDP after per-mid extmap injection: {parse_ret!r}"
        )

    return GstWebRTC.WebRTCSessionDescription.new(desc.type, sdpmsg)


def add_extmaps_to_sdp_media(
    sdp: str,
    extmaps: Sequence[Union[SdpExtmap, Tuple[int, str]]],
    *,
    media_types: Optional[Sequence[str]] = None,
) -> str:
    """
    Append a list of ``a=extmap`` attributes to each matching media section.

    For each ``m=`` section whose type is included in ``media_types`` (default:
    ``audio`` and ``video``), inserts lines of the form::

        a=extmap:<id> <uri>

    immediately after ``a=mid:`` if present, otherwise after the ``m=`` line.
    If an extension id is already declared in that section, it is not
    duplicated.

    Args:
        sdp: Full SDP document (LF or CRLF line endings accepted).
        extmaps: Extension mappings as :class:`SdpExtmap` or ``(id, uri)`` tuples.
        media_types: Which ``m=`` kinds receive the extmaps (e.g. ``("video",)``).
            Default is ``("audio", "video")``. Use a broader tuple to include
            ``application`` if needed.

    Returns:
        SDP string with CRLF line endings (RFC 4566 style).

    Raises:
        ValueError: If any extmap has invalid id or empty uri.
    """
    normalized = _normalize_extmaps(extmaps)
    kinds = (
        tuple(m.lower() for m in media_types)
        if media_types is not None
        else ("audio", "video")
    )

    session_lines, media_sections = _split_sdp_sections(sdp)
    new_sections: List[List[str]] = []
    for sec in media_sections:
        if not sec:
            new_sections.append(sec)
            continue
        mtype = _media_type_from_mline(sec[0])
        if mtype is not None and mtype in kinds:
            new_sections.append(_insert_extmap_lines(sec, normalized))
        else:
            new_sections.append(sec)

    out_lines = list(session_lines)
    for sec in new_sections:
        out_lines.extend(sec)

    return "\r\n".join(out_lines) + "\r\n"


def add_extmaps_to_webrtc_session_description(
    desc: GstWebRTC.WebRTCSessionDescription,
    extmaps: Sequence[Union[SdpExtmap, Tuple[int, str]]],
    *,
    media_types: Optional[Sequence[str]] = None,
) -> GstWebRTC.WebRTCSessionDescription:
    """
    Return a new :class:`GstWebRTC.WebRTCSessionDescription` with the same
    ``WebRTCSDPType`` as ``desc`` but SDP text updated by
    :func:`add_extmaps_to_sdp_media`.

    Use this when setting a local/remote description after mutating SDP (e.g.
    inject TWCC ``a=extmap`` lines to match the RTP capsfilter).

    ``desc`` is not modified; a new Gst object is allocated.
    """
    sdp_text = desc.sdp.as_text()
    new_text = add_extmaps_to_sdp_media(sdp_text, extmaps, media_types=media_types)

    res, sdpmsg = GstSdp.SDPMessage.new()
    parse_ret = GstSdp.sdp_message_parse_buffer(bytes(new_text, "utf-8"), sdpmsg)
    if parse_ret != GstSdp.SDPResult.OK:
        raise ValueError(f"Failed to parse SDP after extmap injection: {parse_ret!r}")

    # WebRTCSessionDescription is a boxed type: use .type / .sdp, not GObject.get_property.
    sdp_type = desc.type
    return GstWebRTC.WebRTCSessionDescription.new(sdp_type, sdpmsg)
