"""SDP shaping for the libwebrtc peer.

Two string transforms sit on either side of libwebrtc's own SDP handling:
:func:`deduplicate_bundle_pts` sanitises an inbound offer libwebrtc would
otherwise reject, and :func:`embed_ice_candidates` folds gathered candidates
into the outbound answer so it is complete on its own (non-trickle mode).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# One gathered candidate: its 0-based m-line index (``None`` when the stack did
# not tag one) and the ``candidate`` value without the ``a=`` prefix.
Candidate = tuple[int | None, str]


def embed_ice_candidates(sdp: str, candidates: Sequence[Candidate]) -> str:
    """Inject gathered ICE candidate lines into the answer's m-sections.

    libwebrtc's ``create_answer`` returns SDP without ``a=candidate`` lines; the
    candidates arrive separately via the ICE-candidate callback. Folding them
    into the answer — each at the end of the m-section its m-line index names,
    followed by ``a=end-of-candidates`` — makes the answer self-contained, so a
    client that does not trickle still has every candidate up front.

    Args:
        sdp: The answer SDP as libwebrtc produced it.
        candidates: The gathered candidates to embed.

    Returns:
        The answer SDP with candidate lines embedded, or the input unchanged when
        there are no candidates.
    """
    if not candidates:
        return sdp

    by_mline: dict[int, list[str]] = {}
    for mline_index, candidate in candidates:
        if mline_index is not None:
            by_mline.setdefault(mline_index, []).append(f"a={candidate}\r\n")

    raw = sdp.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    lines = raw.split("\n")
    out: list[str] = []
    mline_idx = -1

    for i, line in enumerate(lines):
        is_last = i == len(lines) - 1
        next_is_msection = not is_last and lines[i + 1].startswith("m=")

        if line.startswith("m="):
            mline_idx += 1

        out.append(line + "\r\n")

        if mline_idx >= 0 and (next_is_msection or is_last):
            cand_lines = by_mline.get(mline_idx, [])
            out.extend(cand_lines)
            if cand_lines:
                out.append("a=end-of-candidates\r\n")

    return "".join(out)


def _bundle_mids(sdp: str) -> set[str]:
    for line in sdp.splitlines():
        match = re.match(r"a=group:BUNDLE\s+(.+)", line)
        if match:
            return set(match.group(1).split())
    return set()


def _mid_of(section: list[str]) -> str | None:
    for line in section:
        match = re.match(r"a=mid:(\S+)", line.strip())
        if match:
            return match.group(1)
    return None


def _rtx_apts(section: list[str]) -> dict[int, int]:
    rtx_pts: set[int] = set()
    apt_map: dict[int, int] = {}
    for line in section:
        stripped = line.strip()
        rtx = re.match(r"a=rtpmap:(\d+)\s+rtx/", stripped, re.IGNORECASE)
        if rtx:
            rtx_pts.add(int(rtx.group(1)))
        apt = re.match(r"a=fmtp:(\d+)\s+apt=(\d+)", stripped)
        if apt:
            apt_map[int(apt.group(1))] = int(apt.group(2))
    return {pt: apt_map[pt] for pt in rtx_pts if pt in apt_map}


def _strip_payload_types(section: list[str], bad: set[int]) -> list[str]:
    if not bad:
        return section
    result: list[str] = []
    for line in section:
        stripped = line.strip()
        media = re.match(r"(m=\S+\s+\d+\s+\S+)((?:\s+\d+)+)", stripped)
        if media:
            kept = [pt for pt in media.group(2).split() if int(pt) not in bad]
            eol = "\r\n" if line.endswith("\r\n") else "\n"
            result.append(media.group(1) + (" " if kept else "") + " ".join(kept) + eol)
            continue
        drop = False
        for attr in ("rtpmap", "fmtp", "rtcp-fb"):
            scoped = re.match(rf"a={attr}:(\d+)\b", stripped)
            if scoped and int(scoped.group(1)) in bad:
                drop = True
                break
        if not drop:
            result.append(line)
    return result


def deduplicate_bundle_pts(sdp: str) -> str:
    """Remove RTX payload-type collisions across BUNDLE m-sections.

    Chrome sometimes assigns the same payload type to RTX codecs with different
    ``apt=`` values in different m-sections of one BUNDLE group, which libwebrtc
    rejects as a codec collision. The first section to claim each RTX payload
    type wins; later sections that disagree have that payload type — and its
    payload-scoped attribute lines — removed. Non-conflicting codecs are left
    untouched.
    """
    bundle_mids = _bundle_mids(sdp)
    if not bundle_mids:
        return sdp

    session_lines: list[str] = []
    sections: list[list[str]] = []
    for line in sdp.splitlines(keepends=True):
        if line.lstrip().startswith("m="):
            sections.append([line])
        elif sections:
            sections[-1].append(line)
        else:
            session_lines.append(line)

    winning_apt: dict[int, int] = {}
    for section in sections:
        if _mid_of(section) not in bundle_mids:
            continue
        for rtx_pt, apt in _rtx_apts(section).items():
            winning_apt.setdefault(rtx_pt, apt)

    rebuilt: list[list[str]] = []
    for section in sections:
        if _mid_of(section) not in bundle_mids:
            rebuilt.append(section)
            continue
        conflicting = {
            rtx_pt for rtx_pt, apt in _rtx_apts(section).items() if winning_apt.get(rtx_pt) != apt
        }
        rebuilt.append(_strip_payload_types(section, conflicting))

    return "".join(session_lines) + "".join("".join(section) for section in rebuilt)
