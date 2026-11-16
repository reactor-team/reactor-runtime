"""SDP shaping for the libwebrtc peer.

Two string transforms sit on either side of libwebrtc's own SDP handling:
:func:`deduplicate_bundle_pts` sanitises an inbound offer libwebrtc would
otherwise reject, and :func:`embed_ice_candidates` folds gathered candidates
into the outbound answer so it is complete on its own (non-trickle mode).

The rest reads and writes what libwebrtc leaves to the application: a
session-level attribute of the runtime's own (:func:`has_session_attribute`,
:func:`add_session_attribute`) and the codecs one m-section lists, in its
order of preference (:func:`section_codecs`).
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


_DIRECTIONS = ("sendrecv", "sendonly", "recvonly", "inactive")
_MID_PREFIX = "a=mid:"


def set_media_direction(sdp: str, mid: str, direction: str) -> str:
    """Rewrite the direction of the m-section carrying *mid*.

    Stopping one track's sender is a local renegotiation: the same descriptions
    go back to libwebrtc with that section's direction flipped, and the engine
    starts or stops the stream behind it. Nothing is signalled — the client's
    view of the session does not change, and neither does anything else in the
    SDP, so re-applying it is not an ICE restart or a codec change.

    A section with no direction line is ``sendrecv`` by default; one is added
    after its mid so the flip has something to act on.

    Args:
        sdp: The description to rewrite.
        mid: The media id naming the section to change.
        direction: One of ``sendrecv``, ``sendonly``, ``recvonly``, ``inactive``.

    Returns:
        The description with that one section's direction replaced.

    Raises:
        ValueError: If *direction* is not an SDP direction.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"not an SDP direction: {direction!r}")

    lines = sdp.split("\r\n")
    bounds = [i for i, line in enumerate(lines) if line.startswith("m=")]
    for start, end in zip(bounds, [*bounds[1:], len(lines)], strict=True):
        section = lines[start:end]
        if f"{_MID_PREFIX}{mid}" not in section:
            continue
        for offset, line in enumerate(section):
            if line.removeprefix("a=") in _DIRECTIONS:
                lines[start + offset] = f"a={direction}"
                break
        else:
            after_mid = next(i for i, line in enumerate(section) if line.startswith(_MID_PREFIX))
            lines.insert(start + after_mid + 1, f"a={direction}")
        break
    return "\r\n".join(lines)


def bump_session_version(sdp: str, by: int) -> str:
    """Advance the session version in the ``o=`` line by *by*.

    RFC 3264 §8 requires the version to increase on each offer or answer within
    a session, so a description re-applied with the version it already carried
    describes no change and is free to be treated as the duplicate it looks
    like. A local renegotiation has to advance it for the same reason a
    signalled one does.

    The origin line is ``o=<user> <sess-id> <sess-version> <nettype> <addrtype>
    <address>``, so only the third field moves.

    Args:
        sdp: The description to rewrite.
        by: How far past the negotiated version to place this one.

    Returns:
        The description with its session version advanced, or unchanged when it
        has no parsable origin line.
    """
    lines = sdp.split("\r\n")
    for index, line in enumerate(lines):
        if not line.startswith("o="):
            continue
        fields = line.split(" ")
        if len(fields) < 3 or not fields[2].isdigit():
            break
        fields[2] = str(int(fields[2]) + by)
        lines[index] = " ".join(fields)
        break
    return "\r\n".join(lines)


ENCODED_STEP_FRAMES = "x-reactor-encoded-step-frames"
"""The session-level attribute that negotiates encoded step frames.

A stepped client that sends and receives encoded step frames puts it in its
offer, and a runtime that encodes them repeats it in its answer.
"""


def _session_lines(sdp: str) -> list[str]:
    """Return the lines before the first m-section, without line endings."""
    session: list[str] = []
    for line in sdp.splitlines():
        if line.startswith("m="):
            break
        session.append(line)
    return session


def has_session_attribute(sdp: str, name: str) -> bool:
    """Return whether *sdp* declares the attribute *name* at session level.

    An attribute inside an m-section does not count. A value after the name
    (``a=<name>:<value>``) is allowed and ignored.
    """
    return any(
        line.strip() == f"a={name}" or line.startswith(f"a={name}:") for line in _session_lines(sdp)
    )


def add_session_attribute(sdp: str, name: str) -> str:
    """Declare the attribute *name* at session level, just before the first m-section.

    Returns *sdp* unchanged when it already declares it. A description with no
    m-section gets the attribute at its end.
    """
    if has_session_attribute(sdp, name):
        return sdp
    eol = "\r\n" if "\r\n" in sdp else "\n"
    lines = sdp.split(eol)
    index = next((i for i, line in enumerate(lines) if line.startswith("m=")), None)
    if index is None:
        body = sdp.rstrip(eol)
        return f"{body}{eol}a={name}{eol}"
    lines.insert(index, f"a={name}")
    return eol.join(lines)


def section_codecs(sdp: str, mid: str) -> list[str]:
    """Return the codecs the m-section carrying *mid* lists, most preferred first.

    Each codec is its encoding name from ``a=rtpmap``, upper-cased (``H264``,
    ``VP8``, ``RTX``), in the order of the m-line's payload types, once each:
    a codec listed under several payload types (H.264 in several profiles)
    keeps its first place. In an offer this is the offerer's preference; in an
    answer, the first codec is the one the section negotiated.

    Returns an empty list when no section carries *mid*.
    """
    sections: list[list[str]] = []
    for line in sdp.splitlines():
        stripped = line.strip()
        if stripped.startswith("m="):
            sections.append([stripped])
        elif sections:
            sections[-1].append(stripped)
    for section in sections:
        if f"{_MID_PREFIX}{mid}" not in section:
            continue
        names: dict[str, str] = {}
        for line in section:
            rtpmap = re.match(r"a=rtpmap:(\d+)\s+([^/\s]+)", line)
            if rtpmap:
                names.setdefault(rtpmap.group(1), rtpmap.group(2).upper())
        payload_types = section[0].split()[3:]
        ordered: list[str] = []
        for pt in payload_types:
            name = names.get(pt)
            if name is not None and name not in ordered:
                ordered.append(name)
        return ordered
    return []
