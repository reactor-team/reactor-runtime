from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IceCandidate:
    """
    Represents a single ICE candidate extracted from SDP.

    Attributes:
        mline_index:
            Zero-based index of the m= section where this candidate belongs.
            Required by many WebRTC stacks (including webrtcbin) when
            adding candidates via add-ice-candidate.

        mid:
            The media identifier (a=mid:<X>) for that m-line, if present.
            Modern WebRTC stacks prefer MID over m-line index.

        candidate:
            Full candidate attribute value WITHOUT the leading 'a='.
            Example:
                "candidate:1234 1 udp 2122260223 192.168.0.2 54321 typ host"
    """

    mline_index: int
    mid: Optional[str]  # from a=mid:<X>
    candidate: str  # full candidate attribute value, i.e. "candidate:...."


# =========================================================================
# SDP: strip ICE candidates from SDP to avoid webrtcbin race condition
# =========================================================================
def strip_ice_candidates_from_sdp(sdp: str) -> Tuple[str, List[IceCandidate]]:
    """
    Remove all:
        - a=candidate:<...>
        - a=end-of-candidates

    from the SDP.

    Keeps:
        - ice-ufrag
        - ice-pwd
        - All other attributes untouched

    Why this exists:
        Some versions of webrtcbin can exhibit race conditions if
        ICE candidates are embedded in the SDP AND also added via
        add-ice-candidate signal.

        Strategy:
            1. Strip candidates from SDP before setting remote description.
            2. Later inject candidates explicitly via API.

    Returns:
        (sanitized_sdp, candidates)

    Where:
        sanitized_sdp:
            SDP string without ICE candidate lines.

        candidates:
            List of IceCandidate objects extracted from SDP.
    """

    # Normalize line endings to '\n' for consistent parsing.
    # SDP spec mandates CRLF, but real-world SDPs may mix endings.
    s = sdp.replace("\r\n", "\n").replace("\r", "\n")

    # Drop fully empty lines.
    # Internal empty lines are extremely rare in SDP.
    lines = [ln.rstrip() for ln in s.split("\n") if ln != ""]

    out: List[str] = []  # Reconstructed SDP lines (without candidates)
    cands: List[IceCandidate] = []  # Extracted ICE candidates

    # mline_index tracks which m= section we are currently in.
    # Starts at -1 so first m= increments to 0.
    mline_index = -1

    # current_mid tracks the a=mid:<X> associated with current m-line.
    current_mid: Optional[str] = None

    def flush_line(ln: str) -> None:
        """
        Append line to reconstructed SDP output.
        Isolated in a function to make control flow clearer.
        """
        out.append(ln)

    for ln in lines:
        # lstrip to tolerate indentation or formatting differences.
        line = ln.lstrip()

        # ---------------------------------------------------------
        # New media section
        # ---------------------------------------------------------
        if line.startswith("m="):
            mline_index += 1
            current_mid = None  # reset mid for new section
            flush_line(ln)
            continue

        # ---------------------------------------------------------
        # MID attribute (maps media section to logical id)
        # ---------------------------------------------------------
        if line.startswith("a=mid:"):
            # MID is defined per m-line
            current_mid = line[len("a=mid:") :].strip()
            flush_line(ln)
            continue

        # ---------------------------------------------------------
        # ICE candidate line
        # ---------------------------------------------------------
        if line.startswith("a=candidate:"):
            # Extract candidate without the leading "a="
            # Example stored:
            #   "candidate:1 1 udp 2122260223 192.168.0.2 54321 typ host"
            cands.append(
                IceCandidate(
                    # Defensive: ensure index is never negative
                    mline_index=max(mline_index, 0),
                    mid=current_mid,
                    candidate=line[len("a=") :].strip(),
                )
            )
            # Do NOT add this line to output SDP
            continue

        # ---------------------------------------------------------
        # End-of-candidates marker
        # ---------------------------------------------------------
        if line.startswith("a=end-of-candidates"):
            # This line signals ICE gathering complete.
            # We remove it because we manage candidates externally.
            continue

        # ---------------------------------------------------------
        # Any other line → preserve as-is
        # ---------------------------------------------------------
        flush_line(ln)

    # Reassemble SDP using CRLF as required by RFC 4566.
    sanitized = "\r\n".join(out) + "\r\n"

    return sanitized, cands
