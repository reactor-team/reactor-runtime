"""WebRTC signalling value types.

The SDP and ICE artefacts exchanged while negotiating a WebRTC connection, plus
the track map a client declares with its offer. These exist only at or below the
acceptor and never appear above it — keeping them here is what lets the runner
stay blind to WebRTC.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from reactor_runtime.core import TrackDirection, TrackInfo, TrackKind

# Client WebRTC direction terms are flipped to the model's perspective: a track
# the client only receives is one the model sends out, and vice versa.
_CLIENT_DIRECTION: dict[str, TrackDirection] = {
    "recvonly": TrackDirection.OUT,
    "sendonly": TrackDirection.IN,
}


@dataclass(frozen=True)
class SdpOffer:
    """The SDP offer a client sends to open or renegotiate a connection."""

    sdp: str
    type: str = "offer"


@dataclass(frozen=True)
class SdpAnswer:
    """The SDP answer the runtime produces in response to an offer.

    Carries the runtime's own ICE candidates inline, so the answer alone is
    enough for the client to begin connecting.
    """

    sdp: str
    type: str = "answer"


@dataclass(frozen=True)
class IceCandidate:
    """One trickle-ICE candidate from the client.

    Attributes:
        candidate: The candidate string (SDP ``a=candidate`` form); an empty
            string is the end-of-candidates marker (RFC 8838).
        sdp_mid: The media-stream identifier the candidate belongs to, or
            ``None`` when addressed by m-line index instead.
        sdp_mline_index: The index of the m-line the candidate belongs to, or
            ``None`` when addressed by mid instead.
    """

    candidate: str
    sdp_mid: str | None = None
    sdp_mline_index: int | None = None


@dataclass(frozen=True)
class MappedTrack:
    """One client-declared track: its SDP media id paired with its metadata."""

    mid: str
    info: TrackInfo


@dataclass(frozen=True)
class TrackMap:
    """The set of tracks a client declares alongside its offer.

    Keyed conceptually by SDP media id (mid), each entry names a track and the
    direction it flows from the model's perspective. The acceptor and the peer
    use it to wire media routing; the connection derives its outbound
    capabilities from the tracks the model sends.
    """

    tracks: tuple[MappedTrack, ...] = ()

    @classmethod
    def from_client(cls, entries: Iterable[Mapping[str, str]]) -> TrackMap:
        """Build a track map from a client ``track_mapping`` payload.

        Each entry carries ``mid``, ``name``, ``kind`` and a client-perspective
        ``direction`` (``recvonly`` / ``sendonly``), which is flipped to the
        model's perspective.

        Raises:
            ValueError: When an entry names a direction other than ``recvonly``
                or ``sendonly``.
        """
        mapped: list[MappedTrack] = []
        for entry in entries:
            direction = _CLIENT_DIRECTION.get(entry["direction"])
            if direction is None:
                raise ValueError(f"Unknown track direction: {entry['direction']!r}")
            mapped.append(
                MappedTrack(
                    mid=str(entry["mid"]),
                    info=TrackInfo(
                        name=entry["name"],
                        kind=TrackKind(entry["kind"]),
                        direction=direction,
                    ),
                )
            )
        return cls(tracks=tuple(mapped))

    def by_direction(self, direction: TrackDirection) -> list[TrackInfo]:
        """Return the metadata of every track flowing in *direction*."""
        return [t.info for t in self.tracks if t.info.direction == direction]
