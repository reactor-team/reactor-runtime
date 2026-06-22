"""The WebRTC peer seam.

The thin boundary between :class:`~reactor_runtime.transport.webrtc.connection.WebRTCConnection`
and the media engine that actually drives the wire. A peer owns one negotiated
peer connection: it moves encoded messages and media, arbitrates which inbound
tracks are live, samples connection stats, and reports inbound facts back through
the callbacks the connection registers on it.

Splitting the engine off behind this protocol is what lets the connection,
acceptor, and router be built and tested without the media stack present: a
concrete peer is supplied by a :data:`WebRtcPeerFactory`, and a fake conforms by
shape.

Threading: a peer invokes the callbacks it is given on the runtime event loop,
so the connection never marshals threads itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from reactor_runtime.core import ConnId, InputFrame, MediaBundle, TrackDirection
from reactor_runtime.transport.webrtc.config import WebRtcConfig
from reactor_runtime.transport.webrtc.signaling import (
    IceCandidate,
    SdpAnswer,
    SdpOffer,
    TrackMap,
)


@dataclass(frozen=True)
class TrackStat:
    """A single track's sampled transport statistics.

    Attributes:
        name: The track name the sample belongs to.
        direction: The track's flow direction, from the model's perspective.
        fps: Frames per second observed, or ``None`` when unavailable.
        bitrate_bps: Bits per second observed, or ``None`` when unavailable.
        packet_loss: Packets lost in the sampling window, or ``None``.
        jitter: Inter-arrival jitter in seconds, or ``None`` when unavailable.
    """

    name: str
    direction: TrackDirection
    fps: float | None = None
    bitrate_bps: int | None = None
    packet_loss: int | None = None
    jitter: float | None = None


@dataclass(frozen=True)
class PeerStats:
    """A snapshot of a peer's transport statistics.

    Attributes:
        rtt_seconds: Round-trip time in seconds, or ``None`` when unavailable.
        tracks: Per-track samples gathered in the same cycle.
    """

    rtt_seconds: float | None = None
    tracks: tuple[TrackStat, ...] = ()


@runtime_checkable
class WebRtcPeer(Protocol):
    """The media engine behind one WebRTC connection.

    Drives a single negotiated peer connection. The connection holds a peer as
    this shape, sends through it, and registers the inbound callbacks the peer
    invokes as facts arrive on the wire.
    """

    async def add_ice(self, candidate: IceCandidate) -> None:
        """Add a trickle-ICE candidate; valid before and after the wire connects."""

    def send_message(self, payload: bytes | str) -> None:
        """Send an already-encoded frame over the data channel (text or binary)."""

    def send_media(self, bundle: MediaBundle) -> None:
        """Send a media bundle, routing each track to its negotiated sender."""

    def resume_track(self, name: str) -> None:
        """Resume the named track (publisher arbitration)."""

    def pause_track(self, name: str) -> None:
        """Pause the named track (publisher arbitration)."""

    async def stats(self) -> PeerStats:
        """Sample current transport statistics."""

    async def close(self) -> None:
        """Tear the peer connection down."""

    def on_message(self, callback: Callable[[bytes | str], None]) -> None:
        """Register the sink for inbound data-channel frames (text or binary)."""

    def on_media(self, callback: Callable[[str, InputFrame], None]) -> None:
        """Register the sink for inbound media frames, by track name."""

    def on_ping(self, callback: Callable[[], None]) -> None:
        """Register the sink for client liveness pings."""

    def on_connected(self, callback: Callable[[], None]) -> None:
        """Register the sink for the wire reaching its connected state."""

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register the sink for the wire being lost.

        A peer fires this only after releasing its own wire, so the connection
        does not close the peer again on this path.
        """


WebRtcPeerFactory = Callable[
    [ConnId, SdpOffer, TrackMap, WebRtcConfig],
    Awaitable[tuple[WebRtcPeer, SdpAnswer]],
]
"""Build a negotiated peer for *(conn id, offer, tracks, config)*.

Returns the peer and the SDP answer produced during the exchange. The media
stack supplies the concrete factory; until it lands, the acceptor is constructed
with whichever factory the caller injects.
"""
