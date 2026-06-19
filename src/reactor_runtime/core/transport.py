"""Transport-boundary protocols.

The two structural protocols that form the seam between a transport and the
runner, pointing opposite ways. ``Connection`` carries commands down to one
client; ``ConnectionSink`` carries facts up from the transport. Both are
``Protocol``s, so a transport author or a test fake conforms by shape and
signaling never reaches the runner.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from reactor_runtime.core.values import (
    ConnectionCapabilities,
    ConnId,
    InputFrame,
    MediaBundle,
)


@runtime_checkable
class Connection(Protocol):
    """One client connection, regardless of wire — the commands-down half.

    A neutral handle the connection manager holds and the runner sends through.
    Implemented concretely per transport: it moves opaque bytes and media on its
    wire, arbitrates publisher tracks, and owns its own liveness detection. The
    runner only ever holds it as this shape and cannot tell one wire from
    another.

    The object exists before it is live: a transport may build it during its
    handshake (for WebRTC, while negotiating SDP/ICE) and hold it itself. It is
    reported up to the sink only once its wire is actually connected, so a
    half-open connection never reaches the runner.
    """

    id: ConnId
    capabilities: ConnectionCapabilities

    def send_message(self, payload: bytes) -> None:
        """Send already-encoded bytes to this client."""

    def send_media(self, bundle: MediaBundle) -> None:
        """Send a media bundle, or do nothing when the wire carries no media."""

    def resume_track(self, name: str) -> None:
        """Resume the named outbound track (publisher arbitration)."""

    def pause_track(self, name: str) -> None:
        """Pause the named outbound track (publisher arbitration)."""

    async def close(self) -> None:
        """Tear the connection down."""


@runtime_checkable
class ConnectionSink(Protocol):
    """The transport-to-runner upward channel — the facts-up half.

    A transport pushes facts through these as connections come and go and as
    frames arrive; the runner never pulls signaling. Implemented by the runner,
    so several transports can point at the same sink and land in one registry —
    the basis for mixed-transport sessions.
    """

    def connection_opened(self, conn: Connection) -> None:
        """Register a connection whose wire has reached its connected state.

        Fired when the transport is actually connected — for WebRTC, when the
        peer connection reaches its connected state — never at handshake or
        offer time. A transport that builds a connection during signaling holds
        it itself until then, so a client that offers but never connects never
        advances the session.
        """

    def connection_closed(self, conn_id: ConnId) -> None:
        """Drop a previously opened connection that has gone away.

        Reports the loss of a connection that was reported via
        ``connection_opened``. A connection that fails before it ever connects is
        discarded inside the transport and is never seen here.
        """

    def message_received(self, conn_id: ConnId, payload: bytes) -> None:
        """Hand an inbound encoded payload up for decoding and dispatch."""

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        """Hand an inbound media frame up for the named track."""

    def keepalive(self, conn_id: ConnId) -> None:
        """Note liveness for a connection."""
