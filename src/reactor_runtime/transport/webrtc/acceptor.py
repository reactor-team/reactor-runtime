"""The WebRTC acceptor.

Where every WebRTC connection is born and where all of WebRTC's signalling is
concentrated. The acceptor negotiates an offer into a
:class:`~reactor_runtime.transport.webrtc.connection.WebRTCConnection`, wires that
connection's inbound facts straight at the sink, and hands it up only once the
wire is live. SDP and ICE never travel above it, so the runner behind the sink
stays blind to WebRTC.
"""

from __future__ import annotations

from reactor_runtime.core import ConnectionSink, ConnId
from reactor_runtime.transport.acceptor import ConnectionAcceptor
from reactor_runtime.transport.webrtc.config import WebRtcConfig
from reactor_runtime.transport.webrtc.connection import WebRTCConnection
from reactor_runtime.transport.webrtc.peer import WebRtcPeerFactory
from reactor_runtime.transport.webrtc.signaling import IceCandidate, SdpAnswer, SdpOffer, TrackMap


class WebRTCAcceptor(ConnectionAcceptor):
    """Negotiate WebRTC handshakes and register connections once they are live.

    Holds every connection it has negotiated so trickle ICE can keep reaching it
    over the connection's life, and buffers candidates that race ahead of their
    offer. A connection is announced to the sink only on its connected event, so
    an offer that never completes ICE is reaped here and the session never
    advances for it.
    """

    def __init__(
        self,
        *,
        sink: ConnectionSink,
        config: WebRtcConfig,
        peer_factory: WebRtcPeerFactory,
    ) -> None:
        """Bind the acceptor to its sink, config, and peer factory.

        Args:
            sink: The upward channel connections are registered through.
            config: The configuration applied to every negotiated connection.
            peer_factory: Builds the media peer for each offer.
        """
        self._sink = sink
        self._config = config
        self._peer_factory = peer_factory
        self._conns: dict[ConnId, WebRTCConnection] = {}
        self._live: set[ConnId] = set()
        # Candidates that arrived before their connection's offer was negotiated,
        # replayed once it exists. Trickle ICE can race ahead of the answer.
        self._pending_ice: dict[ConnId, list[IceCandidate]] = {}

    async def offer(self, conn_id: ConnId, sdp_offer: SdpOffer, tracks: TrackMap) -> SdpAnswer:
        """Negotiate *sdp_offer* into a connection and return its SDP answer.

        The answer carries the runtime's own ICE candidates and is returned at
        once; the connection is held here and only reaches the sink when its wire
        connects. Any candidates buffered before the offer arrived are replayed.
        """
        previous = self._conns.pop(conn_id, None)
        if previous is not None:
            self._live.discard(conn_id)
            await previous.close()

        conn, answer = await WebRTCConnection.create(
            conn_id, sdp_offer, tracks, self._config, peer_factory=self._peer_factory
        )
        conn.on_message(lambda payload: self._sink.message_received(conn_id, payload))
        conn.on_media(lambda track, frame: self._sink.media_received(conn_id, track, frame))
        conn.on_ping(lambda: self._sink.keepalive(conn_id))
        conn.on_connected(lambda: self._opened(conn_id, conn))
        conn.on_disconnect(lambda: self._closed(conn_id))
        conn.on_closed(lambda: self._forget(conn_id))
        self._conns[conn_id] = conn

        for candidate in self._pending_ice.pop(conn_id, []):
            await conn.add_ice(candidate)
        return answer

    async def add_ice(self, conn_id: ConnId, candidate: IceCandidate) -> None:
        """Forward a trickle-ICE candidate to its connection, buffering if early.

        A candidate for a connection whose offer has not yet been negotiated is
        held and replayed when the offer arrives, rather than dropped.
        """
        conn = self._conns.get(conn_id)
        if conn is None:
            self._pending_ice.setdefault(conn_id, []).append(candidate)
            return
        await conn.add_ice(candidate)

    def _opened(self, conn_id: ConnId, conn: WebRTCConnection) -> None:
        """Announce a connection upward once its wire is live."""
        self._live.add(conn_id)
        self._sink.connection_opened(conn)

    def _closed(self, conn_id: ConnId) -> None:
        """Drop a connection, reporting the loss only if it had opened.

        A connection lost before it ever connected is reaped here and never
        reaches the sink — the session must not advance for an offer that never
        completed.
        """
        self._conns.pop(conn_id, None)
        self._pending_ice.pop(conn_id, None)
        if conn_id in self._live:
            self._live.discard(conn_id)
            self._sink.connection_closed(conn_id)

    def _forget(self, conn_id: ConnId) -> None:
        """Drop a connection torn down on command, without reporting it upward.

        The mirror of :meth:`_closed` for a commanded close (session teardown):
        the connection's owner already drove the close and knows it is gone, so
        the acceptor clears its own bookkeeping but does not notify the sink.
        Without this the acceptor would hold a dead connection for the life of
        the process, since a commanded close is silent.
        """
        self._conns.pop(conn_id, None)
        self._pending_ice.pop(conn_id, None)
        self._live.discard(conn_id)
