"""The WebRTC acceptor.

Where every WebRTC connection is born and where all of WebRTC's signalling is
concentrated. The acceptor negotiates an offer into a
:class:`~reactor_runtime.transport.webrtc.connection.WebRTCConnection`, wires that
connection's inbound facts straight at the sink, and hands it up only once the
wire is live. SDP and ICE never travel above it, so the runner behind the sink
stays blind to WebRTC.

Negotiation runs in the background. A client posts its offer, gets an
acknowledgement, then polls for the answer — because producing the answer can
wait on ICE gathering. :meth:`start_offer` kicks the negotiation off and
:meth:`take_answer` drains the result once it is ready. Trickle ICE is the path
that actually completes the connection, so candidates that arrive while the
offer is still negotiating are buffered and replayed the moment the connection
exists.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time

from reactor_runtime.core import ConnectionSink, ConnId
from reactor_runtime.metrics import WebRtcMetrics
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.transport.acceptor import ConnectionAcceptor
from reactor_runtime.transport.router import TooManyConnectionsError
from reactor_runtime.transport.webrtc.config import IceServer, WebRtcConfig
from reactor_runtime.transport.webrtc.connection import WebRTCConnection
from reactor_runtime.transport.webrtc.peer import WebRtcPeerFactory
from reactor_runtime.transport.webrtc.signaling import IceCandidate, SdpAnswer, SdpOffer, TrackMap

logger = logging.getLogger(__name__)

# Bounds on candidates buffered before their offer is negotiated. Trickle ICE
# can race ahead of the answer, but a candidate for a connection that never
# offers would otherwise be held for the session's life. These cap the buffer in
# both directions — distinct pre-offer connections, and candidates per one — so
# a client sending candidates for ids that never negotiate cannot grow it
# without bound. Real ICE gathering yields well under these before the offer
# lands; excess is dropped, and a connection that does negotiate replays what
# was kept.
_MAX_PENDING_ICE_CONNS = 128
_MAX_PENDING_ICE_PER_CONN = 256


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
        metrics: WebRtcMetrics,
    ) -> None:
        """Bind the acceptor to its sink, config, peer factory, and instruments.

        Args:
            sink: The upward channel connections are registered through.
            config: The configuration applied to every negotiated connection.
            peer_factory: Builds the media peer for each offer.
            metrics: Where the handshake timings are recorded.
        """
        self._sink = sink
        self._config = config
        self._peer_factory = peer_factory
        self._metrics = metrics
        self._conns: dict[ConnId, WebRTCConnection] = {}
        self._live: set[ConnId] = set()
        # Candidates that arrived before their connection's offer was negotiated,
        # replayed once it exists. Trickle ICE can race ahead of the answer.
        self._pending_ice: dict[ConnId, list[IceCandidate]] = {}
        # The negotiation in flight for a connection and the answer it leaves
        # behind. Both are keyed by connection id and live only at the acceptor:
        # the SDP answer never travels above the sink.
        self._negotiating: dict[ConnId, asyncio.Task[None]] = {}
        self._answers: dict[ConnId, SdpAnswer] = {}
        # When each offer arrived, which is when its client started waiting. Both
        # handshake measurements run from here, and the entry is dropped when the
        # connection opens or is reaped so a client that never arrives leaves
        # nothing behind.
        self._offered_at: dict[ConnId, float] = {}

    def start_offer(
        self,
        conn_id: ConnId,
        sdp_offer: SdpOffer,
        tracks: TrackMap,
        version: ProtocolVersion,
        ice_servers: tuple[IceServer, ...] | None = None,
    ) -> None:
        """Begin negotiating *sdp_offer* in the background.

        Returns at once; the answer is produced asynchronously (it can wait on
        ICE gathering) and retrieved through :meth:`take_answer`. A fresh offer
        for a connection already negotiating supersedes it: the in-flight
        negotiation is cancelled and any answer it had staged is dropped.
        *version* is the wire codec negotiated for the connection, fixed for its
        life and applied to every frame it carries.

        *ice_servers*, when given, are the STUN/TURN servers this connection
        gathers against, overriding the configured ones for this connection only;
        ``None`` falls back to the acceptor's configuration. Supplied per offer,
        so a reconnect can carry fresh credentials.

        Raises:
            TooManyConnectionsError: If *conn_id* is a new connection and the
                acceptor already holds its configured maximum. A re-offer on a
                connection already negotiating or live is a reconnect and is
                admitted regardless of the ceiling.
        """
        self._guard_capacity(conn_id)
        in_flight = self._negotiating.pop(conn_id, None)
        if in_flight is not None:
            in_flight.cancel()
        self._answers.pop(conn_id, None)
        offered_at = time.monotonic()
        self._offered_at[conn_id] = offered_at
        self._negotiating[conn_id] = asyncio.create_task(
            self._negotiate(conn_id, sdp_offer, tracks, version, ice_servers, offered_at=offered_at)
        )

    def _guard_capacity(self, conn_id: ConnId) -> None:
        """Refuse a new connection once the concurrent ceiling is reached.

        A connection already negotiating or live is known, so its re-offer (a
        reconnect) passes; only a fresh id is measured against the ceiling. A
        ceiling of ``0`` or less is treated as no limit.
        """
        limit = self._config.max_connections
        if limit <= 0:
            return
        known = conn_id in self._conns or conn_id in self._negotiating
        if not known and len(set(self._conns) | set(self._negotiating)) >= limit:
            raise TooManyConnectionsError(limit)

    def take_answer(self, conn_id: ConnId) -> SdpAnswer | None:
        """Return and clear the negotiated answer, or ``None`` while still pending."""
        return self._answers.pop(conn_id, None)

    async def _negotiate(
        self,
        conn_id: ConnId,
        sdp_offer: SdpOffer,
        tracks: TrackMap,
        version: ProtocolVersion,
        ice_servers: tuple[IceServer, ...] | None = None,
        *,
        offered_at: float,
    ) -> None:
        """Negotiate one offer into a connection and stage its answer.

        The connection is held here and only reaches the sink when its wire
        connects. The connection is registered *before* buffered ICE is replayed
        so a candidate arriving during negotiation is either buffered (the
        connection is still absent) or delivered live (it is present), never lost
        in the gap. A negotiation that fails is logged and dropped — the client's
        poll for the answer times out — rather than left as an unhandled task.

        *ice_servers* override the configured STUN/TURN servers for this
        connection when given; ``None`` uses the acceptor's configuration.
        """
        config = (
            self._config
            if ice_servers is None
            else dataclasses.replace(self._config, ice_servers=ice_servers)
        )
        try:
            previous = self._conns.pop(conn_id, None)
            if previous is not None:
                self._live.discard(conn_id)
                await previous.close()

            conn, answer = await WebRTCConnection.create(
                conn_id, sdp_offer, tracks, config, version, peer_factory=self._peer_factory
            )
            conn.on_message(
                lambda payload, ver, ch: self._sink.message_received(conn_id, payload, ver, ch)
            )
            conn.on_media(lambda track, frame: self._sink.media_received(conn_id, track, frame))
            conn.on_ping(lambda: self._sink.keepalive(conn_id))
            conn.on_connected(lambda: self._opened(conn_id, conn))
            conn.on_disconnect(lambda: self._closed(conn_id, offered_at))
            conn.on_closed(lambda: self._forget(conn_id, offered_at))
            self._conns[conn_id] = conn

            for candidate in self._pending_ice.pop(conn_id, []):
                await conn.add_ice(candidate)
            self._answers[conn_id] = answer
            self._metrics.answered(since=offered_at)
            # The answer is both stashed for the client's HTTP poll (take_answer)
            # and reported up as a transport-agnostic fact, so a consumer driving
            # the runtime without polling (a director) can relay it back instead.
            self._sink.connection_answered(conn_id, {"type": answer.type, "sdp": answer.sdp})
        except asyncio.CancelledError:
            # A superseding offer cancelled this one. The client is no longer
            # waiting on it, so it is neither an answer nor a failure.
            raise
        except Exception:
            self._metrics.negotiation_failed(since=offered_at)
            # No connection was built, so nothing else will ever reap this id.
            self._drop_offer(conn_id, offered_at)
            logger.exception("WebRTC negotiation failed for connection %s", conn_id)
        finally:
            # Only clear our own entry: a superseding offer may have already
            # installed a newer task under this id.
            if self._negotiating.get(conn_id) is asyncio.current_task():
                self._negotiating.pop(conn_id, None)

    async def add_ice(self, conn_id: ConnId, candidate: IceCandidate) -> None:
        """Forward a trickle-ICE candidate to its connection, buffering if early.

        A candidate for a connection whose offer has not yet been negotiated is
        held and replayed when the offer arrives, rather than dropped. The buffer
        is bounded per connection and across connections; a candidate past either
        bound is dropped, so candidates for ids that never offer cannot grow it
        without limit.
        """
        conn = self._conns.get(conn_id)
        if conn is None:
            self._buffer_early_ice(conn_id, candidate)
            return
        await conn.add_ice(candidate)

    def _buffer_early_ice(self, conn_id: ConnId, candidate: IceCandidate) -> None:
        """Hold a pre-offer candidate within the buffer's bounds, else drop it."""
        pending = self._pending_ice.get(conn_id)
        if pending is None:
            if len(self._pending_ice) >= _MAX_PENDING_ICE_CONNS:
                return
            pending = self._pending_ice[conn_id] = []
        if len(pending) >= _MAX_PENDING_ICE_PER_CONN:
            return
        pending.append(candidate)

    def _opened(self, conn_id: ConnId, conn: WebRTCConnection) -> None:
        """Announce a connection upward once its wire is live."""
        offered_at = self._offered_at.pop(conn_id, None)
        if offered_at is not None:
            self._metrics.connected(since=offered_at)
        self._live.add(conn_id)
        self._sink.connection_opened(conn)

    def _closed(self, conn_id: ConnId, offered_at: float) -> None:
        """Drop a connection, reporting the loss only if it had opened.

        A connection lost before it ever connected is reaped here and never
        reaches the sink — the session must not advance for an offer that never
        completed.
        """
        self._conns.pop(conn_id, None)
        self._pending_ice.pop(conn_id, None)
        self._answers.pop(conn_id, None)
        self._drop_offer(conn_id, offered_at)
        if conn_id in self._live:
            self._live.discard(conn_id)
            self._sink.connection_closed(conn_id)

    def _forget(self, conn_id: ConnId, offered_at: float) -> None:
        """Drop a connection torn down on command, without reporting it upward.

        The mirror of :meth:`_closed` for a commanded close (session teardown):
        the connection's owner already drove the close and knows it is gone, so
        the acceptor clears its own bookkeeping but does not notify the sink.
        Without this the acceptor would hold a dead connection for the life of
        the process, since a commanded close is silent.
        """
        self._conns.pop(conn_id, None)
        self._pending_ice.pop(conn_id, None)
        self._answers.pop(conn_id, None)
        self._drop_offer(conn_id, offered_at)
        self._live.discard(conn_id)

    def _drop_offer(self, conn_id: ConnId, offered_at: float) -> None:
        """Forget when an offer arrived, unless a newer offer replaced it.

        A reconnect installs its timestamp and then tears the old connection
        down, so the old connection's own teardown runs while the new offer is
        already waiting. Dropping by id alone would take the new offer's
        timestamp with it and leave the reconnect unmeasured.
        """
        if self._offered_at.get(conn_id) == offered_at:
            del self._offered_at[conn_id]
