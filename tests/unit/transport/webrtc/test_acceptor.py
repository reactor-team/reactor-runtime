import asyncio
from collections.abc import Callable, Mapping

import numpy as np
import pytest
from conftest import FakePeer

from reactor_runtime.core import Connection, ConnId, InputFrame
from reactor_runtime.metrics import RuntimeMetrics, WebRtcMetrics
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport import TooManyConnectionsError
from reactor_runtime.transport.webrtc import (
    SdpAnswer,
    SdpOffer,
    TrackMap,
    WebRTCAcceptor,
    WebRtcConfig,
    WebRTCPeer,
    WebRtcPeerFactory,
)
from reactor_runtime.transport.webrtc.acceptor import (
    _MAX_PENDING_ICE_CONNS,
    _MAX_PENDING_ICE_PER_CONN,
)
from reactor_runtime.transport.webrtc.config import IceServer
from reactor_runtime.transport.webrtc.signaling import IceCandidate


class FakeSink:
    """A ConnectionSink that records every fact pushed up to it."""

    def __init__(self) -> None:
        self.opened: list[ConnId] = []
        self.closed: list[ConnId] = []
        self.messages: list[tuple[ConnId, bytes | str]] = []
        self.media: list[tuple[ConnId, str]] = []
        self.keepalives: list[ConnId] = []
        self.answered: list[tuple[ConnId, dict[str, str]]] = []

    def connection_opened(self, conn: Connection) -> None:
        self.opened.append(conn.id)

    def connection_closed(self, conn_id: ConnId) -> None:
        self.closed.append(conn_id)

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        self.messages.append((conn_id, payload))

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        self.media.append((conn_id, track))

    def keepalive(self, conn_id: ConnId) -> None:
        self.keepalives.append(conn_id)

    def resume_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def pause_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def publish_requested(self, conn_id: ConnId, name: str, request_id: str) -> None:
        pass

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        pass

    def schema_requested(self, conn_id: ConnId, request_id: str) -> None:
        pass

    def clip_requested(self, conn_id: ConnId, duration_seconds: float, request_id: str) -> None:
        pass

    def recording_requested(self, conn_id: ConnId, request_id: str) -> None:
        pass

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        self.answered.append((conn_id, dict(answer)))


def _metrics() -> RuntimeMetrics:
    return RuntimeMetrics(version="0.0.0", model="fake:Model")


def _acceptor(
    sink: FakeSink,
    peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    metrics: RuntimeMetrics | None = None,
) -> WebRTCAcceptor:
    return WebRTCAcceptor(
        sink=sink,
        config=WebRtcConfig(ping_timeout=0.0, negotiation_timeout=0.0),
        peer_factory=factory_for(peer),
        metrics=WebRtcMetrics(metrics or _metrics()),
    )


async def _negotiate(
    acceptor: WebRTCAcceptor,
    conn_id: ConnId,
    offer: SdpOffer,
    tracks: TrackMap,
    version: ProtocolVersion = ProtocolVersion.V0,
) -> SdpAnswer | None:
    """Start an offer and drive its background negotiation to completion.

    ``start_offer`` is synchronous: it queues the negotiation task without
    yielding, so the task can be captured from ``_negotiating`` and awaited
    before draining the answer.
    """
    acceptor.start_offer(conn_id, offer, tracks, version)
    await acceptor._negotiating[conn_id]
    return acceptor.take_answer(conn_id)


async def test_negotiation_produces_the_answer_without_opening(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    answer = await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    assert answer is not None
    assert answer.sdp == "answer-sdp"
    assert sink.opened == []
    # The answer is reported up as a transport-agnostic fact before the wire
    # connects, alongside being stashed for the HTTP poll.
    assert sink.answered == [(ConnId(7), {"type": "answer", "sdp": "answer-sdp"})]


async def test_offer_ice_servers_override_the_configured_ones(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    servers = (
        IceServer(urls=("turn:turn.example:3478",), username="u", credential="p"),
        IceServer(urls=("stun:stun.example:3478",)),
    )
    acceptor.start_offer(ConnId(7), SdpOffer("offer"), out_av_tracks, ProtocolVersion.V0, servers)
    await acceptor._negotiating[ConnId(7)]

    assert fake_peer.last_config is not None
    assert fake_peer.last_config.ice_servers == servers


async def test_offer_without_ice_servers_keeps_the_configured_ones(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    acceptor.start_offer(ConnId(7), SdpOffer("offer"), out_av_tracks, ProtocolVersion.V0)
    await acceptor._negotiating[ConnId(7)]

    assert fake_peer.last_config is not None
    # The acceptor's configured servers (empty by default) are used untouched.
    assert fake_peer.last_config.ice_servers == ()


async def test_take_answer_is_none_until_negotiation_completes(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    acceptor.start_offer(ConnId(7), SdpOffer("offer"), out_av_tracks, ProtocolVersion.V0)
    # The negotiation task has been queued but not yet run.
    assert acceptor.take_answer(ConnId(7)) is None
    await acceptor._negotiating[ConnId(7)]
    answer = acceptor.take_answer(ConnId(7))
    assert answer is not None
    assert answer.sdp == "answer-sdp"
    # Draining is one-shot.
    assert acceptor.take_answer(ConnId(7)) is None


async def test_connection_opened_only_on_connect(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    assert sink.opened == []
    fake_peer.fire_connected()
    assert sink.opened == [ConnId(7)]
    fake_peer.fire_disconnect()


async def test_offer_that_never_connects_never_reaches_sink(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    fake_peer.fire_disconnect()
    assert sink.opened == []
    assert sink.closed == []


async def test_sink_callbacks_are_wired(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    fake_peer.fire_connected()

    fake_peer.fire_message(b"hello")
    fake_peer.fire_media("webcam", InputFrame(np.zeros((1, 1, 3), dtype=np.uint8)))
    fake_peer.fire_ping()

    assert sink.messages == [(ConnId(7), b"hello")]
    assert sink.media == [(ConnId(7), "webcam")]
    assert sink.keepalives == [ConnId(7)]
    fake_peer.fire_disconnect()


async def test_disconnect_after_open_reports_closed(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    fake_peer.fire_connected()
    fake_peer.fire_disconnect()
    assert sink.closed == [ConnId(7)]


async def test_commanded_close_forgets_without_reporting(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    fake_peer.fire_connected()

    await acceptor._conns[ConnId(7)].close()

    assert ConnId(7) not in acceptor._conns
    assert ConnId(7) not in acceptor._live
    assert sink.closed == []


async def test_add_ice_reaches_connection(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    await acceptor.add_ice(ConnId(7), IceCandidate("cand"))
    assert fake_peer.ice == [IceCandidate("cand")]


async def test_ice_before_offer_is_buffered_then_flushed(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    await acceptor.add_ice(ConnId(7), IceCandidate("early"))
    assert fake_peer.ice == []
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    assert fake_peer.ice == [IceCandidate("early")]


async def test_re_offer_supersedes_a_pending_negotiation(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    acceptor.start_offer(ConnId(7), SdpOffer("first"), out_av_tracks, ProtocolVersion.V0)
    first = acceptor._negotiating[ConnId(7)]
    acceptor.start_offer(ConnId(7), SdpOffer("second"), out_av_tracks, ProtocolVersion.V0)
    second = acceptor._negotiating[ConnId(7)]
    assert second is not first
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    answer = acceptor.take_answer(ConnId(7))
    assert answer is not None


def _sample(metrics: RuntimeMetrics, name: str, **labels: str) -> float | None:
    return metrics.registry.get_sample_value(name, labels or None)


async def test_an_answered_offer_is_measured(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    metrics = _metrics()
    acceptor = _acceptor(FakeSink(), fake_peer, factory_for, metrics)

    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)

    assert _sample(metrics, "runtime_webrtc_negotiation_seconds_count", outcome="ok") == 1.0
    # The answer is not a connection: the wire is still cold, so the client has
    # not arrived and there is nothing to measure yet.
    assert _sample(metrics, "runtime_webrtc_connect_seconds_count") == 0.0


async def test_a_negotiation_that_raises_is_measured_as_a_failure(
    fake_peer: FakePeer,
    out_av_tracks: TrackMap,
) -> None:
    def refuses(*args: object, **kwargs: object) -> WebRtcPeerFactory:
        async def factory(
            conn_id: ConnId,
            offer: SdpOffer,
            tracks: TrackMap,
            config: WebRtcConfig,
            version: ProtocolVersion,
            /,
        ) -> tuple[WebRTCPeer, SdpAnswer]:
            raise RuntimeError("no peer today")

        return factory

    metrics = _metrics()
    acceptor = _acceptor(FakeSink(), fake_peer, refuses, metrics)

    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)

    assert _sample(metrics, "runtime_webrtc_negotiation_seconds_count", outcome="failed") == 1.0
    assert _sample(metrics, "runtime_webrtc_negotiation_seconds_count", outcome="ok") is None
    # An offer that produced no connection is reaped by nothing, so the acceptor
    # drops its own bookkeeping rather than holding it for the life of the process.
    assert acceptor._offered_at == {}


async def test_a_live_wire_is_measured_from_the_offer(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    metrics = _metrics()
    acceptor = _acceptor(FakeSink(), fake_peer, factory_for, metrics)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)

    fake_peer.fire_connected()

    # Measured from the offer, not from the answer: the client waits through both
    # legs, and only the total says how long it took to reach a usable wire.
    assert _sample(metrics, "runtime_webrtc_connect_seconds_count") == 1.0
    fake_peer.fire_disconnect()


async def test_a_reconnect_on_a_live_connection_is_measured(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    metrics = _metrics()
    acceptor = _acceptor(FakeSink(), fake_peer, factory_for, metrics)
    await _negotiate(acceptor, ConnId(7), SdpOffer("first"), out_av_tracks)
    fake_peer.fire_connected()

    await _negotiate(acceptor, ConnId(7), SdpOffer("second"), out_av_tracks)
    fake_peer.fire_connected()

    # The second offer installs its timestamp, and negotiating it closes the
    # connection the first one left behind. That teardown must not carry off the
    # bookkeeping of the offer that is still waiting, or a reconnect would read
    # as a client that answered and never arrived.
    assert _sample(metrics, "runtime_webrtc_connect_seconds_count") == 2.0
    fake_peer.fire_disconnect()


async def test_an_offer_that_never_connects_measures_no_wire(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    metrics = _metrics()
    acceptor = _acceptor(FakeSink(), fake_peer, factory_for, metrics)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)

    fake_peer.fire_disconnect()

    # A client that gives up is an absent connection, not a slow one. Recording a
    # duration for it would report the time it took to fail as a connect time.
    assert _sample(metrics, "runtime_webrtc_connect_seconds_count") == 0.0
    assert _sample(metrics, "runtime_webrtc_negotiation_seconds_count", outcome="ok") == 1.0


async def test_a_superseded_offer_is_not_measured(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    metrics = _metrics()
    acceptor = _acceptor(FakeSink(), fake_peer, factory_for, metrics)

    acceptor.start_offer(ConnId(7), SdpOffer("first"), out_av_tracks, ProtocolVersion.V0)
    first = acceptor._negotiating[ConnId(7)]
    acceptor.start_offer(ConnId(7), SdpOffer("second"), out_av_tracks, ProtocolVersion.V0)
    second = acceptor._negotiating[ConnId(7)]
    with pytest.raises(asyncio.CancelledError):
        await first
    await second

    # The client abandoned the first offer by re-offering, so it is neither an
    # answer nor a failure and counting it either way would be a lie.
    assert _sample(metrics, "runtime_webrtc_negotiation_seconds_count", outcome="ok") == 1.0
    assert _sample(metrics, "runtime_webrtc_negotiation_seconds_count", outcome="failed") is None


def _capped_acceptor(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    limit: int,
    negotiation_timeout: float = 0.0,
) -> WebRTCAcceptor:
    return WebRTCAcceptor(
        sink=FakeSink(),
        config=WebRtcConfig(
            ping_timeout=0.0, max_connections=limit, negotiation_timeout=negotiation_timeout
        ),
        peer_factory=factory_for(fake_peer),
        metrics=WebRtcMetrics(_metrics()),
    )


async def test_a_new_offer_past_the_ceiling_is_refused(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    acceptor = _capped_acceptor(fake_peer, factory_for, limit=1)
    acceptor.start_offer(ConnId(7), SdpOffer("first"), out_av_tracks, ProtocolVersion.V0)

    # The first connection fills the single slot, so a second, distinct
    # connection is refused rather than negotiated into a second native peer.
    with pytest.raises(TooManyConnectionsError):
        acceptor.start_offer(ConnId(8), SdpOffer("second"), out_av_tracks, ProtocolVersion.V0)
    assert ConnId(8) not in acceptor._negotiating
    await acceptor._negotiating[ConnId(7)]


async def test_a_reconnect_is_admitted_at_the_ceiling(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    acceptor = _capped_acceptor(fake_peer, factory_for, limit=1)
    await _negotiate(acceptor, ConnId(7), SdpOffer("first"), out_av_tracks)

    # A re-offer on the connection already holding the slot is a reconnect, not a
    # new peer, so it passes even with no slot free.
    acceptor.start_offer(ConnId(7), SdpOffer("again"), out_av_tracks, ProtocolVersion.V0)
    await acceptor._negotiating[ConnId(7)]
    assert acceptor.take_answer(ConnId(7)) is not None


async def test_a_freed_slot_admits_a_later_connection(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    acceptor = _capped_acceptor(fake_peer, factory_for, limit=1)
    await _negotiate(acceptor, ConnId(7), SdpOffer("first"), out_av_tracks)
    fake_peer.fire_connected()
    fake_peer.fire_disconnect()

    # The first connection dropped and freed the slot, so a fresh connection is
    # admitted: the ceiling bounds concurrency, not the session's lifetime total.
    acceptor.start_offer(ConnId(8), SdpOffer("second"), out_av_tracks, ProtocolVersion.V0)
    await acceptor._negotiating[ConnId(8)]


async def test_zero_ceiling_disables_the_cap(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    acceptor = _capped_acceptor(fake_peer, factory_for, limit=0)
    tasks = []
    for cid in range(7, 20):
        acceptor.start_offer(ConnId(cid), SdpOffer("offer"), out_av_tracks, ProtocolVersion.V0)
        tasks.append(acceptor._negotiating[ConnId(cid)])
    await asyncio.gather(*tasks)


async def test_pre_offer_ice_is_bounded_per_connection(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    acceptor = _acceptor(FakeSink(), fake_peer, factory_for)
    for i in range(_MAX_PENDING_ICE_PER_CONN + 50):
        await acceptor.add_ice(ConnId(7), IceCandidate(f"cand-{i}", "0", 0))

    # A connection that buffers ICE before ever offering cannot grow the buffer
    # without bound; candidates past the per-connection cap are dropped.
    assert len(acceptor._pending_ice[ConnId(7)]) == _MAX_PENDING_ICE_PER_CONN


async def test_pre_offer_ice_is_bounded_across_connections(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    acceptor = _acceptor(FakeSink(), fake_peer, factory_for)
    for cid in range(_MAX_PENDING_ICE_CONNS + 50):
        await acceptor.add_ice(ConnId(cid), IceCandidate("cand", "0", 0))

    # Candidates addressed to a flood of connections that never offer cannot grow
    # the set of buffered connections without bound.
    assert len(acceptor._pending_ice) == _MAX_PENDING_ICE_CONNS


async def test_pre_offer_ice_evicts_the_oldest_connection_at_the_bound(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    acceptor = _acceptor(FakeSink(), fake_peer, factory_for)
    for cid in range(_MAX_PENDING_ICE_CONNS):
        await acceptor.add_ice(ConnId(cid), IceCandidate("cand", "0", 0))

    # A fresh connection at the bound evicts the oldest pre-offer entry rather
    # than being refused, so junk ids that never offer cannot starve it.
    await acceptor.add_ice(ConnId(9999), IceCandidate("late", "0", 0))
    assert len(acceptor._pending_ice) == _MAX_PENDING_ICE_CONNS
    assert ConnId(0) not in acceptor._pending_ice
    assert ConnId(9999) in acceptor._pending_ice


async def test_negotiation_deadline_reaps_a_connection_that_never_goes_live(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    acceptor = _capped_acceptor(fake_peer, factory_for, limit=1, negotiation_timeout=0.05)
    answer = await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    assert answer is not None
    assert ConnId(7) in acceptor._conns

    # The connection answered but never fired connected, so the deadline closes
    # it and frees its slot.
    _, deadline = acceptor._deadlines[ConnId(7)]
    await deadline
    assert ConnId(7) not in acceptor._conns
    assert ConnId(7) not in acceptor._deadlines
    assert fake_peer.closed is True

    # The freed slot admits a fresh connection at the single-slot ceiling.
    acceptor.start_offer(ConnId(8), SdpOffer("second"), out_av_tracks, ProtocolVersion.V0)
    await acceptor._negotiating[ConnId(8)]


async def test_the_deadline_is_cancelled_when_the_wire_connects(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    acceptor = _capped_acceptor(fake_peer, factory_for, limit=1, negotiation_timeout=0.05)
    await _negotiate(acceptor, ConnId(7), SdpOffer("offer"), out_av_tracks)
    fake_peer.fire_connected()

    # A connection that reached its live wire keeps its slot; its deadline is
    # cancelled and never reaps it.
    assert ConnId(7) not in acceptor._deadlines
    await asyncio.sleep(0.1)
    assert ConnId(7) in acceptor._live
    assert fake_peer.closed is False
    fake_peer.fire_disconnect()


async def test_a_reconnects_deadline_survives_the_superseded_teardown(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    acceptor = _capped_acceptor(fake_peer, factory_for, limit=1, negotiation_timeout=0.05)
    await _negotiate(acceptor, ConnId(7), SdpOffer("first"), out_av_tracks)

    # Re-offering closes the superseded connection; that teardown runs after the
    # new offer armed its deadline and must not carry it off.
    await _negotiate(acceptor, ConnId(7), SdpOffer("second"), out_av_tracks)
    assert ConnId(7) in acceptor._deadlines

    # The reconnect never goes live either, so its own deadline reaps it and
    # frees the slot — re-offering cannot pin a slot for the process's life.
    _, deadline = acceptor._deadlines[ConnId(7)]
    await deadline
    assert ConnId(7) not in acceptor._conns
    assert ConnId(7) not in acceptor._deadlines
    acceptor.start_offer(ConnId(8), SdpOffer("third"), out_av_tracks, ProtocolVersion.V0)
    await acceptor._negotiating[ConnId(8)]


async def test_a_failed_negotiation_leaves_no_deadline_behind(
    fake_peer: FakePeer,
    out_av_tracks: TrackMap,
) -> None:
    def refuses(*args: object, **kwargs: object) -> WebRtcPeerFactory:
        async def factory(
            conn_id: ConnId,
            offer: SdpOffer,
            tracks: TrackMap,
            config: WebRtcConfig,
            version: ProtocolVersion,
            /,
        ) -> tuple[WebRTCPeer, SdpAnswer]:
            raise RuntimeError("no peer today")

        return factory

    acceptor = _capped_acceptor(fake_peer, refuses, limit=1, negotiation_timeout=0.05)
    for attempt in range(5):
        acceptor.start_offer(
            ConnId(100 + attempt), SdpOffer("offer"), out_av_tracks, ProtocolVersion.V0
        )
        await acceptor._negotiating[ConnId(100 + attempt)]

    # A failed negotiation reaps its own offer, so its deadline is torn down
    # with it: repeated failing offers on distinct ids cannot grow the map.
    assert acceptor._deadlines == {}
    assert acceptor._offered_at == {}
