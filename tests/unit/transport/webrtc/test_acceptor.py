import asyncio
from collections.abc import Callable

import numpy as np
import pytest
from conftest import FakePeer

from reactor_runtime.core import Connection, ConnId, InputFrame
from reactor_runtime.transport.webrtc import (
    SdpAnswer,
    SdpOffer,
    TrackMap,
    WebRTCAcceptor,
    WebRtcConfig,
    WebRtcPeerFactory,
)
from reactor_runtime.transport.webrtc.signaling import IceCandidate


class FakeSink:
    """A ConnectionSink that records every fact pushed up to it."""

    def __init__(self) -> None:
        self.opened: list[ConnId] = []
        self.closed: list[ConnId] = []
        self.messages: list[tuple[ConnId, bytes | str]] = []
        self.media: list[tuple[ConnId, str]] = []
        self.keepalives: list[ConnId] = []

    def connection_opened(self, conn: Connection) -> None:
        self.opened.append(conn.id)

    def connection_closed(self, conn_id: ConnId) -> None:
        self.closed.append(conn_id)

    def message_received(self, conn_id: ConnId, payload: bytes | str) -> None:
        self.messages.append((conn_id, payload))

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        self.media.append((conn_id, track))

    def keepalive(self, conn_id: ConnId) -> None:
        self.keepalives.append(conn_id)


def _acceptor(
    sink: FakeSink,
    peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> WebRTCAcceptor:
    return WebRTCAcceptor(
        sink=sink,
        config=WebRtcConfig(ping_timeout=0.0),
        peer_factory=factory_for(peer),
    )


async def _negotiate(
    acceptor: WebRTCAcceptor, conn_id: ConnId, offer: SdpOffer, tracks: TrackMap
) -> SdpAnswer | None:
    """Start an offer and drive its background negotiation to completion.

    ``start_offer`` is synchronous: it queues the negotiation task without
    yielding, so the task can be captured from ``_negotiating`` and awaited
    before draining the answer.
    """
    acceptor.start_offer(conn_id, offer, tracks)
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


async def test_take_answer_is_none_until_negotiation_completes(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    acceptor.start_offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
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
    acceptor.start_offer(ConnId(7), SdpOffer("first"), out_av_tracks)
    first = acceptor._negotiating[ConnId(7)]
    acceptor.start_offer(ConnId(7), SdpOffer("second"), out_av_tracks)
    second = acceptor._negotiating[ConnId(7)]
    assert second is not first
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    answer = acceptor.take_answer(ConnId(7))
    assert answer is not None
