from collections.abc import Callable

import numpy as np
from conftest import FakePeer

from reactor_runtime.core import Connection, ConnId, InputFrame
from reactor_runtime.transport.webrtc import (
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


async def test_offer_returns_answer_without_opening(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    answer = await acceptor.offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
    assert answer.sdp == "answer-sdp"
    assert sink.opened == []


async def test_connection_opened_only_on_connect(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    sink = FakeSink()
    acceptor = _acceptor(sink, fake_peer, factory_for)
    await acceptor.offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
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
    await acceptor.offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
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
    await acceptor.offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
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
    await acceptor.offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
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
    await acceptor.offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
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
    await acceptor.offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
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
    await acceptor.offer(ConnId(7), SdpOffer("offer"), out_av_tracks)
    assert fake_peer.ice == [IceCandidate("early")]
