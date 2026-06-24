import asyncio
from collections.abc import Callable

import numpy as np
from conftest import FakePeer

from reactor_runtime.core import Connection, ConnId, InputFrame, MediaBundle
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.transport.webrtc import (
    PeerStats,
    SdpOffer,
    TrackMap,
    WebRtcConfig,
    WebRTCConnection,
    WebRtcPeerFactory,
)
from reactor_runtime.transport.webrtc.signaling import IceCandidate


async def _connect(
    peer: FakePeer,
    factory: WebRtcPeerFactory,
    tracks: TrackMap,
    *,
    ping_timeout: float = 0.0,
) -> WebRTCConnection:
    conn, _ = await WebRTCConnection.create(
        ConnId(1),
        SdpOffer("offer"),
        tracks,
        WebRtcConfig(ping_timeout=ping_timeout),
        ProtocolVersion.V0,
        peer_factory=factory,
    )
    return conn


async def test_create_returns_answer_and_capabilities(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn, answer = await WebRTCConnection.create(
        ConnId(9),
        SdpOffer("offer"),
        out_av_tracks,
        WebRtcConfig(),
        ProtocolVersion.V0,
        peer_factory=factory_for(fake_peer),
    )
    assert answer.sdp == "answer-sdp"
    assert conn.id == ConnId(9)
    assert conn.capabilities.carries_video is True
    assert conn.capabilities.carries_audio is True
    assert conn.protocol_version is ProtocolVersion.V0


async def test_data_only_tracks_carry_no_media(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
) -> None:
    inbound_only = TrackMap.from_client(
        [{"mid": "0", "name": "webcam", "kind": "video", "direction": "sendonly"}]
    )
    conn = await _connect(fake_peer, factory_for(fake_peer), inbound_only)
    assert conn.capabilities.carries_video is False
    assert conn.capabilities.carries_audio is False


async def test_is_a_connection(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    assert isinstance(conn, Connection)


async def test_outbound_commands_delegate_to_peer(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    bundle = MediaBundle()
    conn.send_message(b"payload")
    conn.send_message('{"type":"current_mode"}')
    conn.send_media(bundle)
    conn.resume_track("main_video")
    conn.pause_track("main_video")
    await conn.add_ice(IceCandidate("cand", "0", 0))
    assert fake_peer.messages == [b"payload", '{"type":"current_mode"}']
    assert fake_peer.sent_media == [bundle]
    assert fake_peer.resumed == ["main_video"]
    assert fake_peer.paused == ["main_video"]
    assert fake_peer.ice == [IceCandidate("cand", "0", 0)]


async def test_inbound_message_and_media_forwarded(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    messages: list[tuple[bytes | str, ProtocolVersion]] = []
    media: list[tuple[str, InputFrame]] = []
    conn.on_message(lambda payload, version: messages.append((payload, version)))
    conn.on_media(lambda track, frame: media.append((track, frame)))

    frame = InputFrame(np.zeros((1, 1, 3), dtype=np.uint8))
    fake_peer.fire_message(b"hello")
    fake_peer.fire_message('{"scope":"runtime"}')
    fake_peer.fire_media("webcam", frame)

    assert messages == [
        (b"hello", ProtocolVersion.V0),
        ('{"scope":"runtime"}', ProtocolVersion.V0),
    ]
    assert media == [("webcam", frame)]


async def test_ping_reports_keepalive(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    pings: list[int] = []
    conn.on_ping(lambda: pings.append(1))
    fake_peer.fire_ping()
    assert pings == [1]


async def test_close_is_silent(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    events: list[str] = []
    conn.on_connected(lambda: events.append("up"))
    conn.on_disconnect(lambda: events.append("down"))

    fake_peer.fire_connected()
    await conn.close()

    assert fake_peer.closed is True
    assert events == ["up"]


async def test_close_fires_on_closed_once_not_disconnect(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    events: list[str] = []
    conn.on_disconnect(lambda: events.append("down"))
    conn.on_closed(lambda: events.append("closed"))

    fake_peer.fire_connected()
    await conn.close()
    await conn.close()

    assert events == ["closed"]


async def test_peer_loss_does_not_fire_on_closed(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    events: list[str] = []
    conn.on_disconnect(lambda: events.append("down"))
    conn.on_closed(lambda: events.append("closed"))

    fake_peer.fire_connected()
    fake_peer.fire_disconnect()

    assert events == ["down"]


async def test_peer_disconnect_reports_loss_once(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    downs: list[int] = []
    conn.on_disconnect(lambda: downs.append(1))

    fake_peer.fire_connected()
    fake_peer.fire_disconnect()
    fake_peer.fire_disconnect()

    assert downs == [1]


async def test_watchdog_times_out_when_no_ping(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks, ping_timeout=0.02)
    conn._WATCHDOG_POLL_SECONDS = 0.01
    downs: list[int] = []
    conn.on_disconnect(lambda: downs.append(1))

    fake_peer.fire_connected()
    await asyncio.sleep(0.12)

    assert downs == [1]
    assert fake_peer.closed is True


async def test_peer_disconnect_does_not_reclose_peer(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks)
    conn.on_disconnect(lambda: None)
    fake_peer.fire_connected()
    fake_peer.fire_disconnect()
    # A peer reports disconnect only after releasing its own wire, so the
    # connection must not close it again on this path.
    assert fake_peer.closed is False


async def test_pings_keep_watchdog_alive(
    fake_peer: FakePeer,
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    conn = await _connect(fake_peer, factory_for(fake_peer), out_av_tracks, ping_timeout=0.06)
    conn._WATCHDOG_POLL_SECONDS = 0.01
    downs: list[int] = []
    conn.on_disconnect(lambda: downs.append(1))

    fake_peer.fire_connected()
    for _ in range(4):
        await asyncio.sleep(0.02)
        fake_peer.fire_ping()
    assert downs == []
    await conn.close()


async def test_stats_polling_samples(
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    peer = FakePeer(stats=PeerStats(rtt_seconds=0.25))
    conn = await _connect(peer, factory_for(peer), out_av_tracks, ping_timeout=0.0)
    conn._STATS_INTERVAL_SECONDS = 0.01
    samples: list[PeerStats] = []
    conn.on_stats(samples.append)

    peer.fire_connected()
    await asyncio.sleep(0.05)

    assert conn.latest_stats == PeerStats(rtt_seconds=0.25)
    assert len(samples) >= 1
    await conn.close()


async def test_stats_loop_survives_a_failed_sample(
    factory_for: Callable[..., WebRtcPeerFactory],
    out_av_tracks: TrackMap,
) -> None:
    peer = FakePeer(stats=PeerStats(rtt_seconds=0.5))
    peer.stats_fail_times = 1
    conn = await _connect(peer, factory_for(peer), out_av_tracks, ping_timeout=0.0)
    conn._STATS_INTERVAL_SECONDS = 0.01
    peer.fire_connected()
    await asyncio.sleep(0.06)

    # The first sample raised; the sampler kept going and recorded a later one.
    assert conn.latest_stats == PeerStats(rtt_seconds=0.5)
    await conn.close()
