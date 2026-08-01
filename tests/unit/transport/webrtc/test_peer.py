"""Unit tests for the libwebrtc peer's connection-independent logic.

These exercise the peer without a live libwebrtc connection: protocol sniffing,
media routing and pause, ICE-config mapping, stats mapping, and the loss/close
lifecycle. Importing the peer pulls in the native ``reactor_webrtc`` module, so
the suite skips cleanly when the wheel is absent.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

rw = pytest.importorskip("reactor_webrtc")

from reactor_runtime.core import ConnId, MediaBundle  # noqa: E402
from reactor_runtime.core.values import (  # noqa: E402
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.protocol import Channel, ProtocolVersion  # noqa: E402
from reactor_runtime.transport.webrtc.config import (  # noqa: E402
    IceServer,
    IceTransportPolicy,
    WebRtcConfig,
)
from reactor_runtime.transport.webrtc.peer import (  # noqa: E402
    _AUDIO_BUFFER_MAX_SAMPLES,
    WebRTCPeer,
    _build_rtc_config,
    _is_terminal_state,
    libwebrtc_peer_factory,
)
from reactor_runtime.transport.webrtc.signaling import MappedTrack, SdpOffer, TrackMap  # noqa: E402


class _FakeTrack:
    def __init__(self) -> None:
        self.pushed: list[tuple[bytes, int, int]] = []

    def push_video_frame(self, bgra: bytes, width: int, height: int) -> None:
        self.pushed.append((bgra, width, height))


class _FakeChannel:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, bool]] = []

    def send(self, data: bytes, binary: bool = True) -> None:
        self.sent.append((data, binary))


def _video_bundle(name: str, value: int = 1) -> MediaBundle:
    info = TrackInfo(name=name, kind=TrackKind.VIDEO, direction=TrackDirection.OUT)
    return MediaBundle(
        tracks={name: TrackData(info=info, data=np.full((2, 2, 3), value, np.uint8))}
    )


# ── Protocol sniffing and liveness ──────────────────────────────────────────


async def test_message_sink_sniffs_once_and_reports_ping() -> None:
    peer = WebRTCPeer()
    peer._loop = asyncio.get_running_loop()
    messages: list[tuple[bytes | str, ProtocolVersion, Channel]] = []
    pings: list[int] = []
    peer.on_message(lambda payload, version, channel: messages.append((payload, version, channel)))
    peer.on_ping(lambda: pings.append(1))

    sink = peer._make_message_sink(Channel.DATA)
    sink(b'{"type": "hello"}', False)  # text JSON latches v0
    sink(b"\xff\xff", True)  # binary afterwards keeps the latched version
    await asyncio.sleep(0.01)

    assert messages[0] == ('{"type": "hello"}', ProtocolVersion.V0, Channel.DATA)
    assert messages[1] == (b"\xff\xff", ProtocolVersion.V0, Channel.DATA)
    assert pings == [1, 1]


async def test_control_channel_sink_tags_control() -> None:
    peer = WebRTCPeer()
    peer._loop = asyncio.get_running_loop()
    seen: list[Channel] = []
    peer.on_message(lambda _p, _v, channel: seen.append(channel))
    peer.on_ping(lambda: None)

    peer._make_message_sink(Channel.CONTROL)(b'{"a":1}', False)
    await asyncio.sleep(0.01)

    assert seen == [Channel.CONTROL]


# ── Sending ─────────────────────────────────────────────────────────────────


def test_send_message_and_control_route_by_channel() -> None:
    peer = WebRTCPeer()
    data: Any = _FakeChannel()
    control: Any = _FakeChannel()
    peer._data_channel = data
    peer._control_channel = control

    peer.send_message("hi")  # str encodes to a text frame
    peer.send_control(b"\x01\x02")  # bytes send as a binary frame

    assert data.sent == [(b"hi", False)]
    assert control.sent == [(b"\x01\x02", True)]


def test_send_media_drops_without_out_tracks() -> None:
    peer = WebRTCPeer()
    peer.send_media(_video_bundle("v"))
    assert peer._frame_queue.empty()


def test_send_media_enqueues_and_drops_when_full() -> None:
    peer = WebRTCPeer()
    track: Any = _FakeTrack()
    peer._out_tracks["v"] = track
    for _ in range(peer._frame_queue.maxsize + 5):
        peer.send_media(_video_bundle("v"))
    assert peer._frame_queue.full()
    assert peer._frame_queue.qsize() == peer._frame_queue.maxsize


# ── Outbound routing ─────────────────────────────────────────────────────────


def test_push_bundle_routes_video_to_its_track() -> None:
    peer = WebRTCPeer()
    track: Any = _FakeTrack()
    peer._out_tracks["v"] = track
    peer._push_bundle(_video_bundle("v", value=9))
    assert len(track.pushed) == 1
    bgra, width, height = track.pushed[0]
    assert (width, height) == (2, 2)
    assert len(bgra) == 2 * 2 * 4


def test_push_bundle_skips_paused_track() -> None:
    peer = WebRTCPeer()
    track: Any = _FakeTrack()
    peer._out_tracks["v"] = track
    peer.pause_track("v")
    peer._push_bundle(_video_bundle("v"))
    assert track.pushed == []
    peer.resume_track("v")
    peer._push_bundle(_video_bundle("v"))
    assert len(track.pushed) == 1


def test_push_bundle_buffers_audio_for_the_feeder() -> None:
    peer = WebRTCPeer()
    info = TrackInfo(name="a", kind=TrackKind.AUDIO, direction=TrackDirection.OUT)
    bundle = MediaBundle(tracks={"a": TrackData(info=info, data=np.zeros((1, 240), np.int16))})
    peer._push_bundle(bundle)
    assert peer._audio_buf.size == 240


def test_push_bundle_gap_fill_buffers_no_audio() -> None:
    peer = WebRTCPeer()
    track: Any = _FakeTrack()
    peer._out_tracks["v"] = track
    peer._push_bundle(_video_bundle("v"))
    assert len(track.pushed) == 1
    assert peer._audio_buf.size == 0


def test_enqueue_audio_caps_the_buffer_depth() -> None:
    peer = WebRTCPeer()
    for _ in range(50):
        peer._enqueue_audio(np.zeros(1_000, dtype=np.int16))
    assert peer._audio_buf.size == _AUDIO_BUFFER_MAX_SAMPLES


# ── ICE configuration ────────────────────────────────────────────────────────


def test_build_rtc_config_maps_ice_servers() -> None:
    config = WebRtcConfig(
        ice_servers=(IceServer(urls=("turn:host:3478",), username="u", credential="p"),)
    )
    rtc = _build_rtc_config(config)
    assert len(rtc.ice_servers) == 1
    assert list(rtc.ice_servers[0].urls) == ["turn:host:3478"]
    assert rtc.ice_servers[0].username == "u"
    assert rtc.ice_servers[0].password == "p"


def test_build_rtc_config_without_servers_is_empty() -> None:
    assert list(_build_rtc_config(WebRtcConfig()).ice_servers) == []


def test_build_rtc_config_relay_policy_does_not_raise() -> None:
    config = WebRtcConfig(transport_policy=IceTransportPolicy.RELAY)
    assert _build_rtc_config(config) is not None


# ── State classification ─────────────────────────────────────────────────────


def test_is_terminal_state() -> None:
    assert _is_terminal_state(rw.PeerConnectionState.Disconnected)
    assert _is_terminal_state(rw.PeerConnectionState.Failed)
    assert _is_terminal_state(rw.PeerConnectionState.Closed)
    assert not _is_terminal_state(rw.PeerConnectionState.Connected)
    assert not _is_terminal_state(rw.PeerConnectionState.New)
    assert not _is_terminal_state(rw.PeerConnectionState.Connecting)


# ── Inbound track naming ─────────────────────────────────────────────────────


def test_inbound_name_matches_metadata_by_kind_order() -> None:
    in_tracks = [
        TrackInfo(name="cam", kind=TrackKind.VIDEO, direction=TrackDirection.IN),
        TrackInfo(name="mic", kind=TrackKind.AUDIO, direction=TrackDirection.IN),
    ]
    assert WebRTCPeer._inbound_name(in_tracks, TrackKind.VIDEO, 0) == "cam"
    assert WebRTCPeer._inbound_name(in_tracks, TrackKind.AUDIO, 0) == "mic"


def test_inbound_name_falls_back_when_unmapped() -> None:
    assert WebRTCPeer._inbound_name([], TrackKind.VIDEO, 2) == "video-3"


# ── Stats mapping ─────────────────────────────────────────────────────────────


def test_stats_from_report_maps_tracks_and_rtt() -> None:
    peer = WebRTCPeer()
    peer._track_map = TrackMap(
        tracks=(
            MappedTrack(
                mid="0",
                info=TrackInfo(name="out_v", kind=TrackKind.VIDEO, direction=TrackDirection.OUT),
            ),
            MappedTrack(
                mid="1",
                info=TrackInfo(name="in_v", kind=TrackKind.VIDEO, direction=TrackDirection.IN),
            ),
        )
    )
    report: Any = SimpleNamespace(
        outbound_rtp=[SimpleNamespace(target_bitrate_bps=5000.0)],
        inbound_rtp=[SimpleNamespace(packets_lost=3, jitter_s=0.02)],
        candidate_pairs=[
            SimpleNamespace(
                state=rw.IceCandidatePairState.Succeeded, current_round_trip_time_s=0.05
            )
        ],
    )
    stats = peer._stats_from_report(report)

    assert stats.rtt_seconds == 0.05
    out = next(t for t in stats.tracks if t.direction is TrackDirection.OUT)
    assert out.name == "out_v"
    assert out.bitrate_bps == 5000
    inbound = next(t for t in stats.tracks if t.direction is TrackDirection.IN)
    assert inbound.name == "in_v"
    assert inbound.packet_loss == 3
    assert inbound.jitter == 0.02


def test_stats_from_report_ignores_negative_packet_loss() -> None:
    peer = WebRTCPeer()
    peer._track_map = TrackMap(
        tracks=(
            MappedTrack(
                mid="1",
                info=TrackInfo(name="in_v", kind=TrackKind.VIDEO, direction=TrackDirection.IN),
            ),
        )
    )
    report: Any = SimpleNamespace(
        outbound_rtp=[],
        inbound_rtp=[SimpleNamespace(packets_lost=-4, jitter_s=0.0)],
        candidate_pairs=[],
    )
    stats = peer._stats_from_report(report)
    assert stats.rtt_seconds is None
    assert stats.tracks[0].packet_loss == 0


# ── Lifecycle ─────────────────────────────────────────────────────────────────


async def test_report_loss_fires_disconnect_once() -> None:
    peer = WebRTCPeer()
    peer._loop = asyncio.get_running_loop()
    fired: list[int] = []
    peer.on_disconnect(lambda: fired.append(1))

    peer._report_loss()
    peer._report_loss()
    await asyncio.sleep(0.01)

    assert fired == [1]
    assert peer._stop_event.is_set()


async def test_close_is_idempotent() -> None:
    peer = WebRTCPeer()
    await peer.close()
    await peer.close()
    assert peer._stop_event.is_set()
    assert peer._pc is None


async def test_factory_rejects_empty_offer() -> None:
    with pytest.raises(ValueError, match="empty SDP offer"):
        await libwebrtc_peer_factory(
            ConnId(1), SdpOffer(sdp="   "), TrackMap(), WebRtcConfig(), ProtocolVersion.V0
        )
