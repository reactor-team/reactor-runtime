"""Unit tests for the libwebrtc peer's connection-independent logic.

These exercise the peer without a live libwebrtc connection: protocol sniffing,
media routing and pause, ICE-config mapping, stats mapping, and the loss/close
lifecycle. Importing the peer pulls in the native ``reactor_webrtc`` module, so
the suite skips cleanly when the wheel is absent.
"""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

rw = pytest.importorskip("reactor_webrtc")

from reactor_runtime.core import ConnId, MediaBundle  # noqa: E402
from reactor_runtime.core.values import (  # noqa: E402
    InputFrame,
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
    _AUDIO_FRAME_SAMPLES,
    _AUDIO_SILENT_FRAME,
    _FRAME_QUEUE_MAX,
    WebRTCPeer,
    _apply_bitrate_limits,
    _build_rtc_config,
    _is_terminal_state,
    libwebrtc_peer_factory,
)
from reactor_runtime.transport.webrtc.signaling import MappedTrack, SdpOffer, TrackMap  # noqa: E402


class _FakeTrack:
    def __init__(self) -> None:
        self.pushed: list[tuple[bytes, int, int]] = []
        self.user_data: list[bytes | None] = []

    def push_video_frame(
        self, bgra: bytes, width: int, height: int, user_data: bytes | None = None
    ) -> None:
        self.pushed.append((bgra, width, height))
        self.user_data.append(user_data)


class _FakeAudioTrack:
    def __init__(self) -> None:
        self.pushed: list[tuple[bytes, int, int]] = []

    def push_pcm(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        self.pushed.append((pcm, sample_rate, channels))


class _FakeChannel:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, bool]] = []

    def send(self, data: bytes, binary: bool = True) -> None:
        self.sent.append((data, binary))


def _video_bundle(
    name: str, value: int = 1, metadata: bytes | list[bytes] | None = None
) -> MediaBundle:
    info = TrackInfo(name=name, kind=TrackKind.VIDEO, direction=TrackDirection.OUT)
    return MediaBundle(
        tracks={
            name: TrackData(info=info, data=np.full((2, 2, 3), value, np.uint8), metadata=metadata)
        }
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


def test_push_bundle_hands_frame_metadata_to_the_track() -> None:
    peer = WebRTCPeer()
    track: Any = _FakeTrack()
    peer._out_tracks["v"] = track
    peer._push_bundle(_video_bundle("v", metadata=b'{"seed":7}'))
    assert track.user_data == [b'{"seed":7}']


def test_push_bundle_sends_no_metadata_when_the_frame_carries_none() -> None:
    peer = WebRTCPeer()
    track: Any = _FakeTrack()
    peer._out_tracks["v"] = track
    peer._push_bundle(_video_bundle("v"))
    assert track.user_data == [None]


def test_push_bundle_sends_no_metadata_for_an_unsplit_batch() -> None:
    # A bundle reaching the wire holds a single frame, so a list here is a
    # bundle that skipped the split; the frame still goes out, undescribed.
    peer = WebRTCPeer()
    track: Any = _FakeTrack()
    peer._out_tracks["v"] = track
    peer._push_bundle(_video_bundle("v", metadata=[b"a", b"b"]))
    assert len(track.pushed) == 1
    assert track.user_data == [None]


# ── Inbound metadata ─────────────────────────────────────────────────────────


async def test_video_sink_surfaces_the_metadata_the_sender_attached() -> None:
    peer = WebRTCPeer()
    peer._loop = asyncio.get_running_loop()
    frames: list[InputFrame] = []
    peer.on_media(lambda _name, frame: frames.append(frame))

    peer._make_video_sink("cam")(bytes(2 * 2 * 4), 2, 2, rw.FrameMetadata(user_data=b'{"pose":1}'))
    await asyncio.sleep(0.01)

    assert frames[0].metadata == b'{"pose":1}'
    assert frames[0].data.shape == (2, 2, 3)


async def test_video_sink_reads_an_empty_trailer_as_no_metadata() -> None:
    peer = WebRTCPeer()
    peer._loop = asyncio.get_running_loop()
    frames: list[InputFrame] = []
    peer.on_media(lambda _name, frame: frames.append(frame))

    peer._make_video_sink("cam")(bytes(2 * 2 * 4), 2, 2, rw.FrameMetadata(user_data=b""))
    await asyncio.sleep(0.01)

    assert frames[0].metadata is None


async def test_video_sink_accepts_a_frame_that_carries_no_trailer() -> None:
    peer = WebRTCPeer()
    peer._loop = asyncio.get_running_loop()
    frames: list[InputFrame] = []
    peer.on_media(lambda _name, frame: frames.append(frame))

    peer._make_video_sink("cam")(bytes(2 * 2 * 4), 2, 2, None)
    await asyncio.sleep(0.01)

    assert frames[0].metadata is None


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


def test_enqueue_audio_counts_the_samples_the_cap_discards() -> None:
    peer = WebRTCPeer()
    peer._enqueue_audio(np.zeros(_AUDIO_BUFFER_MAX_SAMPLES + 1_500, dtype=np.int16))
    assert peer._dropped_samples == 1_500


def test_send_media_counts_bundles_the_full_queue_rejects() -> None:
    peer = WebRTCPeer()
    peer._out_tracks["v"] = cast(Any, _FakeTrack())
    for _ in range(_FRAME_QUEUE_MAX + 3):
        peer.send_media(_video_bundle("v"))
    assert peer._dropped_bundles == 3


# ── Outbound audio feed ──────────────────────────────────────────────────────


def test_audio_feed_pushes_the_model_samples_when_a_frame_is_ready() -> None:
    peer = WebRTCPeer()
    track = _FakeAudioTrack()
    peer._enqueue_audio(np.full(_AUDIO_FRAME_SAMPLES, 7, dtype=np.int16))

    peer._push_audio_frame(cast(Any, track))

    expected = np.full(_AUDIO_FRAME_SAMPLES, 7, dtype=np.int16).tobytes()
    assert track.pushed == [(expected, 48_000, 1)]
    assert peer._silence_frames == 0


def test_audio_feed_pushes_silence_when_the_buffer_is_empty() -> None:
    """An empty buffer still owes the wire 10 ms, or the sample clock stalls."""
    peer = WebRTCPeer()
    track = _FakeAudioTrack()

    peer._push_audio_frame(cast(Any, track))

    assert track.pushed == [(np.zeros(_AUDIO_FRAME_SAMPLES, np.int16).tobytes(), 48_000, 1)]
    assert peer._silence_frames == 1


def test_audio_feed_counts_a_partial_frame_as_silence() -> None:
    peer = WebRTCPeer()
    track = _FakeAudioTrack()
    peer._enqueue_audio(np.ones(_AUDIO_FRAME_SAMPLES - 1, dtype=np.int16))

    peer._push_audio_frame(cast(Any, track))

    assert peer._silence_frames == 1
    assert peer._audio_buf.size == _AUDIO_FRAME_SAMPLES - 1


def test_audio_feed_survives_a_track_that_raises() -> None:
    peer = WebRTCPeer()

    class _Raising:
        def push_pcm(self, pcm: bytes, rate: int, channels: int) -> None:
            raise RuntimeError("boom")

    peer._push_audio_frame(cast(Any, _Raising()))

    assert peer._silence_frames == 1


def test_audio_feed_loop_keeps_the_wire_fed_through_an_empty_buffer() -> None:
    peer = WebRTCPeer()
    track = _FakeAudioTrack()
    peer._audio_track = cast(Any, track)
    thread = threading.Thread(target=peer._audio_feed_loop, daemon=True)

    thread.start()
    time.sleep(0.15)
    peer._stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    # 150 ms of 10 ms ticks, with generous headroom for a loaded scheduler.
    assert len(track.pushed) >= 5
    assert all(pcm == _AUDIO_SILENT_FRAME for pcm, _, _ in track.pushed)


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


def test_build_rtc_config_maps_relay_policy() -> None:
    config = WebRtcConfig(transport_policy=IceTransportPolicy.RELAY)
    assert _build_rtc_config(config).ice_transport_type == "relay"


def test_build_rtc_config_defaults_to_all_policy() -> None:
    assert _build_rtc_config(WebRtcConfig()).ice_transport_type == "all"


def test_build_rtc_config_maps_port_range() -> None:
    rtc = _build_rtc_config(WebRtcConfig(port_range=(10000, 10100)))
    assert (rtc.min_port, rtc.max_port) == (10000, 10100)


def test_build_rtc_config_leaves_port_range_at_default_when_unset() -> None:
    rtc = _build_rtc_config(WebRtcConfig())
    assert (rtc.min_port, rtc.max_port) == (0, 0)


class _FakePeerConnection:
    def __init__(self) -> None:
        self.bitrate_calls: list[tuple[int, int, int]] = []

    async def set_bitrate(self, min_bps: int, start_bps: int, max_bps: int) -> None:
        self.bitrate_calls.append((min_bps, start_bps, max_bps))


async def test_apply_bitrate_limits_converts_kbps_to_bps_in_min_start_max_order() -> None:
    pc = _FakePeerConnection()
    config = WebRtcConfig(bwe_min_kbps=800, bwe_initial_kbps=3000, bwe_max_kbps=8000)

    await _apply_bitrate_limits(cast("rw.PeerConnection", pc), config)

    assert pc.bitrate_calls == [(800_000, 3_000_000, 8_000_000)]


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
        outbound_rtp=[SimpleNamespace(target_bitrate_bps=5000.0, packets_sent=417)],
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
    assert out.packets_sent == 417
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


def test_stats_report_carries_the_outbound_media_counters() -> None:
    peer = WebRTCPeer()
    peer._silence_frames = 12
    peer._dropped_samples = 480
    peer._dropped_bundles = 2
    report: Any = SimpleNamespace(outbound_rtp=[], inbound_rtp=[], candidate_pairs=[])

    media = peer._stats_from_report(report).media

    assert media.silence_frames == 12
    assert media.dropped_samples == 480
    assert media.dropped_bundles == 2


async def test_stats_on_a_closed_peer_still_reports_media_counters() -> None:
    peer = WebRTCPeer()
    peer._silence_frames = 3

    stats = await peer.stats()

    assert stats.media.silence_frames == 3


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_underrun_warning_needs_a_baseline_sample(caplog: pytest.LogCaptureFixture) -> None:
    peer = WebRTCPeer()
    peer._silence_frames = 10_000

    with caplog.at_level(logging.WARNING):
        peer._warn_on_audio_underrun()

    assert _warnings(caplog) == []


def test_underrun_warning_fires_once_silence_dominates_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    peer = WebRTCPeer()
    # A second of frames, a fifth of it silence the runtime inserted.
    peer._last_silence_sample = (time.monotonic() - 1.0, 0)
    peer._silence_frames = 20

    with caplog.at_level(logging.WARNING):
        peer._warn_on_audio_underrun()

    assert len(_warnings(caplog)) == 1
    assert "below real time" in _warnings(caplog)[0]


def test_underrun_warning_stays_quiet_for_an_occasional_gap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    peer = WebRTCPeer()
    peer._last_silence_sample = (time.monotonic() - 1.0, 0)
    peer._silence_frames = 2

    with caplog.at_level(logging.WARNING):
        peer._warn_on_audio_underrun()

    assert _warnings(caplog) == []


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
