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
    _AUDIO_GRACE_TICKS,
    _AUDIO_SILENT_FRAME,
    _FRAME_QUEUE_MAX,
    WebRTCPeer,
    _apply_bitrate_limits,
    _build_rtc_config,
    _is_terminal_state,
    _video_codec_preferences,
    libwebrtc_peer_factory,
)
from reactor_runtime.transport.webrtc.signaling import (  # noqa: E402
    IceCandidate,
    MappedTrack,
    SdpOffer,
    TrackMap,
)

# A capture timestamp standing in for one read of libwebrtc's clock.
_CAPTURED_US = 1_000_000


class _FakeTrack:
    def __init__(self) -> None:
        self.pushed: list[tuple[bytes, int, int]] = []
        self.user_data: list[bytes | None] = []
        self.capture_times: list[int | None] = []

    def kind(self) -> Any:
        return rw.MediaKind.Video

    def push_video_frame(
        self,
        bgra: bytes,
        width: int,
        height: int,
        user_data: bytes | None = None,
        capture_time_us: int | None = None,
    ) -> None:
        self.pushed.append((bgra, width, height))
        self.user_data.append(user_data)
        self.capture_times.append(capture_time_us)


class _FakeAudioTrack:
    def __init__(self) -> None:
        self.pushed: list[tuple[bytes, int, int]] = []
        self.capture_times: list[int | None] = []

    def kind(self) -> Any:
        return rw.MediaKind.Audio

    def push_pcm(
        self,
        pcm: bytes,
        sample_rate: int,
        channels: int,
        capture_time_us: int | None = None,
    ) -> None:
        self.pushed.append((pcm, sample_rate, channels))
        self.capture_times.append(capture_time_us)


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


async def test_video_sink_surfaces_the_capture_stamp_beside_the_metadata() -> None:
    peer = WebRTCPeer()
    peer._loop = asyncio.get_running_loop()
    frames: list[InputFrame] = []
    peer.on_media(lambda _name, frame: frames.append(frame))

    peer._make_video_sink("cam")(
        bytes(2 * 2 * 4),
        2,
        2,
        rw.FrameMetadata(capture_time_us=1_700_000_123_456, user_data=b"{}"),
    )
    await asyncio.sleep(0.01)

    assert frames[0].capture_time_us == 1_700_000_123_456
    assert frames[0].metadata == b"{}"


async def test_video_sink_reads_an_unset_stamp_as_no_capture_time() -> None:
    """Zero is the trailer's "unset", and a sender's clock never reads it.

    A sender on a current transport always declares something, so this is the
    hand-built trailer and the peer that carries the field without filling it —
    still worth pinning, because reading zero as a capture time would put every
    such frame at the epoch.
    """
    peer = WebRTCPeer()
    peer._loop = asyncio.get_running_loop()
    frames: list[InputFrame] = []
    peer.on_media(lambda _name, frame: frames.append(frame))

    peer._make_video_sink("cam")(bytes(2 * 2 * 4), 2, 2, rw.FrameMetadata(user_data=b"{}"))
    await asyncio.sleep(0.01)

    assert frames[0].capture_time_us is None
    assert frames[0].metadata == b"{}"


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
    assert frames[0].capture_time_us is None


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
    peer._out_tracks["a"] = cast(Any, _FakeAudioTrack())
    bundle = MediaBundle(tracks={"a": TrackData(info=info, data=np.zeros((1, 240), np.int16))})
    peer._push_bundle(bundle)
    assert peer._audio_bufs["a"].size == 240


def test_push_bundle_ignores_a_track_the_wire_does_not_hold() -> None:
    peer = WebRTCPeer()
    info = TrackInfo(name="a", kind=TrackKind.AUDIO, direction=TrackDirection.OUT)
    bundle = MediaBundle(tracks={"a": TrackData(info=info, data=np.zeros((1, 240), np.int16))})
    peer._push_bundle(bundle)
    assert peer._audio_bufs == {}


def test_push_bundle_gap_fill_buffers_no_audio() -> None:
    peer = WebRTCPeer()
    track: Any = _FakeTrack()
    peer._out_tracks["v"] = track
    peer._push_bundle(_video_bundle("v"))
    assert len(track.pushed) == 1
    assert peer._audio_bufs == {}


def test_enqueue_audio_caps_the_buffer_depth() -> None:
    peer = WebRTCPeer()
    for _ in range(50):
        peer._enqueue_audio("a", np.zeros(1_000, dtype=np.int16))
    assert peer._audio_bufs["a"].size == _AUDIO_BUFFER_MAX_SAMPLES


def test_enqueue_audio_counts_the_samples_the_cap_discards() -> None:
    peer = WebRTCPeer()
    peer._enqueue_audio("a", np.zeros(_AUDIO_BUFFER_MAX_SAMPLES + 1_500, dtype=np.int16))
    assert peer._dropped_samples == 1_500


def test_send_media_counts_bundles_the_full_queue_rejects() -> None:
    peer = WebRTCPeer()
    peer._out_tracks["v"] = cast(Any, _FakeTrack())
    for _ in range(_FRAME_QUEUE_MAX + 3):
        peer.send_media(_video_bundle("v"))
    assert peer._dropped_bundles == 3


# ── Outbound audio feed ──────────────────────────────────────────────────────


def _audio_peer() -> tuple[WebRTCPeer, _FakeAudioTrack]:
    peer = WebRTCPeer()
    track = _FakeAudioTrack()
    peer._out_tracks["a"] = cast(Any, track)
    return peer, track


def _make_live(peer: WebRTCPeer, track: _FakeAudioTrack, name: str = "a") -> None:
    """Run one real frame through, so the track counts as delivering.

    Silence covers a gap in a running stream, so a track that has never
    delivered gets none — every test about gap-filling has to start here.
    """
    peer._enqueue_audio(name, np.ones(_AUDIO_FRAME_SAMPLES, dtype=np.int16))
    peer._push_audio_frame(name)
    track.pushed.clear()
    track.capture_times.clear()
    peer._silence_frames.clear()


def test_audio_feed_pushes_the_model_samples_when_a_frame_is_ready() -> None:
    peer, track = _audio_peer()
    peer._enqueue_audio("a", np.full(_AUDIO_FRAME_SAMPLES, 7, dtype=np.int16))

    peer._push_audio_frame("a")

    expected = np.full(_AUDIO_FRAME_SAMPLES, 7, dtype=np.int16).tobytes()
    assert track.pushed == [(expected, 48_000, 1)]
    assert peer._silence_frames.get("a", 0) == 0


def test_audio_feed_pushes_silence_when_the_buffer_is_empty() -> None:
    """An empty buffer still owes the wire 10 ms, or the sample clock stalls."""
    peer, track = _audio_peer()
    _make_live(peer, track)

    peer._push_audio_frame("a")

    assert track.pushed == [(np.zeros(_AUDIO_FRAME_SAMPLES, np.int16).tobytes(), 48_000, 1)]
    assert peer._silence_frames["a"] == 1


def test_audio_feed_counts_a_partial_frame_as_silence() -> None:
    peer, track = _audio_peer()
    _make_live(peer, track)
    peer._enqueue_audio("a", np.ones(_AUDIO_FRAME_SAMPLES - 1, dtype=np.int16))

    peer._push_audio_frame("a")

    assert peer._silence_frames["a"] == 1
    assert peer._audio_bufs["a"].size == _AUDIO_FRAME_SAMPLES - 1


def test_a_session_without_audio_reports_no_shortfall() -> None:
    """A wire that carries no audio is owed none, so none is counted missing."""
    peer = WebRTCPeer()
    peer._out_tracks["v"] = cast(Any, _FakeTrack())  # video-only session

    for _ in range(100):  # a second of ticks
        peer._push_audio_tick()

    assert peer._silence_frames == {}


def test_a_track_that_never_delivered_is_fed_but_not_blamed() -> None:
    """The clock runs from the first tick; the model is not charged for it."""
    peer, track = _audio_peer()

    for _ in range(200):  # two seconds of ticks
        peer._push_audio_tick()

    assert len(track.pushed) == 200
    assert all(pcm == _AUDIO_SILENT_FRAME for pcm, _, _ in track.pushed)
    assert peer._silence_frames == {}


def test_a_track_that_goes_quiet_stops_being_covered() -> None:
    """After an unpublish the model owes nothing, so silence stops too."""
    peer, track = _audio_peer()
    _make_live(peer, track)

    for _ in range(300):  # three seconds with nothing arriving
        peer._push_audio_tick()

    # Covered through the grace, then left alone.
    assert len(track.pushed) == _AUDIO_GRACE_TICKS
    assert peer._silence_frames == {"a": _AUDIO_GRACE_TICKS}


def test_a_stall_past_the_grace_keeps_being_counted() -> None:
    """The count is how far the audio sits from where the model put it.

    Silence past the grace stops being the model's fault, but it does not stop
    displacing everything the model sends after it — so it stays on the tally
    even once it stops being warned about.
    """
    peer, track = _audio_peer()
    _make_live(peer, track)

    for _ in range(_AUDIO_GRACE_TICKS + 200):
        peer._push_audio_tick()

    assert peer._silence_frames == {"a": _AUDIO_GRACE_TICKS + 200}


def test_a_track_stalled_past_the_grace_is_no_longer_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model that speaks in bursts would otherwise warn through every gap."""
    peer, track = _audio_peer()
    _make_live(peer, track)
    for _ in range(_AUDIO_GRACE_TICKS):
        peer._push_audio_tick()
    peer._warn_on_audio_underrun()  # the baseline

    for _ in range(200):
        peer._push_audio_tick()
    with caplog.at_level(logging.WARNING):
        peer._warn_on_audio_underrun()

    assert not [r for r in caplog.records if "below real time" in r.getMessage()]


def test_a_gap_inside_the_grace_is_still_covered() -> None:
    """A stall is what silence is for; it has to survive the idle rule."""
    peer, track = _audio_peer()
    _make_live(peer, track)

    for _ in range(_AUDIO_GRACE_TICKS - 1):
        peer._push_audio_tick()

    assert len(track.pushed) == _AUDIO_GRACE_TICKS - 1
    assert all(pcm == _AUDIO_SILENT_FRAME for pcm, _, _ in track.pushed)


def test_audio_arriving_again_reopens_the_grace() -> None:
    """A track that resumes is running again, gaps and all."""
    peer, track = _audio_peer()
    _make_live(peer, track)
    for _ in range(300):
        peer._push_audio_tick()
    covered = len(track.pushed)

    peer._enqueue_audio("a", np.ones(_AUDIO_FRAME_SAMPLES, dtype=np.int16))
    peer._push_audio_tick()  # the real frame
    for _ in range(300):
        peer._push_audio_tick()

    assert len(track.pushed) == covered + 1 + _AUDIO_GRACE_TICKS


def test_audio_feed_survives_a_track_that_raises() -> None:
    peer = WebRTCPeer()

    class _Raising:
        def kind(self) -> Any:
            return rw.MediaKind.Audio

        def push_pcm(
            self, pcm: bytes, rate: int, channels: int, capture_time_us: int | None = None
        ) -> None:
            raise RuntimeError("boom")

    peer._idle_ticks["a"] = 0  # a running track
    peer._out_tracks["a"] = cast(Any, _Raising())
    peer._push_audio_frame("a")

    assert peer._silence_frames["a"] == 1


def test_a_name_that_is_not_an_audio_track_is_left_alone() -> None:
    """The check lives where the counting does, so no caller can bypass it."""
    peer = WebRTCPeer()
    peer._out_tracks["v"] = cast(Any, _FakeTrack())

    peer._push_audio_frame("v")  # a video track
    peer._push_audio_frame("gone")  # a track this wire never held

    assert peer._silence_frames == {}
    assert peer._audio_bufs == {}


# ── Pause and resume ─────────────────────────────────────────────────────────


def _av_bundle() -> MediaBundle:
    video = TrackInfo(name="v", kind=TrackKind.VIDEO, direction=TrackDirection.OUT)
    audio = TrackInfo(name="a", kind=TrackKind.AUDIO, direction=TrackDirection.OUT)
    return MediaBundle(
        tracks={
            "v": TrackData(info=video, data=np.zeros((2, 2, 3), np.uint8)),
            "a": TrackData(info=audio, data=np.ones((1, _AUDIO_FRAME_SAMPLES), np.int16)),
        }
    )


def _wired_peer() -> tuple[WebRTCPeer, _FakeTrack, _FakeAudioTrack]:
    peer = WebRTCPeer()
    video, audio = _FakeTrack(), _FakeAudioTrack()
    peer._out_tracks["v"] = cast(Any, video)
    peer._out_tracks["a"] = cast(Any, audio)
    return peer, video, audio


def test_pausing_video_stops_the_frames_and_leaves_audio_alone() -> None:
    peer, video, audio = _wired_peer()

    peer.pause_track("v")
    for _ in range(3):
        peer._push_bundle(_av_bundle())
        peer._push_audio_tick()

    assert video.pushed == []
    assert len(audio.pushed) == 3


def test_resuming_video_puts_frames_back_on_the_wire() -> None:
    peer, video, _ = _wired_peer()
    peer.pause_track("v")
    peer._push_bundle(_av_bundle())
    assert video.pushed == []

    peer.resume_track("v")
    peer._push_bundle(_av_bundle())

    assert len(video.pushed) == 1


def test_pausing_audio_sends_silence_rather_than_the_models_audio() -> None:
    """The client asked not to hear it, not to lose the clock behind it."""
    peer, _, audio = _wired_peer()

    peer.pause_track("a")
    for _ in range(5):
        peer._push_bundle(_av_bundle())
        peer._push_audio_tick()

    assert len(audio.pushed) == 5
    assert all(pcm == _AUDIO_SILENT_FRAME for pcm, _, _ in audio.pushed)


def test_a_paused_audio_track_is_not_counted_as_under_production() -> None:
    """Otherwise a client that pauses trips the model's under-production warning."""
    peer, _, _ = _wired_peer()

    peer.pause_track("a")
    for _ in range(20):
        peer._push_audio_tick()

    assert peer._silence_frames.get("a", 0) == 0


def test_resuming_audio_puts_the_models_samples_back() -> None:
    peer, _, audio = _wired_peer()
    peer.pause_track("a")
    peer._push_bundle(_av_bundle())
    peer._push_audio_tick()
    assert audio.pushed[0][0] == _AUDIO_SILENT_FRAME

    peer.resume_track("a")
    peer._push_bundle(_av_bundle())
    peer._push_audio_tick()

    assert len(audio.pushed) == 2
    assert audio.pushed[1][0] != _AUDIO_SILENT_FRAME


def test_a_pause_leaves_no_stale_audio_to_replay_on_resume() -> None:
    """The buffer drains during the pause, so resume plays what arrives next."""
    peer, _, audio = _wired_peer()
    peer._push_bundle(_av_bundle())  # a frame's worth arrives
    peer.pause_track("a")

    for _ in range(10):
        peer._push_bundle(_av_bundle())  # discarded while paused
        peer._push_audio_tick()

    assert peer._audio_bufs["a"].size == _AUDIO_FRAME_SAMPLES  # only the pre-pause frame
    assert all(pcm == _AUDIO_SILENT_FRAME for pcm, _, _ in audio.pushed)


# ── Several audio tracks ─────────────────────────────────────────────────────


def _two_audio_peer() -> tuple[WebRTCPeer, _FakeAudioTrack, _FakeAudioTrack]:
    peer = WebRTCPeer()
    voice, music = _FakeAudioTrack(), _FakeAudioTrack()
    for name, track in (("voice", voice), ("music", music)):
        peer._out_tracks[name] = cast(Any, track)
    return peer, voice, music


def _two_audio_bundle(voice_value: int = 111, music_value: int = 222) -> MediaBundle:
    def info(name: str) -> TrackInfo:
        return TrackInfo(name=name, kind=TrackKind.AUDIO, direction=TrackDirection.OUT)

    return MediaBundle(
        tracks={
            "voice": TrackData(
                info=info("voice"),
                data=np.full((1, _AUDIO_FRAME_SAMPLES), voice_value, np.int16),
            ),
            "music": TrackData(
                info=info("music"),
                data=np.full((1, _AUDIO_FRAME_SAMPLES), music_value, np.int16),
            ),
        }
    )


def test_each_audio_track_buffers_on_its_own() -> None:
    """Sharing one buffer would splice the tracks together in time."""
    peer, _, _ = _two_audio_peer()

    peer._push_bundle(_two_audio_bundle())

    assert peer._audio_bufs["voice"].size == _AUDIO_FRAME_SAMPLES
    assert peer._audio_bufs["music"].size == _AUDIO_FRAME_SAMPLES


def test_each_audio_track_gets_only_its_own_samples() -> None:
    peer, voice, music = _two_audio_peer()
    peer._push_bundle(_two_audio_bundle())

    peer._push_audio_tick()

    assert (
        np.frombuffer(voice.pushed[0][0], dtype=np.int16).tolist() == [111] * _AUDIO_FRAME_SAMPLES
    )
    assert (
        np.frombuffer(music.pushed[0][0], dtype=np.int16).tolist() == [222] * _AUDIO_FRAME_SAMPLES
    )


def test_one_tick_feeds_every_audio_track_once() -> None:
    peer, voice, music = _two_audio_peer()
    peer._push_bundle(_two_audio_bundle())

    peer._push_audio_tick()

    assert len(voice.pushed) == 1
    assert len(music.pushed) == 1


def test_an_empty_track_falls_silent_while_the_other_plays() -> None:
    peer, voice, music = _two_audio_peer()
    _make_live(peer, voice, "voice")
    _make_live(peer, music, "music")
    peer._enqueue_audio("voice", np.full(_AUDIO_FRAME_SAMPLES, 111, np.int16))

    peer._push_audio_tick()

    assert voice.pushed[0][0] != _AUDIO_SILENT_FRAME
    assert music.pushed[0][0] == _AUDIO_SILENT_FRAME
    assert peer._silence_frames == {"music": 1}


def test_pausing_one_audio_track_leaves_the_other_playing() -> None:
    peer, voice, music = _two_audio_peer()

    peer.pause_track("voice")
    peer._push_bundle(_two_audio_bundle())
    peer._push_audio_tick()

    assert voice.pushed[0][0] == _AUDIO_SILENT_FRAME  # the clock, not the audio
    assert music.pushed[0][0] != _AUDIO_SILENT_FRAME


def test_resuming_one_audio_track_leaves_the_other_undisturbed() -> None:
    peer, voice, music = _two_audio_peer()
    peer.pause_track("voice")
    peer._push_bundle(_two_audio_bundle())
    peer._push_audio_tick()

    peer.resume_track("voice")
    peer._push_bundle(_two_audio_bundle())
    peer._push_audio_tick()

    assert len(voice.pushed) == 2
    assert voice.pushed[1][0] != _AUDIO_SILENT_FRAME
    assert len(music.pushed) == 2


def test_the_under_production_warning_names_the_starved_track(
    caplog: pytest.LogCaptureFixture,
) -> None:
    peer, _, _ = _two_audio_peer()
    peer._last_silence_sample = (time.monotonic() - 1.0, {"voice": 0, "music": 0})
    peer._silence_frames = {"voice": 20, "music": 0}

    with caplog.at_level(logging.WARNING):
        peer._warn_on_audio_underrun()

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "'voice'" in warnings[0]


def test_the_feeder_leaves_the_video_tracks_alone() -> None:
    """The outbound tracks hold both kinds; only the audio ones are fed."""
    peer, video, audio = _wired_peer()
    _make_live(peer, audio)

    peer._push_audio_tick()

    assert video.pushed == []
    assert len(audio.pushed) == 1
    assert peer._silence_frames == {"a": 1}


def test_media_health_totals_the_silence_of_every_track() -> None:
    peer, _, _ = _two_audio_peer()
    peer._silence_frames = {"voice": 12, "music": 5}

    assert peer._media_health().silence_frames == 17


def test_the_feeder_ignores_a_pause_on_a_track_that_is_not_its_own() -> None:
    peer, _, audio = _wired_peer()

    peer.pause_track("v")
    peer._push_bundle(_av_bundle())
    peer._push_audio_tick()

    assert len(audio.pushed) == 1


def test_audio_feed_loop_keeps_the_wire_fed_through_an_empty_buffer() -> None:
    peer, track = _audio_peer()
    _make_live(peer, track)
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


# ── Trickle ICE ──────────────────────────────────────────────────────────────


class _IceRecordingPc:
    def __init__(self) -> None:
        self.added: list[Any] = []

    async def add_ice_candidate(self, candidate: Any) -> None:
        self.added.append(candidate)


async def test_add_ice_forwards_a_real_candidate() -> None:
    peer = WebRTCPeer()
    pc = _IceRecordingPc()
    peer._pc = cast(Any, pc)

    candidate = IceCandidate("candidate:1 1 udp 2122260223 192.0.2.1 50000 typ host", "0", 0)
    await peer.add_ice(candidate)

    assert len(pc.added) == 1
    assert pc.added[0].candidate == candidate.candidate


async def test_add_ice_forwards_the_end_of_candidates_marker() -> None:
    """The empty marker reaches the binding, which accepts it as a no-op."""
    peer = WebRTCPeer()
    pc = _IceRecordingPc()
    peer._pc = cast(Any, pc)

    await peer.add_ice(IceCandidate(""))

    assert len(pc.added) == 1
    assert pc.added[0].candidate == ""


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


# ── Video codec preferences ──────────────────────────────────────────────────


def test_video_codec_preferences_maps_known_names_in_order() -> None:
    config = WebRtcConfig(
        supported_video_codecs=({"codec": "VP9"}, {"codec": "H264"}, {"codec": "AV1"})
    )
    assert _video_codec_preferences(config) == [
        rw.VideoCodec.Vp9,
        rw.VideoCodec.H264,
        rw.VideoCodec.Av1,
    ]


def test_video_codec_preferences_skips_names_the_binding_does_not_recognize() -> None:
    config = WebRtcConfig(supported_video_codecs=({"codec": "VP8"}, {"codec": "Theora"}))
    assert _video_codec_preferences(config) == [rw.VideoCodec.Vp8]


def test_video_codec_preferences_defaults_to_the_documented_order() -> None:
    assert _video_codec_preferences(WebRtcConfig()) == [
        rw.VideoCodec.Vp9,
        rw.VideoCodec.Vp8,
        rw.VideoCodec.H264,
        rw.VideoCodec.Av1,
        rw.VideoCodec.H265,
    ]


class _FakeTransceiver:
    def __init__(self, kind: Any, mid: str | None) -> None:
        self._kind = kind
        self._mid = mid
        self.codec_preference_calls: list[list[Any]] = []
        self.track_calls: list[Any] = []
        self.direction_calls: list[Any] = []

    def kind(self) -> Any:
        return self._kind

    def mid(self) -> str | None:
        return self._mid

    async def set_codec_preferences(self, codecs: list[Any]) -> None:
        self.codec_preference_calls.append(list(codecs))

    async def set_track(self, track: Any) -> None:
        self.track_calls.append(track)

    async def set_direction(self, direction: Any) -> None:
        self.direction_calls.append(direction)


class _FakeTransceiverPeerConnection:
    def __init__(self, transceivers: list[_FakeTransceiver]) -> None:
        self._transceivers = transceivers

    async def transceivers(self) -> list[_FakeTransceiver]:
        return self._transceivers


class _FakeTrackFactory:
    def create_video_track(self, name: str) -> Any:
        return SimpleNamespace(name=name)

    def create_audio_track_with_local_source(self, name: str) -> Any:
        return SimpleNamespace(name=name)


async def test_attach_out_tracks_applies_codec_preferences_to_every_video_transceiver() -> None:
    peer = WebRTCPeer()
    peer._config = WebRtcConfig(supported_video_codecs=({"codec": "VP9"}, {"codec": "VP8"}))
    peer._track_by_mid = {
        "0": TrackInfo(name="cam", kind=TrackKind.VIDEO, direction=TrackDirection.OUT),
    }
    out_video = _FakeTransceiver(rw.MediaKind.Video, "0")
    in_video = _FakeTransceiver(rw.MediaKind.Video, "1")  # recvonly: not in _track_by_mid as OUT
    audio = _FakeTransceiver(rw.MediaKind.Audio, "2")
    pc: Any = _FakeTransceiverPeerConnection([out_video, in_video, audio])

    await peer._attach_out_tracks(pc, cast("Any", _FakeTrackFactory()))

    expected = [rw.VideoCodec.Vp9, rw.VideoCodec.Vp8]
    assert out_video.codec_preference_calls == [expected]
    assert in_video.codec_preference_calls == [expected]
    assert audio.codec_preference_calls == []
    assert out_video.track_calls  # the OUT mid still gets its sender track
    assert in_video.track_calls == []  # a receiving transceiver is untouched otherwise


async def test_attach_out_tracks_skips_set_codec_preferences_when_none_configured() -> None:
    peer = WebRTCPeer()
    peer._config = WebRtcConfig(supported_video_codecs=({"codec": "Theora"},))
    video = _FakeTransceiver(rw.MediaKind.Video, None)
    pc: Any = _FakeTransceiverPeerConnection([video])

    await peer._attach_out_tracks(pc, cast("Any", _FakeTrackFactory()))

    assert video.codec_preference_calls == []


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
    peer._silence_frames = {"a": 12}
    peer._dropped_samples = 480
    peer._dropped_bundles = 2
    report: Any = SimpleNamespace(outbound_rtp=[], inbound_rtp=[], candidate_pairs=[])

    media = peer._stats_from_report(report).media

    assert media.silence_frames == 12
    assert media.dropped_samples == 480
    assert media.dropped_bundles == 2


async def test_stats_on_a_closed_peer_still_reports_media_counters() -> None:
    peer = WebRTCPeer()
    peer._silence_frames = {"a": 3}

    stats = await peer.stats()

    assert stats.media.silence_frames == 3


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_underrun_warning_needs_a_baseline_sample(caplog: pytest.LogCaptureFixture) -> None:
    peer = WebRTCPeer()
    peer._silence_frames = {"a": 10_000}

    with caplog.at_level(logging.WARNING):
        peer._warn_on_audio_underrun()

    assert _warnings(caplog) == []


def test_underrun_warning_fires_once_silence_dominates_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    peer = WebRTCPeer()
    # A second of frames, a fifth of it silence the runtime inserted.
    peer._last_silence_sample = (time.monotonic() - 1.0, {"a": 0})
    peer._silence_frames = {"a": 20}

    with caplog.at_level(logging.WARNING):
        peer._warn_on_audio_underrun()

    assert len(_warnings(caplog)) == 1
    assert "below real time" in _warnings(caplog)[0]


def test_underrun_warning_stays_quiet_for_an_occasional_gap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    peer = WebRTCPeer()
    peer._last_silence_sample = (time.monotonic() - 1.0, {"a": 0})
    peer._silence_frames = {"a": 2}

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


def test_the_clock_never_stops_while_the_track_is_on_the_wire() -> None:
    """A feeder that stops leaves the sender reports dating the audio too early.

    libwebrtc estimates a report's RTP timestamp as the last frame's plus the
    time since it, while the packets only advance by samples actually sent. Stop
    for five seconds and the two disagree by five seconds; the client dates the
    audio that follows from the reports and plays it ahead of the video.
    """
    peer, track = _audio_peer()

    peer.pause_track("a")
    for _ in range(200):
        peer._push_audio_tick()
    peer.resume_track("a")
    for _ in range(200):  # long past the idle grace
        peer._push_audio_tick()

    assert len(track.pushed) == 400  # one frame per tick, throughout


# ── Pause as a local renegotiation ───────────────────────────────────────────

_SESSION_OFFER = (
    "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:0\r\na=recvonly\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:1\r\na=recvonly\r\n"
)
_SESSION_ANSWER = (
    "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:0\r\na=sendonly\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:1\r\na=sendonly\r\n"
)


class _RecordingTransceiver:
    def __init__(self, mid: str, kind: Any = rw.MediaKind.Audio) -> None:
        self._mid = mid
        self._kind = kind
        self.tracks: list[Any] = []
        self.directions: list[Any] = []
        self.codec_preferences: list[Any] = []

    def mid(self) -> str:
        return self._mid

    def kind(self) -> Any:
        return self._kind

    async def set_codec_preferences(self, codecs: Any) -> None:
        self.codec_preferences.append(codecs)

    async def set_track(self, track: Any) -> None:
        self.tracks.append(track)

    async def set_direction(self, direction: Any) -> None:
        self.directions.append(direction)


class _RecordingPc:
    def __init__(self) -> None:
        self.applied: list[tuple[str, str]] = []
        self.transceiver_list = [_RecordingTransceiver("0"), _RecordingTransceiver("1")]

    async def set_remote_description(self, sdp: Any) -> None:
        self.applied.append(("remote", sdp.sdp))

    async def set_local_description(self, sdp: Any) -> None:
        self.applied.append(("local", sdp.sdp))

    async def transceivers(self) -> list[_RecordingTransceiver]:
        return self.transceiver_list


def _negotiated_peer() -> tuple[WebRTCPeer, _RecordingPc]:
    peer = WebRTCPeer()
    pc = _RecordingPc()
    peer._pc = cast(Any, pc)
    peer._offer_sdp = _SESSION_OFFER
    peer._answer_sdp = _SESSION_ANSWER
    peer._applied_pauses = frozenset()
    peer._track_by_mid = {
        "0": TrackInfo(name="v", kind=TrackKind.VIDEO, direction=TrackDirection.OUT),
        "1": TrackInfo(name="a", kind=TrackKind.AUDIO, direction=TrackDirection.OUT),
    }
    peer._out_tracks["v"] = cast(Any, _FakeTrack())
    peer._out_tracks["a"] = cast(Any, _FakeAudioTrack())
    return peer, pc


async def test_pausing_takes_only_that_section_out_of_the_session() -> None:
    peer, pc = _negotiated_peer()

    peer.pause_track("a")
    await peer._apply_pauses()

    remote, local = pc.applied[0][1], pc.applied[1][1]
    assert "a=mid:1\r\na=inactive" in remote
    assert "a=mid:1\r\na=inactive" in local
    assert "a=mid:0\r\na=recvonly" in remote  # the video section stays up
    assert "a=mid:0\r\na=sendonly" in local


async def test_resuming_one_of_two_paused_tracks_leaves_the_other_paused() -> None:
    """Every section is written on every pass, or a resume forgets the rest."""
    peer, pc = _negotiated_peer()
    peer.pause_track("a")
    await peer._apply_pauses()
    peer.pause_track("v")
    await peer._apply_pauses()
    pc.applied.clear()

    peer.resume_track("a")
    await peer._apply_pauses()

    remote, local = pc.applied[0][1], pc.applied[1][1]
    assert "a=mid:1\r\na=recvonly" in remote  # resumed
    assert "a=mid:1\r\na=sendonly" in local
    assert "a=mid:0\r\na=inactive" in remote  # still paused
    assert "a=mid:0\r\na=inactive" in local


async def test_a_claim_on_a_track_that_is_not_paused_renegotiates_nothing() -> None:
    """publish_track resumes on claim; renegotiating there restarts the session."""
    peer, pc = _negotiated_peer()

    peer.resume_track("a")
    await peer._apply_pauses()

    assert pc.applied == []


async def test_pausing_puts_the_tracks_back_on_their_transceivers() -> None:
    """A sender that comes out of a re-apply without its track sends nothing."""
    peer, pc = _negotiated_peer()

    peer.pause_track("a")
    await peer._apply_pauses()

    assert pc.transceiver_list[0].tracks == [peer._out_tracks["v"]]
    assert pc.transceiver_list[1].tracks == [peer._out_tracks["a"]]


async def test_the_transceiver_direction_follows_the_pause() -> None:
    """libwebrtc answers from the transceiver, not from the SDP string."""
    peer, pc = _negotiated_peer()

    peer.pause_track("a")
    await peer._apply_pauses()

    assert pc.transceiver_list[0].directions == [rw.TransceiverDirection.SendOnly]
    assert pc.transceiver_list[1].directions == [rw.TransceiverDirection.Inactive]


async def test_the_transceivers_are_restated_before_the_local_description() -> None:
    """The order negotiation itself uses: remote, transceivers, then local."""
    order: list[str] = []

    class _OrderedTransceiver(_RecordingTransceiver):
        async def set_direction(self, direction: Any) -> None:
            order.append("direction")

    class _OrderedPc(_RecordingPc):
        def __init__(self) -> None:
            super().__init__()
            self.transceiver_list = [_OrderedTransceiver("0"), _OrderedTransceiver("1")]

        async def set_remote_description(self, sdp: Any) -> None:
            order.append("remote")

        async def set_local_description(self, sdp: Any) -> None:
            order.append("local")

    peer, _ = _negotiated_peer()
    peer._pc = cast(Any, _OrderedPc())

    peer.pause_track("a")
    await peer._apply_pauses()

    assert order[0] == "remote"
    assert order[-1] == "local"
    assert "direction" in order[1:-1]


async def test_a_failed_apply_leaves_the_session_where_it_was() -> None:
    peer, _ = _negotiated_peer()

    class _Rejecting:
        async def set_remote_description(self, sdp: Any) -> None:
            raise RuntimeError("rejected")

    peer._pc = cast(Any, _Rejecting())
    peer.pause_track("a")

    await peer._apply_pauses()

    assert peer._applied_pauses == frozenset()  # not recorded as applied
    assert peer._offer_sdp == _SESSION_OFFER  # the base is never mutated


async def test_the_base_descriptions_are_never_edited() -> None:
    peer, _ = _negotiated_peer()

    peer.pause_track("a")
    await peer._apply_pauses()
    peer.resume_track("a")
    await peer._apply_pauses()

    assert peer._offer_sdp == _SESSION_OFFER
    assert peer._answer_sdp == _SESSION_ANSWER


async def test_outbound_tracks_negotiate_as_sending() -> None:
    """A joining client reads a normal session; the pause lands behind it."""
    peer = WebRTCPeer()
    peer._track_by_mid = {
        "0": TrackInfo(name="v", kind=TrackKind.VIDEO, direction=TrackDirection.OUT),
        "1": TrackInfo(name="a", kind=TrackKind.AUDIO, direction=TrackDirection.OUT),
        "2": TrackInfo(name="cam", kind=TrackKind.VIDEO, direction=TrackDirection.IN),
    }

    class _Factory:
        def create_video_track(self, name: str) -> Any:
            return _FakeTrack()

        def create_audio_track_with_local_source(self, name: str) -> Any:
            return _FakeAudioTrack()

    pc = _RecordingPc()
    pc.transceiver_list = [
        _RecordingTransceiver("0", rw.MediaKind.Video),
        _RecordingTransceiver("1"),
        _RecordingTransceiver("2", rw.MediaKind.Video),
    ]

    await peer._attach_out_tracks(cast(Any, pc), cast(Any, _Factory()))

    assert pc.transceiver_list[0].directions == [rw.TransceiverDirection.SendOnly]
    assert pc.transceiver_list[1].directions == [rw.TransceiverDirection.SendOnly]
    assert pc.transceiver_list[2].directions == []


async def test_a_first_subscription_leaves_the_others_out() -> None:
    """Everything is paused after negotiation; a resume brings back one track."""
    peer, pc = _negotiated_peer()
    peer._paused_tracks = {"v", "a"}
    peer._applied_pauses = frozenset({"v", "a"})

    peer.resume_track("a")
    await peer._apply_pauses()

    remote, local = pc.applied[0][1], pc.applied[1][1]
    assert "a=mid:1\r\na=recvonly" in remote  # asked for
    assert "a=mid:1\r\na=sendonly" in local
    assert "a=mid:0\r\na=inactive" in remote  # not asked for
    assert "a=mid:0\r\na=inactive" in local
