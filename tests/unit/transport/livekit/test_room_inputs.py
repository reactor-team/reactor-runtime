import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest
from livekit import rtc

from reactor_runtime.core import ConnId, InputFrame, TrackDirection, TrackInfo, TrackKind
from reactor_runtime.transport.livekit.media_in import (
    AUDIO_QUEUE_FRAMES,
    MAX_OPEN_INPUT_STREAMS,
    VIDEO_QUEUE_FRAMES,
    RoomInputs,
    StreamFramer,
)

AGENT = TrackInfo(
    name="agent_audio", kind=TrackKind.AUDIO, rate=24_000, direction=TrackDirection.IN
)
CAMERA = TrackInfo(name="camera", kind=TrackKind.VIDEO, direction=TrackDirection.IN)
MIC = TrackInfo(name="mic", kind=TrackKind.AUDIO, rate=48_000, direction=TrackDirection.IN)


def pcm(samples: np.ndarray) -> bytes:
    return samples.astype("<i2").tobytes()


# --- framing a stream's PCM ------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 480, 1000, 15_000])
def test_a_stream_at_the_tracks_rate_is_cut_sample_for_sample(chunk: int) -> None:
    samples = (np.arange(24_000) % 20_000 - 10_000).astype(np.int16)
    data = pcm(samples)
    framer = StreamFramer(24_000, 24_000, 1)

    frames = [f for i in range(0, len(data), chunk) for f in framer.push(data[i : i + chunk])]
    frames += framer.flush()

    assert all(f.shape == (1, 240) for f in frames)
    assert np.array_equal(np.concatenate(frames, axis=1)[0], samples)


def test_the_last_frame_of_a_stream_is_what_is_left() -> None:
    framer = StreamFramer(24_000, 24_000, 1)
    framer.push(pcm(np.ones(500, np.int16)))

    (last,) = framer.flush()

    assert last.shape == (1, 20)


def test_a_stereo_stream_is_mixed_down_to_mono() -> None:
    left = np.full(240, 1000, np.int16)
    right = np.full(240, 3000, np.int16)
    framer = StreamFramer(24_000, 24_000, 2)

    (frame,) = framer.push(pcm(np.stack([left, right], axis=1).reshape(-1)))

    assert frame.shape == (1, 240)
    assert (frame == 2000).all()


def test_a_stream_at_another_rate_is_resampled_to_the_tracks() -> None:
    t = np.arange(48_000) / 48_000
    tone = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    framer = StreamFramer(24_000, 48_000, 1)

    frames = framer.push(pcm(tone)) + framer.flush()
    out = np.concatenate(frames, axis=1)[0]

    assert abs(out.size - 24_000) <= 240
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
    assert abs(np.fft.rfftfreq(out.size, 1 / 24_000)[spectrum.argmax()] - 440) < 3


# --- feeding input tracks ---------------------------------------------------


@dataclass
class Info:
    stream_id: str
    topic: str = "lk.audio_stream"
    attributes: dict[str, str] = field(
        default_factory=lambda: {"sample_rate": "24000", "num_channels": "1"}
    )


class Reader:
    """A byte stream that yields its chunks once released."""

    def __init__(self, stream_id: str, chunks: list[bytes], **info: Any) -> None:
        self.info = Info(stream_id, **info)
        self._chunks = chunks
        self.released = asyncio.Event()
        self.read: list[bytes] = []
        self.closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[bytes]:
        await self.released.wait()
        for chunk in self._chunks:
            # A closed reader ends, as the SDK's does.
            if self.closed:
                return
            self.read.append(chunk)
            yield chunk

    def close(self) -> None:
        self.closed = True


class Sink:
    def __init__(self) -> None:
        self.frames: list[tuple[ConnId, str, InputFrame]] = []

    def __call__(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        self.frames.append((conn_id, track, frame))

    def metadata(self) -> list[dict[str, Any]]:
        return [json.loads(f.metadata or b"{}") for _, _, f in self.frames]


class Claims:
    """The session's input arbitration: one holder per input track, first come."""

    def __init__(self) -> None:
        self.holders: dict[str, ConnId] = {}

    def claim(self, conn_id: ConnId, name: str) -> bool:
        if name in self.holders:
            return False
        self.holders[name] = conn_id
        return True

    def release(self, conn_id: ConnId, name: str) -> None:
        if self.holders.get(name) == conn_id:
            del self.holders[name]

    def connection_closed(self, conn_id: ConnId) -> None:
        for name in [n for n, holder in self.holders.items() if holder == conn_id]:
            del self.holders[name]


def inputs(sink: Sink, claims: Claims | None = None) -> RoomInputs:
    claims = claims or Claims()
    return RoomInputs(
        {"agent_audio": AGENT, "camera": CAMERA, "mic": MIC},
        {"lk.audio_stream": "agent_audio", "lk.camera_stream": "camera", "reactor.data": "mic"},
        sink,
        claims.claim,
        claims.release,
    )


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def test_a_stream_feeds_its_track_and_ends_with_an_empty_frame() -> None:
    sink = Sink()
    room = inputs(sink)
    reader = Reader("AUDIO_1", [pcm(np.ones(480, np.int16))])

    room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))
    reader.released.set()
    await _settle()

    assert [(c, t, f.data.shape) for c, t, f in sink.frames] == [
        (5120, "agent_audio", (1, 240)),
        (5120, "agent_audio", (1, 240)),
        (5120, "agent_audio", (1, 0)),
    ]
    attributes = {"sample_rate": "24000", "num_channels": "1"}
    assert sink.metadata() == [
        {"stream_id": "AUDIO_1", "attributes": attributes},
        {"stream_id": "AUDIO_1", "attributes": attributes},
        {"stream_id": "AUDIO_1", "ended": True},
    ]


class LiveReader:
    """A byte stream whose chunks arrive one at a time, as the test sends them."""

    def __init__(self, stream_id: str, **info: Any) -> None:
        self.info = Info(stream_id, **info)
        self._chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.closed = False

    def send(self, chunk: bytes) -> None:
        self._chunks.put_nowait(chunk)

    def finish(self) -> None:
        self._chunks.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[bytes]:
        while not self.closed and (chunk := await self._chunks.get()) is not None:
            yield chunk

    def close(self) -> None:
        self.closed = True


def frames_by_stream(sink: Sink) -> list[tuple[str, int | str]]:
    """Each frame as its stream id and first sample, or "end" for an end frame."""
    out: list[tuple[str, int | str]] = []
    for (_, _, frame), meta in zip(sink.frames, sink.metadata(), strict=True):
        out.append((meta["stream_id"], "end" if meta.get("ended") else int(frame.data[0, 0])))
    return out


async def test_a_new_stream_on_the_topic_leaves_the_open_one_feeding() -> None:
    sink = Sink()
    room = inputs(sink)
    first, second = LiveReader("AUDIO_1"), LiveReader("AUDIO_2")
    room.on_byte_stream(cast(Any, first), "agent", ConnId(5120))
    first.send(pcm(np.full(240, 1, np.int16)))
    await _settle()

    room.on_byte_stream(cast(Any, second), "agent", ConnId(5120))
    second.send(pcm(np.full(240, 2, np.int16)))
    await _settle()
    first.send(pcm(np.full(240, 3, np.int16)))
    first.finish()
    await _settle()
    second.send(pcm(np.full(240, 4, np.int16)))
    second.finish()
    await _settle()

    assert frames_by_stream(sink) == [
        ("AUDIO_1", 1),
        ("AUDIO_2", 2),
        ("AUDIO_1", 3),
        ("AUDIO_1", "end"),
        ("AUDIO_2", 4),
        ("AUDIO_2", "end"),
    ]


async def test_a_stream_still_queued_when_the_next_opens_is_read_to_its_end() -> None:
    sink = Sink()
    room = inputs(sink)
    first = Reader("AUDIO_1", [pcm(np.full(240, 1, np.int16)), pcm(np.full(240, 2, np.int16))])
    room.on_byte_stream(cast(Any, first), "agent", ConnId(5120))
    first.released.set()
    second = Reader("AUDIO_2", [pcm(np.full(240, 3, np.int16))])

    # The first stream's chunks and end have arrived, but nothing has read them yet.
    room.on_byte_stream(cast(Any, second), "agent", ConnId(5120))
    second.released.set()
    await _settle()

    assert frames_by_stream(sink) == [
        ("AUDIO_1", 1),
        ("AUDIO_1", 2),
        ("AUDIO_1", "end"),
        ("AUDIO_2", 3),
        ("AUDIO_2", "end"),
    ]


async def test_a_stream_past_the_open_bound_is_refused_and_the_open_ones_go_on() -> None:
    sink = Sink()
    room = inputs(sink)
    readers = [LiveReader(f"AUDIO_{i}") for i in range(MAX_OPEN_INPUT_STREAMS + 1)]
    for reader in readers:
        room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))
    for i, reader in enumerate(readers):
        reader.send(pcm(np.full(240, i, np.int16)))
    await _settle()

    refused = readers[-1]
    assert refused.closed
    assert frames_by_stream(sink) == [(f"AUDIO_{i}", i) for i in range(MAX_OPEN_INPUT_STREAMS)]


async def test_a_stream_ends_once_however_many_ways_it_is_ended() -> None:
    sink = Sink()
    room = inputs(sink)
    reader = LiveReader("AUDIO_1")
    room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))

    room.release("agent", ConnId(5120))
    reader.finish()
    await _settle()
    room.close()

    assert frames_by_stream(sink) == [("AUDIO_1", "end")]


async def test_the_first_participant_to_send_holds_the_track() -> None:
    sink = Sink()
    room = inputs(sink)
    held = Reader("AUDIO_1", [pcm(np.ones(240, np.int16))])
    other = Reader("AUDIO_2", [pcm(np.ones(240, np.int16))])

    room.on_byte_stream(cast(Any, held), "agent", ConnId(5120))
    room.on_byte_stream(cast(Any, other), "intruder", ConnId(10_001))
    held.released.set()
    other.released.set()
    await _settle()

    assert {m["stream_id"] for m in sink.metadata()} == {"AUDIO_1"}
    assert other.closed
    assert other.read == []


async def test_a_participant_that_leaves_ends_its_streams_and_frees_the_track() -> None:
    sink = Sink()
    claims = Claims()
    room = inputs(sink, claims)
    stuck = [LiveReader("AUDIO_1"), LiveReader("AUDIO_2")]
    for reader in stuck:
        room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))

    room.release("agent", ConnId(5120))
    claims.connection_closed(ConnId(5120))
    nxt = Reader("AUDIO_3", [pcm(np.ones(240, np.int16))])
    room.on_byte_stream(cast(Any, nxt), "newcomer", ConnId(10_001))
    nxt.released.set()
    await _settle()

    assert [(m["stream_id"], m.get("ended", False)) for m in sink.metadata()] == [
        ("AUDIO_1", True),
        ("AUDIO_2", True),
        ("AUDIO_3", False),
        ("AUDIO_3", True),
    ]
    assert all(reader.closed for reader in stuck)


async def test_a_stream_mapped_to_a_video_track_is_refused() -> None:
    sink = Sink()
    room = inputs(sink)
    reader = Reader("VIDEO_1", [b"\x00" * 100], topic="lk.camera_stream")

    room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))
    reader.released.set()
    await _settle()

    assert sink.frames == []
    assert reader.closed


@pytest.mark.parametrize(
    "attributes",
    [
        {"sample_rate": "0", "num_channels": "1"},
        {"sample_rate": "24k", "num_channels": "1"},
        {"sample_rate": "1000000", "num_channels": "1"},
        {"sample_rate": "24000", "num_channels": "0"},
        {"sample_rate": "24000", "num_channels": "4096"},
    ],
)
async def test_a_stream_with_an_unusable_format_is_refused_without_holding_the_track(
    attributes: dict[str, str],
) -> None:
    sink = Sink()
    room = inputs(sink)
    bad = Reader("AUDIO_1", [pcm(np.ones(240, np.int16))], attributes=attributes)
    good = Reader("AUDIO_2", [pcm(np.ones(240, np.int16))])

    room.on_byte_stream(cast(Any, bad), "agent", ConnId(5120))
    room.on_byte_stream(cast(Any, good), "other", ConnId(10_001))
    bad.released.set()
    good.released.set()
    await _settle()

    assert bad.closed
    assert {m["stream_id"] for m in sink.metadata()} == {"AUDIO_2"}


async def test_a_stream_without_format_attributes_is_taken_at_the_tracks_rate_in_mono() -> None:
    sink = Sink()
    room = inputs(sink)
    reader = Reader("AUDIO_1", [pcm(np.ones(480, np.int16))], attributes={})

    room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))
    reader.released.set()
    await _settle()

    assert [f.data.shape for _, _, f in sink.frames] == [(1, 240), (1, 240), (1, 0)]


def test_stream_samples_are_read_as_little_endian() -> None:
    framer = StreamFramer(rate=200, stream_rate=200, channels=1)

    (frame,) = framer.push(b"\x01\x00\x00\x01")

    assert frame.tolist() == [[1, 256]]


def test_a_protocol_topic_cannot_feed_an_input_track() -> None:
    assert inputs(Sink()).stream_topics == ["lk.audio_stream", "lk.camera_stream"]


@dataclass
class Track:
    name: str
    kind: int
    sid: str = "TR_1"


async def test_a_track_named_after_an_input_feeds_it_unless_the_input_is_reserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[tuple[str, str]] = []

    async def read_audio(self: Any, track: Any, target: str, rate: int, conn_id: Any) -> None:
        started.append((track.name, target))

    async def read_video(self: Any, track: Any, target: str, conn_id: Any) -> None:
        started.append((track.name, target))

    monkeypatch.setattr(RoomInputs, "_read_audio", read_audio)
    monkeypatch.setattr(RoomInputs, "_read_video", read_video)
    room = inputs(Sink())

    room.on_track(cast(Any, Track("mic", rtc.TrackKind.KIND_AUDIO)), "viewer", ConnId(5120))
    # agent_audio is fed only by its mapped stream.
    room.on_track(cast(Any, Track("agent_audio", rtc.TrackKind.KIND_AUDIO)), "viewer", ConnId(1))
    # A kind that does not match the input track feeds nothing.
    room.on_track(cast(Any, Track("camera", rtc.TrackKind.KIND_AUDIO)), "viewer", ConnId(1))
    await _settle()

    assert started == [("mic", "mic")]


_streams: list["FakeMediaStream"] = []


class FakeMediaStream:
    """Stands in for ``rtc.VideoStream`` and ``rtc.AudioStream``: never yields, records closing."""

    def __init__(self, track: Any, *, capacity: int = 0, **_: Any) -> None:
        self.capacity = capacity
        self.closed = False
        _streams.append(self)

    def __aiter__(self) -> "FakeMediaStream":
        return self

    async def __anext__(self) -> Any:
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


async def test_a_remote_tracks_queue_is_bounded_and_closed_when_its_sender_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _streams.clear()
    monkeypatch.setattr(rtc, "VideoStream", FakeMediaStream)
    monkeypatch.setattr(rtc, "AudioStream", FakeMediaStream)
    claims = Claims()
    room = RoomInputs({"camera": CAMERA, "mic": MIC}, {}, Sink(), claims.claim, claims.release)

    room.on_track(cast(Any, Track("mic", rtc.TrackKind.KIND_AUDIO, "TR_a")), "viewer", ConnId(5120))
    room.on_track(
        cast(Any, Track("camera", rtc.TrackKind.KIND_VIDEO, "TR_v")), "viewer", ConnId(5120)
    )
    await _settle()
    room.release("viewer", ConnId(5120))
    await _settle()

    audio, video = _streams
    assert (audio.capacity, video.capacity) == (AUDIO_QUEUE_FRAMES, VIDEO_QUEUE_FRAMES)
    assert audio.closed
    assert video.closed


def _cameras(monkeypatch: pytest.MonkeyPatch) -> tuple[RoomInputs, Claims, list[ConnId]]:
    started: list[ConnId] = []

    async def read_video(self: Any, track: Any, target: str, conn_id: ConnId) -> None:
        started.append(conn_id)
        await asyncio.Event().wait()

    monkeypatch.setattr(RoomInputs, "_read_video", read_video)
    claims = Claims()
    room = RoomInputs({"camera": CAMERA}, {}, Sink(), claims.claim, claims.release)
    return room, claims, started


def _camera(sid: str) -> Any:
    return cast(Any, Track("camera", rtc.TrackKind.KIND_VIDEO, sid))


async def test_a_refused_track_starts_feeding_once_the_holders_track_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room, claims, started = _cameras(monkeypatch)
    room.on_track(_camera("TR_first"), "first", ConnId(5120))
    room.on_track(_camera("TR_second"), "second", ConnId(5121))
    await _settle()
    assert started == [ConnId(5120)]

    room.on_track_gone("TR_first")
    await _settle()

    assert started == [ConnId(5120), ConnId(5121)]
    assert claims.holders == {"camera": ConnId(5121)}
    room.close()


async def test_a_refused_track_starts_feeding_once_the_holder_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room, claims, started = _cameras(monkeypatch)
    room.on_track(_camera("TR_first"), "first", ConnId(5120))
    room.on_track(_camera("TR_second"), "second", ConnId(5121))
    await _settle()

    room.release("first", ConnId(5120))
    await _settle()

    assert started == [ConnId(5120), ConnId(5121)]
    assert claims.holders == {"camera": ConnId(5121)}
    room.close()


async def test_a_refused_track_starts_once_another_transport_frees_the_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room, claims, started = _cameras(monkeypatch)
    claims.claim(ConnId(4001), "camera")  # a WebRTC client published it first
    room.on_track(_camera("TR_room"), "agent", ConnId(5120))
    await _settle()
    assert started == []

    claims.release(ConnId(4001), "camera")
    room.input_freed("camera")
    await _settle()

    assert started == [ConnId(5120)]
    assert claims.holders == {"camera": ConnId(5120)}
    room.close()


async def test_a_refused_track_that_goes_away_is_not_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room, claims, started = _cameras(monkeypatch)
    room.on_track(_camera("TR_first"), "first", ConnId(5120))
    room.on_track(_camera("TR_second"), "second", ConnId(5121))
    await _settle()
    room.on_track_gone("TR_second")

    room.on_track_gone("TR_first")
    await _settle()

    assert started == [ConnId(5120)]
    assert claims.holders == {}
    room.close()


async def test_a_claim_a_stream_took_outlasts_a_track_of_the_same_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def read_audio(self: Any, track: Any, target: str, rate: int, conn_id: ConnId) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(RoomInputs, "_read_audio", read_audio)
    claims = Claims()
    room = inputs(Sink(), claims)
    reader = Reader("AUDIO_1", [pcm(np.ones(240, np.int16))])
    room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))
    reader.released.set()
    await _settle()  # the utterance has ended
    room.on_track(
        cast(Any, Track("lk.audio_stream", rtc.TrackKind.KIND_AUDIO, "TR_a")), "agent", ConnId(5120)
    )

    room.on_track_gone("TR_a")

    assert claims.holders == {"agent_audio": ConnId(5120)}
    room.close()


async def test_an_input_another_transport_holds_is_not_fed_from_the_room() -> None:
    sink = Sink()
    claims = Claims()
    claims.claim(ConnId(4001), "agent_audio")  # a WebRTC client published it first
    room = inputs(sink, claims)
    reader = Reader("AUDIO_1", [pcm(np.ones(240, np.int16))])

    room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))
    reader.released.set()
    await _settle()

    assert reader.closed
    assert sink.frames == []


async def test_closing_ends_each_open_stream_and_closes_readers_that_never_ran() -> None:
    sink = Sink()
    room = inputs(sink)
    readers = [Reader(f"AUDIO_{i}", [pcm(np.ones(240, np.int16))]) for i in (1, 2)]
    for reader in readers:
        room.on_byte_stream(cast(Any, reader), "agent", ConnId(5120))

    room.close()

    assert all(reader.closed for reader in readers)
    assert sink.metadata() == [
        {"stream_id": "AUDIO_1", "ended": True},
        {"stream_id": "AUDIO_2", "ended": True},
    ]
