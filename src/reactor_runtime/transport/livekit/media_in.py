"""The model's input tracks, fed from the room.

A participant can send media to the session in two ways, and each feeds the
model's declared input tracks:

- **As a track.** A remote track published under a declared input track's name
  feeds that track, which needs the subscribe grant in the session's token.
- **As a byte stream.** A key of the manifest's ``media_inputs`` names a byte
  stream topic, or a remote track name, and the declared input track it feeds.
  A track named there is reserved: only the mapped source feeds it.

A byte stream carries ``pcm_s16le`` audio and has a start and an end. It is cut
into 10 ms frames at the track's declared rate, in mono, using the stream's
``sample_rate`` and ``num_channels`` attributes. Each frame's ``metadata``
names the stream, as JSON::

    {"stream_id": "AUDIO_x7Kq", "attributes": {"sample_rate": "24000", "num_channels": "1"}}

and after its last frame comes one with no samples, ``data`` of shape
``(1, 0)``, and ``{"stream_id": "AUDIO_x7Kq", "ended": true}``. A stream ends when
it closes, when a new stream starts on the same topic while it is open, or when
its sender leaves. A stream without the attributes is taken to be at the track's
rate, in mono; one whose ``sample_rate`` is not between
:data:`MIN_STREAM_RATE` and :data:`MAX_STREAM_RATE`, or whose ``num_channels``
is not between 1 and :data:`MAX_STREAM_CHANNELS`, is refused.

Each input track has one owner across the session, whatever its transport: the
first client to feed it holds it, as a WebRTC client's publish request does,
and media from anyone else is dropped. A participant feeding a track holds it
until it unpublishes that track or leaves; one that has fed a byte stream holds
it until it leaves, since an agent sends one stream per utterance. A remote
track refused because another participant holds its input waits, and starts
feeding the input once that participant frees it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from livekit import rtc

from reactor_runtime.core import ConnId, InputFrame, TrackInfo, TrackKind
from reactor_runtime.log import get_logger
from reactor_runtime.transport.livekit.participant import TOPIC_CHANNELS

logger = get_logger(__name__)

FRAME_SECONDS = 0.01
"""The length of each audio frame cut from a byte stream."""

MIN_STREAM_RATE = 8_000
"""The lowest ``sample_rate`` a byte stream may declare."""

MAX_STREAM_RATE = 192_000
"""The highest ``sample_rate`` a byte stream may declare."""

MAX_STREAM_CHANNELS = 8
"""The most ``num_channels`` a byte stream may declare."""

VIDEO_QUEUE_FRAMES = 4
"""How many decoded frames a remote video track may queue; past it, the oldest is dropped.

The model reads video for its latest picture, so a short queue loses nothing it uses.
"""

AUDIO_QUEUE_FRAMES = 100
"""How many 10 ms frames a remote audio track may queue; past it, the oldest is dropped.

One second: enough to ride out a stall without a gap, and a bound on what a
participant can make the runtime hold.
"""

_EMPTY = np.empty(0, dtype=np.int16)
_PCM_S16LE = np.dtype("<i2")

MediaSink = Callable[[ConnId, str, InputFrame], None]
"""Receives each input frame: the sender's connection, the input track, the frame."""

InputClaim = Callable[[ConnId, str], bool]
"""Claims an input track for a connection, session-wide; returns whether it was granted."""

InputRelease = Callable[[ConnId, str], None]
"""Releases a connection's claim on an input track."""


@dataclass
class _TrackFeed:
    """A remote track that feeds an input track."""

    sender: str
    conn_id: ConnId
    target: str
    task: asyncio.Task[None]


@dataclass
class _WaitingTrack:
    """A remote track refused because another client holds its input track."""

    track: rtc.Track
    sender: str
    conn_id: ConnId
    target: str


@dataclass
class _Hold:
    """A participant's claim on an input track.

    A claim a byte stream took lasts until the participant leaves, whatever
    its tracks do: its sender sends one stream per utterance.
    """

    conn_id: ConnId
    streamed: bool = False


class StreamFramer:
    """Cut one byte stream's ``pcm_s16le`` bytes into 10 ms mono frames at the track's rate.

    Bytes arrive in chunks of any size, so a sample split between two chunks
    is carried over. A stream at the track's rate is cut as it is, sample for
    sample; any other rate is resampled.

    Args:
        rate: The input track's declared rate.
        stream_rate: The stream's own rate.
        channels: The stream's channel count, mixed down to mono.
    """

    def __init__(self, rate: int, stream_rate: int, channels: int) -> None:
        self._channels = channels
        self._frame = max(1, int(rate * FRAME_SECONDS))
        self._bytes = b""
        self._pending: npt.NDArray[np.int16] = _EMPTY
        self._resampler = (
            rtc.AudioResampler(
                stream_rate, rate, num_channels=1, quality=rtc.AudioResamplerQuality.HIGH
            )
            if stream_rate != rate
            else None
        )

    def push(self, data: bytes) -> list[npt.NDArray[np.int16]]:
        """Take a chunk of the stream and return the whole frames it completes."""
        data = self._bytes + data
        whole = len(data) - len(data) % (2 * self._channels)
        self._bytes = data[whole:]
        if not whole:
            return []
        samples = np.frombuffer(data[:whole], dtype=_PCM_S16LE).astype(np.int16, copy=False)
        if self._channels > 1:
            samples = samples.reshape(-1, self._channels).mean(axis=1).round().astype(np.int16)
        return self._frames(self._resample(samples))

    def flush(self) -> list[npt.NDArray[np.int16]]:
        """Return what the stream left: the resampler's tail and the last, shorter frame."""
        if self._resampler is not None:
            tail = [np.frombuffer(bytes(f.data), np.int16) for f in self._resampler.flush()]
            self._pending = np.concatenate([self._pending, *tail]) if tail else self._pending
        frames = self._frames(_EMPTY)
        if self._pending.size:
            frames.append(self._pending.reshape(1, -1))
            self._pending = _EMPTY
        return frames

    def _resample(self, samples: npt.NDArray[np.int16]) -> npt.NDArray[np.int16]:
        if self._resampler is None:
            return samples
        out = self._resampler.push(bytearray(samples.tobytes()))
        if not out:
            return _EMPTY
        return np.concatenate([np.frombuffer(bytes(f.data), np.int16) for f in out])

    def _frames(self, samples: npt.NDArray[np.int16]) -> list[npt.NDArray[np.int16]]:
        pending = np.concatenate([self._pending, samples]) if samples.size else self._pending
        count = pending.size // self._frame
        frames = [
            pending[i * self._frame : (i + 1) * self._frame].reshape(1, -1).copy()
            for i in range(count)
        ]
        self._pending = pending[count * self._frame :]
        return frames


@dataclass
class _Stream:
    """The byte stream that currently feeds a topic's input track."""

    stream_id: str
    sender: str
    conn_id: ConnId
    track: str
    framer: StreamFramer
    metadata: bytes
    reader: rtc.ByteStreamReader


class RoomInputs:
    """Feed the model's input tracks from the room's tracks and byte streams.

    Args:
        inputs: The model's declared input tracks, by name.
        media_inputs: The manifest's ``media_inputs``: each source and the
            input track it feeds.
        sink: Where each input frame goes.
        claim: Claims an input track for a client in the session's arbitration.
        release: Releases a client's claim.
    """

    def __init__(
        self,
        inputs: Mapping[str, TrackInfo],
        media_inputs: Mapping[str, str],
        sink: MediaSink,
        claim: InputClaim,
        release: InputRelease,
    ) -> None:
        self._inputs = dict(inputs)
        self._sink = sink
        self._claim_input = claim
        self._release_input = release
        self._mapped: dict[str, str] = {}
        for source, track in media_inputs.items():
            if source in TOPIC_CHANNELS:
                logger.warning("a media input cannot use a protocol topic", source=source)
            elif track in self._inputs:
                self._mapped[source] = track
        self._reserved = set(self._mapped.values())
        # Each input track a participant holds through the session, and which.
        self._held: dict[str, _Hold] = {}
        # Each remote track feeding an input track, by its track id.
        self._tracks: dict[str, _TrackFeed] = {}
        # Each remote track waiting for its input track to be freed, by its
        # track id, in arrival order: its sender gets no word that it was
        # refused, so the room starts it once a participant frees the input.
        self._waiting: dict[str, _WaitingTrack] = {}
        # Each byte stream topic, and the stream that currently feeds it.
        self._streams: dict[str, _Stream] = {}
        # The readers each participant's media runs on, so its leaving stops them.
        self._readers: dict[str, set[asyncio.Task[None]]] = {}

    @property
    def stream_topics(self) -> list[str]:
        """The byte stream topics that feed an input track."""
        return list(self._mapped)

    def on_byte_stream(self, reader: rtc.ByteStreamReader, sender: str, conn_id: ConnId) -> None:
        """Start feeding an input track from a byte stream a client opened."""
        topic = reader.info.topic
        track = self._mapped.get(topic)
        info = self._inputs.get(track or "")
        if track is None or info is None or info.kind is not TrackKind.AUDIO:
            reader.close()
            return
        attributes = dict(reader.info.attributes or {})
        rate = _attribute(
            attributes, "sample_rate", int(info.rate), MIN_STREAM_RATE, MAX_STREAM_RATE
        )
        channels = _attribute(attributes, "num_channels", 1, 1, MAX_STREAM_CHANNELS)
        if rate is None or channels is None:
            logger.warning(
                "refused an input byte stream with an unusable audio format",
                topic=topic,
                sample_rate=attributes.get("sample_rate"),
                num_channels=attributes.get("num_channels"),
            )
            reader.close()
            return
        if not self._claim(track, conn_id, streamed=True):
            logger.debug("dropped a stream for an input track another client holds")
            reader.close()
            return
        previous = self._streams.get(topic)
        if previous is not None:
            self._end(topic, previous)
        stream = _Stream(
            stream_id=reader.info.stream_id,
            sender=sender,
            conn_id=conn_id,
            track=track,
            framer=StreamFramer(int(info.rate), rate, channels),
            metadata=json.dumps(
                {"stream_id": reader.info.stream_id, "attributes": attributes}
            ).encode(),
            reader=reader,
        )
        self._streams[topic] = stream
        self._spawn(sender, self._feed(topic, stream, reader))

    def on_track(self, track: rtc.Track, sender: str, conn_id: ConnId) -> None:
        """Start feeding an input track from a remote track a client published."""
        target = self._mapped.get(track.name)
        if target is None and track.name in self._inputs and track.name not in self._reserved:
            target = track.name
        info = self._inputs.get(target or "")
        if target is None or info is None:
            return
        if (track.kind == rtc.TrackKind.KIND_VIDEO) != (info.kind is TrackKind.VIDEO):
            return
        if not self._feed_track(track, sender, conn_id, target):
            logger.debug("a remote track waits for an input track another client holds")
            self._waiting[track.sid] = _WaitingTrack(track, sender, conn_id, target)

    def on_track_gone(self, sid: str) -> None:
        """Stop feeding from a remote track that was unpublished or unsubscribed.

        Its sender's claim on the input track is released, unless the sender
        still feeds that input from another track, or a byte stream took the
        claim. A track waiting for that input then starts feeding it.
        """
        self._waiting.pop(sid, None)
        feed = self._tracks.pop(sid, None)
        if feed is None:
            return
        feed.task.cancel()
        hold = self._held.get(feed.target)
        if hold is None or hold.conn_id != feed.conn_id or hold.streamed:
            return
        if any(
            other.conn_id == feed.conn_id and other.target == feed.target
            for other in self._tracks.values()
        ):
            return
        del self._held[feed.target]
        self._release_input(feed.conn_id, feed.target)
        self._start_waiting(feed.target)

    def release(self, sender: str, conn_id: ConnId) -> None:
        """Stop a participant that left: end its open streams and free the inputs it held.

        A track waiting for one of those inputs then starts feeding it.
        """
        for topic, stream in list(self._streams.items()):
            if stream.sender == sender:
                self._end(topic, stream)
        for task in self._readers.pop(sender, set()):
            task.cancel()
        for sid in [sid for sid, feed in self._tracks.items() if feed.sender == sender]:
            del self._tracks[sid]
        for sid in [sid for sid, waiting in self._waiting.items() if waiting.sender == sender]:
            del self._waiting[sid]
        freed = [track for track, hold in self._held.items() if hold.conn_id == conn_id]
        for track in freed:
            del self._held[track]
            self._release_input(conn_id, track)
        for track in freed:
            self._start_waiting(track)

    def close(self) -> None:
        """Stop every reader, ending each open stream with its end frame.

        Each stream's reader is closed directly: a reader task cancelled
        before it first runs never reaches its own cleanup.
        """
        for topic, stream in list(self._streams.items()):
            self._end(topic, stream)
        for tasks in self._readers.values():
            for task in tasks:
                task.cancel()
        self._readers.clear()
        self._tracks.clear()
        self._waiting.clear()

    def _claim(self, track: str, conn_id: ConnId, *, streamed: bool = False) -> bool:
        """Claim *track* for *conn_id*, or confirm it already holds it."""
        hold = self._held.get(track)
        if hold is not None:
            if hold.conn_id != conn_id:
                return False
            hold.streamed = hold.streamed or streamed
            return True
        if not self._claim_input(conn_id, track):
            return False
        self._held[track] = _Hold(conn_id, streamed)
        return True

    def _feed_track(self, track: rtc.Track, sender: str, conn_id: ConnId, target: str) -> bool:
        """Start feeding *target* from a remote track, if its sender can claim it."""
        if not self._claim(target, conn_id):
            return False
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            task = self._spawn(sender, self._read_video(track, target, conn_id))
        else:
            rate = int(self._inputs[target].rate)
            task = self._spawn(sender, self._read_audio(track, target, rate, conn_id))
        self._tracks[track.sid] = _TrackFeed(sender, conn_id, target, task)
        return True

    def _start_waiting(self, target: str) -> None:
        """Start the first track waiting for *target* that can now claim it."""
        for sid, waiting in list(self._waiting.items()):
            if waiting.target == target and self._feed_track(
                waiting.track, waiting.sender, waiting.conn_id, target
            ):
                del self._waiting[sid]
                return

    def _spawn(self, sender: str, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(coro)
        tasks = self._readers.setdefault(sender, set())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    async def _feed(self, topic: str, stream: _Stream, reader: rtc.ByteStreamReader) -> None:
        try:
            async for chunk in reader:
                if self._streams.get(topic) is not stream:
                    # Superseded by a newer stream on the topic, which ended this one.
                    return
                for samples in stream.framer.push(chunk):
                    self._push(stream, samples)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("reading an input byte stream failed", topic=topic, exc_info=True)
        finally:
            reader.close()
        if self._streams.get(topic) is stream:
            self._end(topic, stream)

    def _end(self, topic: str, stream: _Stream) -> None:
        """Send what the stream left, then its end frame, and stop it feeding the track."""
        if self._streams.get(topic) is stream:
            del self._streams[topic]
        stream.reader.close()
        for samples in stream.framer.flush():
            self._push(stream, samples)
        ended = json.dumps({"stream_id": stream.stream_id, "ended": True}).encode()
        self._sink(
            stream.conn_id,
            stream.track,
            InputFrame(data=np.zeros((1, 0), dtype=np.int16), metadata=ended),
        )

    def _push(self, stream: _Stream, samples: npt.NDArray[np.int16]) -> None:
        self._sink(stream.conn_id, stream.track, InputFrame(data=samples, metadata=stream.metadata))

    async def _read_video(self, track: rtc.Track, target: str, conn_id: ConnId) -> None:
        stream = rtc.VideoStream(track, capacity=VIDEO_QUEUE_FRAMES)
        try:
            async for event in stream:
                frame = event.frame.convert(rtc.VideoBufferType.RGB24)
                data = np.frombuffer(bytes(frame.data), np.uint8)
                picture = data.reshape(frame.height, frame.width, 3).copy()
                self._sink(conn_id, target, InputFrame(data=picture))
        finally:
            await stream.aclose()

    async def _read_audio(self, track: rtc.Track, target: str, rate: int, conn_id: ConnId) -> None:
        stream = rtc.AudioStream(
            track, capacity=AUDIO_QUEUE_FRAMES, sample_rate=rate, num_channels=1
        )
        try:
            async for event in stream:
                samples = np.frombuffer(bytes(event.frame.data), np.int16).reshape(1, -1)
                self._sink(conn_id, target, InputFrame(data=samples.copy()))
        finally:
            await stream.aclose()


def _attribute(
    attributes: Mapping[str, str], key: str, default: int, low: int, high: int
) -> int | None:
    """Read an integer stream attribute: *default* when absent, ``None`` when unusable."""
    value = attributes.get(key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if low <= parsed <= high else None
