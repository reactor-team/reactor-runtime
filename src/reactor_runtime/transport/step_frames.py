"""Decode the encoded frames of a client's step request, before the runner reads it.

A step request's ``media_input`` frames each name their encoding. The runner
reads raw frames only; an encoded frame is the transport's to decode, as the
transport decodes a streamed video track. :class:`StepFrameGate` sits on a
connection's inbound messages: a request that carries an encoded frame is
decoded on a worker thread, and goes up to the runner as it arrived, with the
picture each encoded frame decodes to beside it. A request the gate cannot
decode, because it does not know the encoding or the bytes do not decode, is
answered as refused on the connection and never reaches the runner. Every other
message goes on untouched.

An input video track's encoded frames are one stream for the connection's
life, whatever the encoding (H.264, VP8, VP9 or AV1): each request continues
the stream the one before it left, a refused one included, and only the
track's first frames on a connection, or its first after frames that did not
decode, start at a keyframe. Each
such track keeps one decoder from reactor-webrtc for that life: it opens with
the first request that carries the track's encoded frames, and it is released
when the connection unpublishes the track or closes, which a session's end
does. A request's tracks decode at once, each on its own decoder, on a pool of
threads the process shares.

A connection's messages reach the runner in the order they arrived, which the
wire promises: while a request decodes, the messages behind it wait.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from collections import deque
from collections.abc import Callable, Collection, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import reactor_webrtc as rw
from google.protobuf.message import DecodeError

from reactor_runtime.codes import INPUT_DECODE_FAILED
from reactor_runtime.core import DecodedStep
from reactor_runtime.protocol import Channel, Codec, ProtocolVersion, select
from reactor_wire.v1 import control_pb2, data_pb2, step_pb2

logger = logging.getLogger(__name__)

_PASSTHROUGH = frozenset({step_pb2.FRAME_ENCODING_UNSPECIFIED, step_pb2.FRAME_ENCODING_RAW})
"""Encodings the runner reads itself: raw, and unspecified, which it refuses."""

Pictures = dict[tuple[int, int], npt.NDArray[np.uint8]]
"""The RGB picture of each encoded frame of a request, by ``(entry, position)``."""

# One codec per wire version, built on first use.
_codecs: dict[ProtocolVersion, Codec] = {}

_POOL = ThreadPoolExecutor(
    max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="step-frames"
)
"""The threads a request's tracks decode on.

Shared by every connection, and started as work arrives, so a process that
decodes no step frame starts none.
"""


@dataclass(frozen=True)
class StepFrameRefusal:
    """Why a request's encoded frames could not be decoded.

    Attributes:
        entry: The frame at fault, as the runner names a request's entries
            (``media_input[i].frames[j]``).
        reason: What went wrong, for the client.
        code: The stable code the refused result carries:
            ``input_decode_failed`` when a frame of a model input track broke
            the track's stream, so the client starts the track again at a
            keyframe. ``None`` when the stream is whole (a frame that decodes
            to another shape than it declares) or when there is no stream to
            restart (a track that is not an input video track, an encoding
            that is not supported): a keyframe would not help.
    """

    entry: str
    reason: str
    code: str | None = None


def _stream_lost(entry: str, reason: str) -> StepFrameRefusal:
    """Refuse a frame that broke its track's stream: the client restarts it at a keyframe."""
    return StepFrameRefusal(entry, reason, INPUT_DECODE_FAILED)


def _codec(version: ProtocolVersion) -> Codec:
    codec = _codecs.get(version)
    if codec is None:
        codec = select(version)
        _codecs[version] = codec
    return codec


def encoded_step(
    payload: bytes | str, version: ProtocolVersion, channel: Channel
) -> data_pb2.DataClientMessage | None:
    """Return the step request *payload* carries if one of its frames is encoded, else ``None``.

    Only a v1 data-channel frame can carry a step request. A frame that does not
    parse is the runner's to reject, so it is returned as ``None`` too.
    """
    if version is not ProtocolVersion.V1 or channel is not Channel.DATA or isinstance(payload, str):
        return None
    try:
        message = _codec(version).decode_inbound(payload, channel)
    except (ValueError, DecodeError):
        return None
    if not isinstance(message, data_pb2.DataClientMessage):
        return None
    if message.WhichOneof("payload") != "request_step":
        return None
    for entry in message.request_step.media_input:
        if any(frame.encoding not in _PASSTHROUGH for frame in entry.frames):
            return message
    return None


def unpublished_track(
    payload: bytes | str, version: ProtocolVersion, channel: Channel
) -> str | None:
    """Return the name of the track *payload* unpublishes, or ``None`` if it unpublishes none.

    Only a v1 control frame is read: a v0 client cannot send a step request,
    so its tracks never hold a decoder.
    """
    if version is not ProtocolVersion.V1 or channel is not Channel.CONTROL:
        return None
    if isinstance(payload, str):
        return None
    try:
        message = _codec(version).decode_inbound(payload, channel)
    except (ValueError, DecodeError):
        return None
    if not isinstance(message, control_pb2.ControlClientMessage):
        return None
    if message.WhichOneof("payload") != "unpublish_track":
        return None
    return message.unpublish_track.name


class TrackDecoder:
    """One input track's decoder, kept across a connection's step requests.

    Decodes one encoding, on the backend this process has for it: hardware
    when the platform has one, software otherwise. It keeps its reference
    frames between requests, so a track's frames are one stream for the
    connection's life, and it takes a keyframe first. Not thread-safe: one
    request decodes on it at a time.

    Attributes:
        encoding: The wire encoding of the frames it decodes.
    """

    def __init__(self, encoding: step_pb2.FrameEncoding.ValueType) -> None:
        """Open a decoder for *encoding*, one of the encoded :class:`FrameEncoding` values.

        Raises:
            reactor_webrtc.CodecUnsupported: This process has no decoder for it.
        """
        self.encoding = encoding
        self._decoder = _video_codecs().decoder(_CODECS[encoding])

    def decode(self, data: bytes) -> npt.NDArray[np.uint8] | None:
        """Decode the stream's next frame to its RGB picture, ``None`` when it gives none.

        Raises:
            reactor_webrtc.CodecNeedsKeyFrame: The decoder waits for a keyframe
                and *data* is not one; its state is left as it was.
            reactor_webrtc.CodecError: The frame does not decode; the decoder
                then waits for a keyframe.
        """
        return self._decoder.decode(data)


def decode_step_frames(
    message: data_pb2.DataClientMessage,
    decoders: dict[str, TrackDecoder],
    video_inputs: Collection[str],
) -> Pictures | StepFrameRefusal:
    """Decode each encoded frame of *message*'s step request to its picture.

    Each track's encoded frames continue the stream the track's decoder in
    *decoders* holds from earlier requests. A track without one gets a new one
    there, and so does a track whose frames change encoding; a new decoder
    starts at a keyframe, so a connection's first frames on a track must start
    at one. Encoded frames decode only for a track named in *video_inputs*, the
    model's input video tracks, so *decoders* holds at most one decoder for
    each of them. The request's tracks decode at once; entries that name the
    same track decode one after another, on its one decoder. The request is
    left as it is.

    Every frame that can decode does, even in a request that is refused, so a
    track's stream moves on with each request. A track whose frames cannot
    decode (an encoding this process does not decode, a frame the decoder
    rejects, frames that do not start at a keyframe where the stream needs one)
    loses its stream: its decoder leaves *decoders*, and its next frames must
    start at a keyframe.

    Returns:
        The RGB picture of each encoded frame, by ``(entry, position)``, or why
        the request cannot be decoded: the first entry at fault, in request
        order.
    """
    tracks: dict[str, _Track] = {}
    for index, entry in enumerate(message.request_step.media_input):
        checked = _check_entry(entry, index, video_inputs)
        if checked is None:
            continue
        track = tracks.setdefault(entry.track, _Track(entry.track))
        if track.lost is not None:
            continue
        if isinstance(checked, StepFrameRefusal):
            track.lost = checked
        elif track.streams and track.streams[0].encoding != checked.encoding:
            track.lost = _stream_lost(
                f"media_input[{index}].frames[{checked.positions[0]}]",
                f"track {entry.track!r} takes one frame encoding in a request",
            )
        else:
            track.streams.append(checked)
    live = [track for track in tracks.values() if track.lost is None]
    opened = _each(lambda track: _decoder_for(track.streams[0], decoders.get(track.name)), live)
    for track, decoder in zip(live, opened, strict=True):
        if isinstance(decoder, StepFrameRefusal):
            track.lost = decoder
        else:
            decoders[track.name] = decoder
    ready = [track for track in live if track.lost is None]
    pictures: Pictures = {}
    refusals: list[StepFrameRefusal] = []
    for track, decoded in zip(
        ready, _each(lambda t: _decode_track(t.streams, decoders[t.name]), ready), strict=True
    ):
        pictures.update(decoded.pictures)
        if decoded.lost:
            track.lost = decoded.refusal
        elif decoded.refusal is not None:
            refusals.append(decoded.refusal)
    for track in tracks.values():
        if track.lost is not None:
            refusals.append(track.lost)
            decoders.pop(track.name, None)
    if refusals:
        return min(refusals, key=lambda refusal: _entry_order(refusal.entry))
    return pictures


_CODECS: dict[step_pb2.FrameEncoding.ValueType, rw.VideoCodec] = {
    step_pb2.FRAME_ENCODING_H264: rw.VideoCodec.H264,
    step_pb2.FRAME_ENCODING_VP8: rw.VideoCodec.Vp8,
    step_pb2.FRAME_ENCODING_VP9: rw.VideoCodec.Vp9,
    step_pb2.FRAME_ENCODING_AV1: rw.VideoCodec.Av1,
}
"""The codec each encoded frame encoding is decoded with."""

_backends_lock = threading.Lock()
_backends: list[rw.VideoCodecs] = []


def _video_codecs() -> rw.VideoCodecs:
    """Return the process's video codec backends, collected on first use."""
    with _backends_lock:
        if not _backends:
            _backends.append(rw.VideoCodecs())
        return _backends[0]


@dataclass(frozen=True)
class _Stream:
    """One entry's encoded frames, checked and ready to decode."""

    index: int
    entry: step_pb2.TrackFrames
    positions: list[int]
    encoding: step_pb2.FrameEncoding.ValueType


@dataclass
class _Track:
    """One track's encoded frames in a request, and the fault that lost its stream, if any."""

    name: str
    streams: list[_Stream] = field(default_factory=list)
    lost: StepFrameRefusal | None = None


@dataclass(frozen=True)
class _Decoded:
    """What one track's frames decoded to.

    Attributes:
        pictures: The picture of each frame that decoded, by ``(entry, position)``.
        refusal: The first frame at fault, if any.
        lost: Whether the fault broke the stream, so the track starts over.
    """

    pictures: dict[tuple[int, int], npt.NDArray[np.uint8]]
    refusal: StepFrameRefusal | None = None
    lost: bool = False


def _each[T, R](work: Callable[[T], R], items: list[T]) -> list[R]:
    """Run *work* on every item at once, on the shared pool; one item runs here."""
    if len(items) <= 1:
        return [work(item) for item in items]
    return list(_POOL.map(work, items))


def _entry_order(entry: str) -> tuple[int, int]:
    """Order a refusal's entry (``media_input[i]`` or ``media_input[i].frames[j]``) by request."""
    numbers = [int(part.split("]")[0]) for part in entry.split("[")[1:]]
    return numbers[0], numbers[1] if len(numbers) > 1 else -1


def _check_entry(
    entry: step_pb2.TrackFrames, index: int, video_inputs: Collection[str]
) -> _Stream | StepFrameRefusal | None:
    """Check one entry's encoded frames; ``None`` when it carries none."""
    where = f"media_input[{index}]"
    positions = [p for p, f in enumerate(entry.frames) if f.encoding not in _PASSTHROUGH]
    if not positions:
        return None
    for position in positions:
        if entry.frames[position].encoding not in _CODECS:
            name = _encoding_name(entry.frames[position].encoding)
            return StepFrameRefusal(
                f"{where}.frames[{position}]", f"frame encoding {name} is not supported"
            )
    if entry.track not in video_inputs:
        return StepFrameRefusal(
            where,
            f"{where} names {entry.track!r}, which is not an input video track of the model",
        )
    encoding = entry.frames[positions[0]].encoding
    for position in positions:
        frame = entry.frames[position]
        if frame.encoding != encoding:
            return _stream_lost(
                f"{where}.frames[{position}]",
                f"track {entry.track!r} takes one frame encoding in a request",
            )
        if not frame.data:
            return _stream_lost(f"{where}.frames[{position}]", "an encoded frame carries no data")
        if encoding == step_pb2.FRAME_ENCODING_H264:
            # A decoder takes two concatenated pictures as one frame and
            # returns only one of them, so each is checked as one access unit.
            problem = _access_unit_problem(frame.data)
            if problem is not None:
                return _stream_lost(
                    f"{where}.frames[{position}]",
                    f"the H.264 frame {problem}; each frame is one access unit",
                )
    return _Stream(index=index, entry=entry, positions=positions, encoding=encoding)


def _encoding_name(value: int) -> str:
    if value in step_pb2.FrameEncoding.values():
        return step_pb2.FrameEncoding.Name(value)
    return str(value)


_VCL_NALS = frozenset(range(1, 6))
"""NAL unit types that carry a picture's slices (coded slice data, non-IDR and IDR)."""

_ACCESS_UNIT_START_NALS = frozenset({6, 7, 8, 9})
"""SEI, SPS, PPS and access unit delimiter: after a picture's slices, each starts the
next access unit (H.264 7.4.1.2.3)."""


def _nal_units(data: bytes) -> list[bytes]:
    """The NAL units of an Annex B byte stream, without their start codes.

    A unit's trailing zero bytes are the leading byte of a four-byte start code
    or padding; a unit's own bytes never end in zero (rbsp_trailing_bits).
    """
    units = []
    at = data.find(b"\x00\x00\x01")
    while at != -1:
        start = at + 3
        following = data.find(b"\x00\x00\x01", start)
        unit = data[start : len(data) if following == -1 else following].rstrip(b"\x00")
        if unit:
            units.append(unit)
        at = following
    return units


def _access_unit_problem(data: bytes) -> str | None:
    """Return what keeps *data* from being one H.264 access unit with a picture, if anything.

    A picture begins at a slice whose ``first_mb_in_slice`` is 0 (its first bit
    set, the one-bit code for 0), so a second such slice, or an SEI, SPS, PPS or
    delimiter after a slice, starts another access unit.
    """
    seen_slice = False
    for unit in _nal_units(data):
        kind = unit[0] & 0x1F
        if kind in _VCL_NALS:
            starts_picture = len(unit) > 1 and bool(unit[1] & 0x80)
            if seen_slice and starts_picture:
                return "holds more than one picture"
            seen_slice = True
        elif seen_slice and kind in _ACCESS_UNIT_START_NALS:
            return "holds more than one access unit"
    if not seen_slice:
        return "holds no picture"
    return None


def _decoder_for(stream: _Stream, current: TrackDecoder | None) -> TrackDecoder | StepFrameRefusal:
    """Return the decoder *stream*'s track decodes on: *current*, or a new one for its encoding."""
    if current is not None and current.encoding == stream.encoding:
        return current
    try:
        return TrackDecoder(stream.encoding)
    except rw.CodecError as error:
        name = _encoding_name(stream.encoding)
        return StepFrameRefusal(
            f"media_input[{stream.index}].frames[{stream.positions[0]}]",
            f"frame encoding {name} is not supported here: {error}",
        )


def _decode_track(streams: list[_Stream], decoder: TrackDecoder) -> _Decoded:
    """Decode one track's entries in request order, each frame to its picture.

    A frame that decodes to another shape than it declares is a fault that
    leaves the stream whole, so decoding goes on past it; any other fault stops
    the track there.
    """
    pictures: dict[tuple[int, int], npt.NDArray[np.uint8]] = {}
    mismatch: StepFrameRefusal | None = None
    for stream in streams:
        for position in stream.positions:
            where = f"media_input[{stream.index}].frames[{position}]"
            frame = stream.entry.frames[position]
            try:
                picture = decoder.decode(frame.data)
            except rw.CodecNeedsKeyFrame:
                return _Decoded(
                    pictures,
                    _stream_lost(
                        where,
                        f"track {stream.entry.track!r} must start at a keyframe: its first "
                        "frame on a connection, and its first after frames that did not decode",
                    ),
                    lost=True,
                )
            except (rw.CodecError, ValueError) as error:
                return _Decoded(
                    pictures, _stream_lost(where, f"the frame does not decode: {error}"), True
                )
            if picture is None:
                return _Decoded(
                    pictures, _stream_lost(where, "the frame decoded to no picture"), True
                )
            declared = list(frame.shape)
            if declared and declared != list(picture.shape) and mismatch is None:
                mismatch = StepFrameRefusal(
                    where,
                    f"the frame decodes to {list(picture.shape)}, not the {declared} it declares",
                )
            pictures[stream.index, position] = picture
    return _Decoded(pictures, mismatch)


def _refusal(version: ProtocolVersion, request_id: str, refusal: StepFrameRefusal) -> bytes | str:
    result = step_pb2.StepResult(
        failure_code=step_pb2.STEP_FAILURE_CODE_REFUSED,
        error=refusal.reason,
        entry=refusal.entry,
        code=refusal.code,
    )
    _, frame = _codec(version).encode_step_result(request_id, result)
    return frame


class StepFrameGate:
    """One connection's inbound messages, with each step request's encoded frames decoded.

    :meth:`submit` takes the connection's messages in the order they arrive and
    hands them on in that order: a step request with an encoded frame to
    *deliver_step*, with its pictures, and every other message to *deliver*. A
    step request with an encoded frame decodes on a worker thread; while it
    does, the messages behind it wait, so nothing overtakes it. A request that
    does not decode is answered as refused through *reply*, and is not
    delivered.

    The gate holds a :class:`TrackDecoder` for each of the model's input video
    tracks that the connection sends encoded frames on, from the first request
    that carries them until the connection unpublishes the track or the gate
    closes, so a connection holds at most one decoder per input video track, and
    one that never sends a step request holds none. Each decoder continues its
    track's stream from one request to the next, and a new gate's start at a
    keyframe. Encoded frames sent to any other track are refused.

    Runs on the event loop the connection's callbacks run on.
    """

    def __init__(
        self,
        deliver: Callable[[bytes | str, ProtocolVersion, Channel], None],
        deliver_step: Callable[[DecodedStep, ProtocolVersion], None],
        reply: Callable[[bytes | str], None],
        video_inputs: Collection[str],
    ) -> None:
        """Bind the gate to the connection's upward deliveries and its outbound send.

        Args:
            deliver: Hands a message on to the runner.
            deliver_step: Hands a decoded step request on to the runner.
            reply: Sends an encoded frame to the client, on the data channel.
            video_inputs: The names of the model's input video tracks, the only
                tracks the gate opens a decoder for.
        """
        self._deliver = deliver
        self._deliver_step = deliver_step
        self._reply = reply
        self._video_inputs = frozenset(video_inputs)
        self._backlog: deque[tuple[bytes | str, ProtocolVersion, Channel]] = deque()
        self._draining: asyncio.Task[None] | None = None
        self._decoders: dict[str, TrackDecoder] = {}

    @property
    def decoders(self) -> int:
        """The number of track decoders the gate holds open."""
        return len(self._decoders)

    def submit(self, payload: bytes | str, version: ProtocolVersion, channel: Channel) -> None:
        """Take one inbound message: deliver it now, or queue it behind a decode."""
        if self._draining is None and encoded_step(payload, version, channel) is None:
            self._pass(payload, version, channel)
            return
        self._backlog.append((payload, version, channel))
        if self._draining is None:
            self._draining = asyncio.get_running_loop().create_task(self._drain())

    def close(self) -> None:
        """Drop what is still queued and release every decoder; the connection is going away."""
        self._backlog.clear()
        self._release(self._decoders)
        if self._draining is not None:
            self._draining.cancel()
            self._draining = None

    def _pass(self, payload: bytes | str, version: ProtocolVersion, channel: Channel) -> None:
        """Deliver a message that carries no encoded frame, releasing any track it unpublishes."""
        name = unpublished_track(payload, version, channel)
        if name is not None:
            self._release([name])
        self._deliver(payload, version, channel)

    def _release(self, names: Iterable[str]) -> None:
        """Drop the decoders of *names*, freeing them on a worker thread.

        Freeing a decoder can wait on its backend, a hardware session or
        threads, which stays off the event loop as opening one does.
        """
        released = [self._decoders.pop(name) for name in list(names) if name in self._decoders]
        if not released:
            return
        # With no running loop, or its executor shut down, they are freed here.
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().run_in_executor(None, released.clear)

    async def _drain(self) -> None:
        try:
            while self._backlog:
                payload, version, channel = self._backlog.popleft()
                message = encoded_step(payload, version, channel)
                if message is None or not isinstance(payload, bytes):
                    self._pass(payload, version, channel)
                    continue
                # Decoded against a copy, so a close while the decode runs
                # releases every decoder, the ones it opens included.
                decoders = dict(self._decoders)
                outcome = await asyncio.to_thread(
                    decode_step_frames, message, decoders, self._video_inputs
                )
                # A track that lost its stream left the copy, to start over.
                self._release(name for name in self._decoders if name not in decoders)
                self._decoders.update(decoders)
                if isinstance(outcome, StepFrameRefusal):
                    logger.debug("step request refused at %s: %s", outcome.entry, outcome.reason)
                    self._reply(_refusal(version, message.request_id, outcome))
                    continue
                self._deliver_step(DecodedStep(payload=payload, pictures=outcome), version)
        finally:
            self._draining = None
