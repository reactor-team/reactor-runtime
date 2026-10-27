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

Each input video track a connection sends H.264 on keeps one decoder for the
connection's life: it opens with the first request that carries the track's
H.264 frames, and it is released when the connection unpublishes the track or
closes. Every request's frames start at a
keyframe, so a decoder is reset, not rebuilt, between requests.

A connection's messages reach the runner in the order they arrived, which the
wire promises: while a request decodes, the messages behind it wait.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable, Collection
from dataclasses import dataclass

import av
import numpy as np
import numpy.typing as npt
from google.protobuf.message import DecodeError

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


@dataclass(frozen=True)
class StepFrameRefusal:
    """Why a request's encoded frames could not be decoded.

    Attributes:
        entry: The frame at fault, as the runner names a request's entries
            (``media_input[i].frames[j]``).
        reason: What went wrong, for the client.
    """

    entry: str
    reason: str


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


def track_event(
    payload: bytes | str, version: ProtocolVersion, channel: Channel
) -> tuple[bool, str] | None:
    """Return whether *payload* publishes (``True``) or unpublishes a track, and its name.

    Only a v1 control frame is read: a v0 client cannot send a step request,
    so its tracks never need a decoder. Any other frame returns ``None``.
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
    which = message.WhichOneof("payload")
    if which == "publish_track":
        return True, message.publish_track.name
    if which == "unpublish_track":
        return False, message.unpublish_track.name
    return None


class TrackDecoder:
    """One track's H.264 decoder, kept across a connection's step requests.

    Opens on construction, with its frame threads started, so only the first
    request that uses it pays for that. FFmpeg picks how many: one for each CPU
    the process may run on, plus one, and at most 16. The decoder holds them
    until it is released. Not thread-safe: a connection decodes one request at
    a time.
    """

    def __init__(self) -> None:
        """Open an H.264 decoder that decodes several frames at once."""
        self._codec = av.CodecContext.create("h264", "r")
        # Frame threads decode several frames at once, holding back up to one
        # frame per thread. That costs a request nothing: its frames are
        # flushed out together.
        self._codec.thread_type = "AUTO"
        self._codec.open()

    def decode(self, packets: list[bytes]) -> dict[int, av.VideoFrame]:
        """Decode *packets* as one stream from a keyframe, each picture keyed by its packet's index.

        The decoder is reset first, so nothing a previous request left in it,
        nor an error it failed on, reaches these pictures. A packet missing
        from the result decoded to no picture.

        Raises:
            av.FFmpegError: The packets do not decode.
        """
        self._codec.flush_buffers()
        decoded: dict[int, av.VideoFrame] = {}
        for index, data in enumerate(packets):
            packet = av.Packet(data)
            # The index rides through the decoder as the packet's pts, so each
            # picture finds the frame it came from even when the decoder reorders.
            packet.pts = index
            for picture in self._codec.decode(packet):
                if picture.pts is not None:
                    decoded[picture.pts] = picture
        for picture in self._codec.decode(None):
            if picture.pts is not None:
                decoded[picture.pts] = picture
        return decoded


def decode_step_frames(
    message: data_pb2.DataClientMessage,
    decoders: dict[str, TrackDecoder],
    video_inputs: Collection[str],
) -> Pictures | StepFrameRefusal:
    """Decode each encoded frame of *message*'s step request to its picture.

    Each track's encoded frames are one stream, fed in order to the track's
    decoder in *decoders*; a track without one gets a new one there. Encoded
    frames decode only for a track named in *video_inputs*, the model's input
    video tracks, so *decoders* holds at most one decoder for each of them; an
    entry that sends them to any other track is refused before a decoder opens.
    The request is left as it is.

    Returns:
        The picture of each encoded frame, by ``(entry, position)``, or why the
        request cannot be decoded.
    """
    pictures: Pictures = {}
    for index, entry in enumerate(message.request_step.media_input):
        where = f"media_input[{index}]"
        encodings = {frame.encoding for frame in entry.frames} - _PASSTHROUGH
        if not encodings:
            continue
        unknown = encodings - {step_pb2.FRAME_ENCODING_H264}
        if unknown:
            position = next(p for p, f in enumerate(entry.frames) if f.encoding in unknown)
            name = _encoding_name(entry.frames[position].encoding)
            return StepFrameRefusal(
                f"{where}.frames[{position}]", f"frame encoding {name} is not supported"
            )
        if entry.track not in video_inputs:
            return StepFrameRefusal(
                where,
                f"{where} names {entry.track!r}, which is not an input video track of the model",
            )
        decoded = _decode_h264(entry, index, decoders)
        if isinstance(decoded, StepFrameRefusal):
            return decoded
        pictures.update(decoded)
    return pictures


def _encoding_name(value: int) -> str:
    if value in step_pb2.FrameEncoding.values():
        return step_pb2.FrameEncoding.Name(value)
    return str(value)


_IDR_NAL = 5
"""The NAL unit type of an IDR slice: the access unit an H.264 decoder can start at."""

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


def _access_unit(data: bytes) -> tuple[int | None, str | None]:
    """Read *data* as one access unit: the type of its first slice, or what is wrong with it.

    Returns ``(first_vcl_type, None)`` for one access unit with a picture, or
    ``(None, problem)``. A picture begins at a slice whose ``first_mb_in_slice``
    is 0 (its first bit set, the one-bit code for 0), so a second such slice,
    or an SEI, SPS, PPS or delimiter after a slice, starts another access unit.
    """
    first_vcl: int | None = None
    for unit in _nal_units(data):
        kind = unit[0] & 0x1F
        if kind in _VCL_NALS:
            starts_picture = len(unit) > 1 and bool(unit[1] & 0x80)
            if first_vcl is not None and starts_picture:
                return None, "holds more than one picture"
            if first_vcl is None:
                first_vcl = kind
        elif first_vcl is not None and kind in _ACCESS_UNIT_START_NALS:
            return None, "holds more than one access unit"
    if first_vcl is None:
        return None, "holds no picture"
    return first_vcl, None


def _decode_h264(
    entry: step_pb2.TrackFrames, index: int, decoders: dict[str, TrackDecoder]
) -> Pictures | StepFrameRefusal:
    """Decode the H.264 frames of one track, the *index* entry of its request."""
    where = f"media_input[{index}]"
    positions = [
        p for p, f in enumerate(entry.frames) if f.encoding == step_pb2.FRAME_ENCODING_H264
    ]
    # Each frame is checked as one access unit before anything decodes: a
    # decoder takes two concatenated pictures as one packet and returns only
    # one of them, and handed a stream that starts past its keyframe it
    # conceals the missing reference and returns pictures anyway.
    for count, position in enumerate(positions):
        data = entry.frames[position].data
        if not data:
            return StepFrameRefusal(f"{where}.frames[{position}]", "an H.264 frame carries no data")
        first_vcl, problem = _access_unit(data)
        if problem is not None:
            return StepFrameRefusal(
                f"{where}.frames[{position}]",
                f"the H.264 frame {problem}; each frame is one access unit",
            )
        if count == 0 and first_vcl != _IDR_NAL:
            return StepFrameRefusal(
                f"{where}.frames[{position}]",
                "a track's H.264 frames must start at a keyframe (an IDR access unit)",
            )
    decoder = decoders.get(entry.track)
    if decoder is None:
        decoder = decoders[entry.track] = TrackDecoder()
    try:
        decoded = decoder.decode([entry.frames[p].data for p in positions])
    except av.FFmpegError as error:
        return StepFrameRefusal(where, f"the H.264 frames do not decode: {error}")
    if len(decoded) < len(positions):
        missing = next(p for i, p in enumerate(positions) if i not in decoded)
        return StepFrameRefusal(
            f"{where}.frames[{missing}]",
            f"{len(positions)} H.264 frames decoded to {len(decoded)} pictures; the stream must "
            "start at a keyframe, and each frame must decode to one picture",
        )
    pictures: Pictures = {}
    for i, position in enumerate(positions):
        picture = decoded[i]
        shape = [picture.height, picture.width, 3]
        declared = list(entry.frames[position].shape)
        if declared and declared != shape:
            return StepFrameRefusal(
                f"{where}.frames[{position}]",
                f"the H.264 frame decodes to {shape}, not the {declared} it declares",
            )
        pictures[index, position] = np.ascontiguousarray(picture.to_ndarray(format="rgb24"))
    return pictures


def _refusal(version: ProtocolVersion, request_id: str, refusal: StepFrameRefusal) -> bytes | str:
    result = step_pb2.StepResult(
        failure_code=step_pb2.STEP_FAILURE_CODE_REFUSED, error=refusal.reason, entry=refusal.entry
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
    tracks that the connection sends H.264 frames on, from the first request
    that carries them until the connection unpublishes the track or the gate
    closes, so a connection holds at most one decoder per input video track, and
    one that never sends a step request holds none. H.264 frames sent to any
    other track are refused.

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
        self._decoders.clear()
        if self._draining is not None:
            self._draining.cancel()
            self._draining = None

    def _pass(self, payload: bytes | str, version: ProtocolVersion, channel: Channel) -> None:
        """Deliver a message that carries no encoded frame, releasing any track it unpublishes."""
        event = track_event(payload, version, channel)
        if event is not None and not event[0]:
            self._decoders.pop(event[1], None)
        self._deliver(payload, version, channel)

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
                self._decoders.update(decoders)
                if isinstance(outcome, StepFrameRefusal):
                    logger.debug("step request refused at %s: %s", outcome.entry, outcome.reason)
                    self._reply(_refusal(version, message.request_id, outcome))
                    continue
                self._deliver_step(DecodedStep(payload=payload, pictures=outcome), version)
        finally:
            self._draining = None
