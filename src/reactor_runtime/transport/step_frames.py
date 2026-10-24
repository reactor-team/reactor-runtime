"""Decode the encoded frames of a client's step request, before the runner reads it.

A step request's ``media_input`` frames each name their encoding. The runner
reads raw frames only; an encoded frame is the transport's to decode, as the
transport decodes a streamed video track. :class:`StepFrameGate` sits on a
connection's inbound messages: a request that carries an encoded frame is
decoded on a worker thread, and the runner gets the same request with every
encoded frame replaced by the raw frame it decodes to. A request the gate cannot
decode, because it does not know the encoding or the bytes do not decode, is
answered as refused on the connection and never reaches the runner. Every other
message goes on untouched.

A connection's messages reach the runner in the order they arrived, which the
wire promises: while a request decodes, the messages behind it wait.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import av
from google.protobuf.message import DecodeError

from reactor_runtime.protocol import Channel, Codec, ProtocolVersion, select
from reactor_wire.v1 import data_pb2, step_pb2

logger = logging.getLogger(__name__)

_PASSTHROUGH = frozenset({step_pb2.FRAME_ENCODING_UNSPECIFIED, step_pb2.FRAME_ENCODING_RAW})
"""Encodings the runner reads itself: raw, and unspecified, which it refuses."""

_VIDEO_DTYPE = "uint8"

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


def decode_step_frames(message: data_pb2.DataClientMessage) -> StepFrameRefusal | None:
    """Replace each encoded frame of *message*'s step request with the raw frame it decodes to.

    Decodes in place. Each track's encoded frames are one stream, fed to one
    decoder in order. Returns why the request cannot be decoded, leaving
    *message* in an unspecified state, or ``None`` once every frame is raw.
    """
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
        refusal = _decode_h264(entry, where)
        if refusal is not None:
            return refusal
    return None


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


def _decode_h264(entry: step_pb2.TrackFrames, where: str) -> StepFrameRefusal | None:
    """Decode the H.264 frames of one track in place, one raw frame for each."""
    positions = [
        p for p, f in enumerate(entry.frames) if f.encoding == step_pb2.FRAME_ENCODING_H264
    ]
    # Each frame is checked as one access unit before anything decodes: a
    # decoder takes two concatenated pictures as one packet and returns only
    # one of them, and handed a stream that starts past its keyframe it
    # conceals the missing reference and returns pictures anyway.
    for index, position in enumerate(positions):
        data = entry.frames[position].data
        if not data:
            return StepFrameRefusal(f"{where}.frames[{position}]", "an H.264 frame carries no data")
        first_vcl, problem = _access_unit(data)
        if problem is not None:
            return StepFrameRefusal(
                f"{where}.frames[{position}]",
                f"the H.264 frame {problem}; each frame is one access unit",
            )
        if index == 0 and first_vcl != _IDR_NAL:
            return StepFrameRefusal(
                f"{where}.frames[{position}]",
                "a track's H.264 frames must start at a keyframe (an IDR access unit)",
            )
    decoder = av.CodecContext.create("h264", "r")
    decoded: dict[int, av.VideoFrame] = {}
    try:
        for position in positions:
            packet = av.Packet(entry.frames[position].data)
            # The position rides through the decoder as the packet's pts, so each
            # picture finds the frame it came from even when the decoder reorders.
            packet.pts = position
            for picture in decoder.decode(packet):
                if picture.pts is not None:
                    decoded[picture.pts] = picture
        for picture in decoder.decode(None):
            if picture.pts is not None:
                decoded[picture.pts] = picture
    except av.FFmpegError as error:
        return StepFrameRefusal(where, f"the H.264 frames do not decode: {error}")
    missing = [p for p in positions if p not in decoded]
    if missing:
        return StepFrameRefusal(
            f"{where}.frames[{missing[0]}]",
            f"{len(positions)} H.264 frames decoded to {len(decoded)} pictures; the stream must "
            "start at a keyframe, and each frame must decode to one picture",
        )
    for position in positions:
        frame = entry.frames[position]
        picture = decoded[position]
        shape = [picture.height, picture.width, 3]
        if list(frame.shape) and list(frame.shape) != shape:
            return StepFrameRefusal(
                f"{where}.frames[{position}]",
                f"the H.264 frame decodes to {shape}, not the {list(frame.shape)} it declares",
            )
        frame.encoding = step_pb2.FRAME_ENCODING_RAW
        frame.dtype = _VIDEO_DTYPE
        del frame.shape[:]
        frame.shape.extend(shape)
        frame.data = picture.to_ndarray(format="rgb24").tobytes()
    return None


def _refusal(version: ProtocolVersion, request_id: str, refusal: StepFrameRefusal) -> bytes | str:
    result = step_pb2.StepResult(
        failure_code=step_pb2.STEP_FAILURE_CODE_REFUSED, error=refusal.reason, entry=refusal.entry
    )
    _, frame = _codec(version).encode_step_result(request_id, result)
    return frame


class StepFrameGate:
    """One connection's inbound messages, with each step request's encoded frames decoded.

    :meth:`submit` takes the connection's messages in the order they arrive and
    hands them to *deliver* in that order. A step request with an encoded frame
    decodes on a worker thread and goes on as a raw one; while it does, the
    messages behind it wait, so nothing overtakes it. A request that does not
    decode is answered as refused through *reply*, and is not delivered.

    Runs on the event loop the connection's callbacks run on.
    """

    def __init__(
        self,
        deliver: Callable[[bytes | str, ProtocolVersion, Channel], None],
        reply: Callable[[bytes | str], None],
    ) -> None:
        """Bind the gate to the connection's upward delivery and its outbound send.

        Args:
            deliver: Hands a message on to the runner.
            reply: Sends an encoded frame to the client, on the data channel.
        """
        self._deliver = deliver
        self._reply = reply
        self._backlog: deque[tuple[bytes | str, ProtocolVersion, Channel]] = deque()
        self._draining: asyncio.Task[None] | None = None

    def submit(self, payload: bytes | str, version: ProtocolVersion, channel: Channel) -> None:
        """Take one inbound message: deliver it now, or queue it behind a decode."""
        if self._draining is None and encoded_step(payload, version, channel) is None:
            self._deliver(payload, version, channel)
            return
        self._backlog.append((payload, version, channel))
        if self._draining is None:
            self._draining = asyncio.get_running_loop().create_task(self._drain())

    def close(self) -> None:
        """Drop what is still queued; the connection is going away."""
        self._backlog.clear()
        if self._draining is not None:
            self._draining.cancel()
            self._draining = None

    async def _drain(self) -> None:
        try:
            while self._backlog:
                payload, version, channel = self._backlog.popleft()
                message = encoded_step(payload, version, channel)
                if message is not None:
                    refusal = await asyncio.to_thread(decode_step_frames, message)
                    if refusal is not None:
                        logger.debug(
                            "step request refused at %s: %s", refusal.entry, refusal.reason
                        )
                        self._reply(_refusal(version, message.request_id, refusal))
                        continue
                    payload = message.SerializeToString()
                self._deliver(payload, version, channel)
        finally:
            self._draining = None
