"""Media frames on the wire, keyed by their ``FrameEncoding``.

A stepped session carries its input frames in a ``RequestStep`` message's
``media_input``, each tagged with the encoding its bytes use. Each encoding has one codec here,
in a registry keyed by the ``FrameEncoding`` value, so a compressed encoding is
added as one more entry without touching the callers.

The raw encoding carries uncompressed samples in row-major, little-endian
order: video is ``uint8`` RGB shaped ``(height, width, 3)``, audio is ``int16``
PCM shaped ``(channels, samples)``.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from reactor_runtime.core.values import InputFrame, TrackKind
from reactor_wire.v1 import step_pb2


class FrameEncodingError(ValueError):
    """A frame cannot be decoded: its encoding is unknown, or its bytes do not fit it."""


class FrameCodec(Protocol):
    """Turns a wire frame of one encoding into the array a model reads."""

    def decode(self, frame: step_pb2.Frame, kind: TrackKind) -> npt.NDArray[Any]:
        """Return the frame's samples as an array of the track kind's shape.

        Raises:
            FrameEncodingError: The frame's dtype, shape, or bytes do not fit
                the encoding and the track kind.
        """
        ...


# The element type and the rank the raw encoding carries for each track kind,
# in the shape the runtime hands a model for that kind.
_RAW_LAYOUT: dict[TrackKind, tuple[str, int]] = {
    TrackKind.VIDEO: ("uint8", 3),
    TrackKind.AUDIO: ("int16", 2),
}


class RawFrames:
    """The uncompressed encoding: the samples as they are, row-major and little-endian."""

    def decode(self, frame: step_pb2.Frame, kind: TrackKind) -> npt.NDArray[Any]:
        """Return the frame's samples, checked against the track kind's layout.

        Audio with several channels is mixed down to ``(1, M)`` mono, the shape
        a model reads audio in from any transport.
        """
        dtype, rank = _RAW_LAYOUT[kind]
        if frame.dtype != dtype:
            raise FrameEncodingError(f"a raw {kind} frame is {dtype}, not {frame.dtype!r}")
        shape = tuple(int(size) for size in frame.shape)
        if len(shape) != rank or any(size == 0 for size in shape):
            raise FrameEncodingError(
                f"a raw {kind} frame has {rank} non-zero dimensions, got shape {list(shape)}"
            )
        if kind is TrackKind.VIDEO and shape[2] != 3:
            raise FrameEncodingError(
                f"a raw video frame is (height, width, 3), got shape {list(shape)}"
            )
        wire = np.dtype(dtype).newbyteorder("<")
        expected = math.prod(shape) * wire.itemsize
        if len(frame.data) != expected:
            raise FrameEncodingError(
                f"a raw {kind} frame of shape {list(shape)} is {expected} bytes, "
                f"got {len(frame.data)}"
            )
        samples = np.frombuffer(frame.data, dtype=wire).reshape(shape).astype(dtype)
        if kind is TrackKind.AUDIO and shape[0] > 1:
            samples = samples.mean(axis=0, keepdims=True).round().astype(np.int16)
        return samples


FRAME_CODECS: dict[int, FrameCodec] = {step_pb2.FRAME_ENCODING_RAW: RawFrames()}
"""The codec for each ``FrameEncoding`` the runtime reads, by its enum value."""


def decode_frame(frame: step_pb2.Frame, kind: TrackKind) -> InputFrame:
    """Decode one wire frame of a *kind* track into the frame a model reads.

    The frame's metadata and capture time ride along unchanged.

    Raises:
        FrameEncodingError: The encoding is unspecified or unknown, or the
            frame does not fit it.
    """
    codec = FRAME_CODECS.get(frame.encoding)
    if codec is None:
        name = step_pb2.FrameEncoding.Name(frame.encoding) if _known(frame.encoding) else None
        raise FrameEncodingError(f"frame encoding {name or frame.encoding} is not supported")
    return InputFrame(
        data=codec.decode(frame, kind),
        metadata=frame.metadata if frame.HasField("metadata") else None,
        capture_time_us=frame.capture_time_us if frame.HasField("capture_time_us") else None,
    )


def _known(value: int) -> bool:
    return value in step_pb2.FrameEncoding.values()
