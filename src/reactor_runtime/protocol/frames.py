"""Media frames on the wire, keyed by their ``FrameEncoding``.

A stepped session carries its input frames in a ``RequestStep`` message's
``media_input`` and its output frames inside a ``StepResult``, each tagged with
the encoding its bytes use. Each encoding has one codec here,
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

from reactor_runtime.core.values import InputFrame, MediaBundle, TrackData, TrackKind
from reactor_wire.v1 import platform_pb2, step_pb2


class FrameEncodingError(ValueError):
    """A frame cannot be carried: its encoding is unknown, or its samples do not fit it."""


class FrameCodec(Protocol):
    """Turns a wire frame of one encoding into an array, and an array into one."""

    def check(self, frame: step_pb2.Frame, kind: TrackKind) -> tuple[int, ...]:
        """Check the frame fits the encoding and the track kind, without copying its samples.

        Returns:
            The shape :meth:`decode` returns for the frame.

        Raises:
            FrameEncodingError: The frame's dtype, shape, or bytes do not fit
                the encoding and the track kind.
        """
        ...

    def decode(self, frame: step_pb2.Frame, kind: TrackKind) -> npt.NDArray[Any]:
        """Return the frame's samples as an array of the track kind's shape.

        Raises:
            FrameEncodingError: The frame's dtype, shape, or bytes do not fit
                the encoding and the track kind.
        """
        ...

    def encode(self, data: npt.NDArray[Any], kind: TrackKind) -> step_pb2.Frame:
        """Return one frame of a *kind* track as a wire frame of this encoding.

        Raises:
            FrameEncodingError: The array cannot be carried as one frame of
                the track kind.
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

    def check(self, frame: step_pb2.Frame, kind: TrackKind) -> tuple[int, ...]:
        """Check the frame against the track kind's layout, and return its decoded shape.

        Audio with several channels decodes to ``(1, M)`` mono.
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
        if kind is TrackKind.AUDIO:
            return (1, shape[1])
        return shape

    def decode(self, frame: step_pb2.Frame, kind: TrackKind) -> npt.NDArray[Any]:
        """Return the frame's samples, checked against the track kind's layout.

        Audio with several channels is mixed down to ``(1, M)`` mono, the shape
        a model reads audio in from any transport.
        """
        self.check(frame, kind)
        dtype, _ = _RAW_LAYOUT[kind]
        shape = tuple(int(size) for size in frame.shape)
        wire = np.dtype(dtype).newbyteorder("<")
        samples = np.frombuffer(frame.data, dtype=wire).reshape(shape).astype(dtype)
        if kind is TrackKind.AUDIO and shape[0] > 1:
            samples = samples.mean(axis=0, keepdims=True).round().astype(np.int16)
        return samples

    def encode(self, data: npt.NDArray[Any], kind: TrackKind) -> step_pb2.Frame:
        """Return the samples in the track kind's layout, converted the way the media path does.

        Video is taken as ``uint8``, as the streamed path takes it. Audio keeps
        its channels: a one-dimensional array is one channel, float samples in
        ``[-1, 1]`` are scaled to ``int16``.
        """
        dtype, rank = _RAW_LAYOUT[kind]
        try:
            if kind is TrackKind.AUDIO:
                samples = _int16_audio(data)
            else:
                samples = np.asarray(data).astype(np.uint8, copy=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise FrameEncodingError(
                f"a {kind} frame of dtype {np.asarray(data).dtype} cannot be carried: {error}"
            ) from error
        if (
            samples.ndim != rank
            or 0 in samples.shape
            or (kind is TrackKind.VIDEO and samples.shape[2] != 3)
        ):
            raise FrameEncodingError(f"a {kind} frame cannot be shaped {list(np.shape(data))}")
        wire = np.dtype(dtype).newbyteorder("<")
        return step_pb2.Frame(
            encoding=step_pb2.FRAME_ENCODING_RAW,
            dtype=dtype,
            shape=list(samples.shape),
            data=np.ascontiguousarray(samples, dtype=wire).tobytes(),
        )


_INT16_MAX = 32767.0


def _int16_audio(data: npt.NDArray[Any]) -> npt.NDArray[np.int16]:
    """Return audio as ``(channels, samples)`` ``int16``, keeping every channel."""
    samples = np.asarray(data)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if np.issubdtype(samples.dtype, np.floating):
        return (samples * _INT16_MAX).clip(-_INT16_MAX - 1, _INT16_MAX).astype(np.int16)
    return np.rint(samples).astype(np.int16)


FRAME_CODECS: dict[int, FrameCodec] = {step_pb2.FRAME_ENCODING_RAW: RawFrames()}
"""The codec for each ``FrameEncoding`` the runtime reads, by its enum value."""


def check_frame(frame: step_pb2.Frame, kind: TrackKind) -> tuple[int, ...]:
    """Check one wire frame of a *kind* track without copying its samples.

    A frame that passes decodes with :func:`decode_frame`, so the copy can
    happen later, on another thread.

    Returns:
        The shape of the samples :func:`decode_frame` returns for it.

    Raises:
        FrameEncodingError: The encoding is unspecified or unknown, or the
            frame does not fit it.
    """
    return _codec(frame).check(frame, kind)


def decode_frame(frame: step_pb2.Frame, kind: TrackKind) -> InputFrame:
    """Decode one wire frame of a *kind* track into the frame a model reads.

    The frame's metadata and capture time ride along unchanged.

    Raises:
        FrameEncodingError: The encoding is unspecified or unknown, or the
            frame does not fit it.
    """
    return decoded_frame(frame, _codec(frame).decode(frame, kind))


def decoded_frame(frame: step_pb2.Frame, data: npt.NDArray[Any]) -> InputFrame:
    """Return *data*, the samples a transport decoded *frame* to, as the frame a model reads.

    The frame's metadata and capture time ride along unchanged, as they do
    through :func:`decode_frame`.
    """
    return InputFrame(
        data=data,
        metadata=frame.metadata if frame.HasField("metadata") else None,
        capture_time_us=frame.capture_time_us if frame.HasField("capture_time_us") else None,
    )


def _codec(frame: step_pb2.Frame) -> FrameCodec:
    codec = FRAME_CODECS.get(frame.encoding)
    if codec is None:
        name = step_pb2.FrameEncoding.Name(frame.encoding) if _known(frame.encoding) else None
        raise FrameEncodingError(f"frame encoding {name or frame.encoding} is not supported")
    return codec


def _known(value: int) -> bool:
    return value in step_pb2.FrameEncoding.values()


def encode_frame(
    data: npt.NDArray[Any],
    kind: TrackKind,
    *,
    metadata: bytes | None = None,
    encoding: int = step_pb2.FRAME_ENCODING_RAW,
) -> step_pb2.Frame:
    """Encode one frame of a *kind* track, with the metadata the model attached to it.

    Raises:
        FrameEncodingError: The encoding is unknown, or the array cannot be
            carried as one frame of the track kind.
    """
    codec = FRAME_CODECS.get(encoding)
    if codec is None:
        raise FrameEncodingError(f"frame encoding {encoding} is not supported")
    frame = codec.encode(data, kind)
    if metadata is not None:
        frame.metadata = metadata
    return frame


_WIRE_KINDS = {
    TrackKind.VIDEO: platform_pb2.TRACK_KIND_VIDEO,
    TrackKind.AUDIO: platform_pb2.TRACK_KIND_AUDIO,
}


def track_frames(
    bundle: MediaBundle | None, *, video_fps: float | None = None
) -> list[step_pb2.TrackFrames]:
    """Encode a step's media as one entry per output track, in the bundle's order.

    A batched video track, ``(N, H, W, 3)``, becomes its ``N`` frames in order,
    each with its own metadata; a single value of metadata goes on every one.
    An unbatched video track and an audio track are one frame each. An audio
    track with no samples, such as ``(1, 0)``, has no frames: an empty audio
    payload is how a model says it has no audio for the step, and the streamed
    path takes it the same way. A step without media has no tracks.

    Each entry carries the rate its frames play at: *video_fps*, the step's own
    rate, for a video track, falling back to the track's declared rate, and
    the track's sample rate for audio.

    Raises:
        FrameEncodingError: A track's payload cannot be carried as frames of
            its kind.
    """
    if bundle is None:
        return []
    return [
        step_pb2.TrackFrames(
            track=name,
            kind=_WIRE_KINDS[track.info.kind],
            frames=[
                encode_frame(data, track.info.kind, metadata=metadata)
                for data, metadata in _frames_of(track)
            ],
            rate=(
                video_fps if track.info.kind is TrackKind.VIDEO and video_fps else track.info.rate
            ),
        )
        for name, track in bundle.tracks.items()
    ]


def media_size(bundle: MediaBundle | None) -> int:
    """Return the bytes a step's media takes as frames, at least, without copying it.

    Video travels as one byte per element and audio as two, so this is the
    frames' data alone: a result is never smaller.
    """
    if bundle is None:
        return 0
    return sum(
        track.data.size * (1 if track.info.kind is TrackKind.VIDEO else 2)
        for track in bundle.tracks.values()
    )


def _has_no_samples(data: npt.NDArray[Any]) -> bool:
    """Whether audio is a well-formed payload of no samples, ``(0,)`` or ``(channels, 0)``.

    Any other empty shape, such as ``(0, 480)`` or ``(1, 0, 2)``, is malformed
    and is left to :func:`encode_frame` to reject.
    """
    shape = np.shape(data)
    return shape == (0,) or (len(shape) == 2 and shape[0] > 0 and shape[1] == 0)


def _frames_of(track: TrackData) -> list[tuple[npt.NDArray[Any], bytes | None]]:
    """Split one track's payload into its frames, each with the metadata it carries."""
    data = track.data
    if track.info.kind is TrackKind.AUDIO and _has_no_samples(data):
        # No audio this step: the track goes out with no frames, so the step
        # keeps its video instead of failing. Empty video stays an error.
        return []
    if track.info.kind is TrackKind.VIDEO and data.ndim == 4:
        count = data.shape[0]
        metadata = track.metadata
        if isinstance(metadata, list):
            if len(metadata) != count:
                raise FrameEncodingError(
                    f"track {track.info.name!r} carries {len(metadata)} metadata entries "
                    f"for {count} frames"
                )
            return [(data[index], metadata[index]) for index in range(count)]
        return [(data[index], metadata) for index in range(count)]
    single = track.metadata
    if isinstance(single, list):
        raise FrameEncodingError(
            f"track {track.info.name!r} carries one frame but {len(single)} metadata entries"
        )
    return [(data, single)]
