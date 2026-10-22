"""Decoding the frames a step request carries."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_runtime.core.values import TrackKind
from reactor_runtime.protocol.frames import FrameEncodingError, check_frame, decode_frame
from reactor_wire.v1 import step_pb2

RAW = step_pb2.FRAME_ENCODING_RAW


def _raw(
    array: np.ndarray, *, metadata: bytes | None = None, capture_time_us: int | None = None
) -> step_pb2.Frame:
    return step_pb2.Frame(
        encoding=RAW,
        dtype=str(array.dtype),
        shape=list(array.shape),
        data=array.astype(array.dtype.newbyteorder("<")).tobytes(),
        metadata=metadata,
        capture_time_us=capture_time_us,
    )


def test_a_raw_video_frame_decodes_to_the_rgb_array_it_carried() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    frame = decode_frame(_raw(image), TrackKind.VIDEO)
    assert frame.data.dtype == np.uint8
    np.testing.assert_array_equal(frame.data, image)
    assert frame.metadata is None
    assert frame.capture_time_us is None


def test_a_raw_mono_audio_frame_decodes_to_the_samples_it_carried() -> None:
    pcm = np.array([[1, -2, 300]], dtype=np.int16)
    frame = decode_frame(_raw(pcm), TrackKind.AUDIO)
    assert frame.data.dtype == np.int16
    np.testing.assert_array_equal(frame.data, pcm)


def test_a_raw_audio_frame_with_several_channels_is_mixed_down_to_mono() -> None:
    pcm = np.array([[1, -2, 300], [5, 6, -32768]], dtype=np.int16)
    frame = decode_frame(_raw(pcm), TrackKind.AUDIO)
    assert frame.data.dtype == np.int16
    np.testing.assert_array_equal(frame.data, np.array([[3, 2, -16234]], dtype=np.int16))


def test_a_decoded_frame_is_writable_and_owns_its_samples() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    frame = decode_frame(_raw(image), TrackKind.VIDEO)
    frame.data[0, 0, 0] = 9
    assert frame.data[0, 0, 0] == 9


def test_metadata_and_capture_time_ride_along() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    frame = decode_frame(_raw(image, metadata=b'{"seed":1}', capture_time_us=42), TrackKind.VIDEO)
    assert frame.metadata == b'{"seed":1}'
    assert frame.capture_time_us == 42


@pytest.mark.parametrize("encoding", [step_pb2.FRAME_ENCODING_UNSPECIFIED, 99])
def test_an_unknown_encoding_is_refused(encoding: int) -> None:
    frame = _raw(np.zeros((1, 1, 3), dtype=np.uint8))
    # Proto3 enums are open: a peer may send a value this build has no name for.
    setattr(frame, "encoding", encoding)  # noqa: B010
    with pytest.raises(FrameEncodingError, match="not supported"):
        decode_frame(frame, TrackKind.VIDEO)


@pytest.mark.parametrize(
    ("array", "kind", "match"),
    [
        (np.zeros((1, 1, 3), dtype=np.float32), TrackKind.VIDEO, "uint8"),
        (np.zeros((1, 1, 4), dtype=np.uint8), TrackKind.VIDEO, r"\(height, width, 3\)"),
        (np.zeros((2, 3), dtype=np.uint8), TrackKind.VIDEO, "3 non-zero dimensions"),
        (np.zeros((1, 3), dtype=np.uint8), TrackKind.AUDIO, "int16"),
        (np.zeros((4,), dtype=np.int16), TrackKind.AUDIO, "2 non-zero dimensions"),
        (np.zeros((1, 0), dtype=np.int16), TrackKind.AUDIO, "2 non-zero dimensions"),
    ],
)
def test_a_frame_that_does_not_fit_the_track_kind_is_refused(
    array: np.ndarray, kind: TrackKind, match: str
) -> None:
    with pytest.raises(FrameEncodingError, match=match):
        decode_frame(_raw(array), kind)


def test_bytes_that_do_not_fill_the_shape_are_refused() -> None:
    frame = _raw(np.zeros((2, 2, 3), dtype=np.uint8))
    frame.data = frame.data[:-1]
    with pytest.raises(FrameEncodingError, match="is 12 bytes, got 11"):
        decode_frame(frame, TrackKind.VIDEO)


@pytest.mark.parametrize(
    ("array", "kind"),
    [
        (np.zeros((2, 3, 3), dtype=np.uint8), TrackKind.VIDEO),
        (np.zeros((1, 5), dtype=np.int16), TrackKind.AUDIO),
        (np.zeros((2, 5), dtype=np.int16), TrackKind.AUDIO),
    ],
)
def test_a_checked_frame_has_the_shape_it_decodes_to(array: np.ndarray, kind: TrackKind) -> None:
    frame = _raw(array)
    assert check_frame(frame, kind) == decode_frame(frame, kind).data.shape


def test_a_check_refuses_what_decoding_refuses() -> None:
    frame = _raw(np.zeros((2, 3, 3), dtype=np.uint8))
    frame.data = frame.data[:-1]
    with pytest.raises(FrameEncodingError, match="bytes"):
        check_frame(frame, TrackKind.VIDEO)
