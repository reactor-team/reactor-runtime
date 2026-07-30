from typing import Any

import numpy as np
import pytest

from reactor_runtime.interface.engine.frames import to_video_frames


class _FakeTensor:
    """Stands in for a device tensor: detach, cast, move, convert."""

    def __init__(self, array: np.ndarray, floating: bool = True) -> None:
        self._array = array
        self.dtype = type("Dtype", (), {"is_floating_point": floating})()
        self.moved = False

    def detach(self) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return _FakeTensor(self._array.astype(np.float32))

    def cpu(self) -> "_FakeTensor":
        self.moved = True
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def _chw(value: float, frames: int = 2) -> np.ndarray:
    return np.full((frames, 3, 4, 5), value, dtype=np.float32)


# -- layout --------------------------------------------------------------------


def test_channels_first_is_moved_last() -> None:
    assert to_video_frames(_chw(1.0)).shape == (2, 4, 5, 3)


def test_channels_last_is_left_alone() -> None:
    chunk = np.zeros((2, 4, 5, 3), dtype=np.uint8)

    assert to_video_frames(chunk).shape == (2, 4, 5, 3)


def test_a_single_frame_keeps_its_shape() -> None:
    assert to_video_frames(np.zeros((3, 4, 5), dtype=np.uint8)).shape == (4, 5, 3)


def test_a_leading_batch_axis_of_one_is_dropped() -> None:
    assert to_video_frames(np.zeros((1, 2, 3, 4, 5), dtype=np.uint8)).shape == (2, 4, 5, 3)


def test_a_chunk_with_no_channel_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="no RGB channel axis"):
        to_video_frames(np.zeros((2, 4, 4, 4), dtype=np.uint8))


def test_more_than_one_sequence_of_frames_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than one sequence"):
        to_video_frames(np.zeros((2, 2, 3, 4, 5), dtype=np.uint8))


# -- value range ---------------------------------------------------------------


def test_the_default_range_is_the_one_these_decoders_emit() -> None:
    frames = to_video_frames(_chw(-1.0))

    assert frames.dtype == np.uint8
    assert frames.max() == 0


def test_the_top_of_the_range_saturates() -> None:
    assert to_video_frames(_chw(1.0)).min() == 255


def test_the_middle_of_the_range_is_mid_grey() -> None:
    assert to_video_frames(_chw(0.0)).max() == 128


def test_a_declared_range_is_honoured() -> None:
    frames = to_video_frames(_chw(1.0), value_range=(0.0, 1.0))

    assert frames.min() == 255


def test_a_value_outside_the_range_is_clipped() -> None:
    frames = to_video_frames(_chw(4.0))

    assert frames.max() == 255


def test_an_integer_chunk_is_taken_as_display_range() -> None:
    chunk = np.full((1, 3, 2, 2), 200, dtype=np.uint8)

    assert to_video_frames(chunk).max() == 200


# -- tensors -------------------------------------------------------------------


def test_a_tensor_is_brought_into_host_memory() -> None:
    tensor: Any = _FakeTensor(np.zeros((2, 3, 4, 5), dtype=np.float32))

    assert to_video_frames(tensor).shape == (2, 4, 5, 3)


def test_a_floating_tensor_is_widened_before_conversion() -> None:
    # NumPy has no bfloat16, so a floating tensor is cast through float() first.
    tensor: Any = _FakeTensor(np.zeros((2, 3, 4, 5), dtype=np.float32))
    to_video_frames(tensor)

    assert to_video_frames(tensor).dtype == np.uint8


def test_an_integer_tensor_is_not_widened() -> None:
    tensor: Any = _FakeTensor(np.full((3, 4, 5), 7, dtype=np.uint8), floating=False)

    assert to_video_frames(tensor).max() == 7
