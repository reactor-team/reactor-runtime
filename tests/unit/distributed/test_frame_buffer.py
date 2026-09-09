# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""The local and shared transports enforce the same bounded video contract."""

import numpy as np
import pytest

from reactor_runtime.distributed.frames import SharedFrameBuffer, _FrameBuffer


@pytest.fixture(params=[False, True])
def frames(request):
    shape = (4, 2, 3, 3)
    buffer = SharedFrameBuffer(shape, create=True) if request.param else _FrameBuffer(shape)
    yield buffer
    buffer.close()


def test_write_returns_end_row_and_read_is_owned(frames) -> None:
    assert frames.write(np.full((2, 2, 3, 3), 7, dtype=np.uint8), start_row=2) == 4
    result = frames.read(4)
    frames.array[:] = 0
    assert (result[2:] == 7).all()


@pytest.mark.parametrize("count", [-1, 5, True, 1.5])
def test_read_rejects_invalid_counts(frames, count) -> None:
    with pytest.raises(ValueError, match="frame count"):
        frames.read(count)


@pytest.mark.parametrize(
    ("shape", "dtype", "offset"),
    [
        ((5, 2, 3, 3), np.uint8, 0),
        ((2, 2, 3, 3), np.uint8, -1),
        ((2, 1, 3, 3), np.uint8, 0),
        ((2, 2, 3, 3), np.float32, 0),
    ],
)
def test_write_rejects_overflow_broadcasting_and_silent_casts(frames, shape, dtype, offset) -> None:
    with pytest.raises(ValueError, match="frame"):
        frames.write(np.zeros(shape, dtype=dtype), start_row=offset)


@pytest.mark.parametrize("shape", [(0, 2, 3, 3), (4, -1, 3, 3), (4, 2, 3), (True, 2, 3, 3)])
def test_shape_is_validated_before_allocating_shared_memory(shape) -> None:
    with pytest.raises(ValueError, match="frame_shape"):
        SharedFrameBuffer(shape, create=True)
