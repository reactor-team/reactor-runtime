import time

import numpy as np
import pytest

from reactor_runtime.core.values import ConnId, InputFrame
from reactor_runtime.interface.internal.input_buffer import (
    BufferClosed,
    InputBuffer,
    ReadMode,
)


def frame(value: int = 0) -> InputFrame:
    return InputFrame(data=np.full((2, 2, 3), value, dtype=np.uint8), pts=float(value))


def test_try_read_returns_none_when_too_few() -> None:
    buffer = InputBuffer()
    assert buffer.try_read(1) is None


def test_push_then_try_read_latest() -> None:
    buffer = InputBuffer()
    buffer.push(frame(1))
    buffer.push(frame(2))
    read = buffer.try_read(1, mode=ReadMode.LATEST)
    assert read is not None
    assert read[0].pts == 2.0
    assert buffer.available == 0  # LATEST discards the backlog


def test_try_read_fifo_preserves_order_and_leaves_rest() -> None:
    buffer = InputBuffer()
    buffer.push(frame(1))
    buffer.push(frame(2))
    read = buffer.try_read(1, mode=ReadMode.FIFO)
    assert read is not None
    assert read[0].pts == 1.0
    assert buffer.available == 1


def test_total_received_counts_every_push() -> None:
    buffer = InputBuffer()
    buffer.push(frame())
    buffer.push(frame())
    buffer.try_read(2)
    assert buffer.total_received == 2


def test_maxlen_evicts_the_oldest() -> None:
    buffer = InputBuffer(maxlen=2)
    buffer.push(frame(1))
    buffer.push(frame(2))
    buffer.push(frame(3))
    assert buffer.available == 2
    read = buffer.try_read(2, mode=ReadMode.FIFO)
    assert read is not None
    assert [f.pts for f in read] == [2.0, 3.0]


def test_close_drops_pushes_and_raises_on_read() -> None:
    buffer = InputBuffer()
    buffer.close()
    buffer.push(frame())
    assert buffer.available == 0
    with pytest.raises(BufferClosed):
        buffer.try_read(1)


def test_reset_reopens() -> None:
    buffer = InputBuffer()
    buffer.push(frame())
    buffer.close()
    buffer.reset()
    assert buffer.closed is False
    assert buffer.available == 0
    buffer.push(frame(5))
    read = buffer.try_read(1)
    assert read is not None
    assert read[0].pts == 5.0


async def test_read_waits_for_frames() -> None:
    buffer = InputBuffer()
    buffer.push(frame(1))
    buffer.push(frame(2))
    read = await buffer.read(2, mode=ReadMode.FIFO)
    assert [f.pts for f in read] == [1.0, 2.0]


async def test_read_times_out() -> None:
    buffer = InputBuffer()
    with pytest.raises(TimeoutError):
        await buffer.read(1, timeout=0.05)


async def test_read_raises_when_closed_while_waiting() -> None:
    buffer = InputBuffer()
    buffer.close()
    with pytest.raises(BufferClosed):
        await buffer.read(1, timeout=0.5)


async def test_read_rejects_more_than_capacity() -> None:
    buffer = InputBuffer(maxlen=4)
    with pytest.raises(ValueError, match="at most"):
        await buffer.read(5)


def test_a_read_notes_how_long_its_frames_waited() -> None:
    buf = InputBuffer(maxlen=4)
    buf.push(frame(1))
    buf.push(frame(2))
    time.sleep(0.005)

    assert buf.try_read(1) is not None

    [(source, wait)] = buf.take_read_waits()
    assert source is None
    assert wait >= 5.0
    assert buf.take_read_waits() == []


def test_a_read_names_the_connection_that_sent_each_frame() -> None:
    buf = InputBuffer(maxlen=4)
    buf.push(frame(1), ConnId(1001))
    buf.push(frame(2), ConnId(1002))

    buf.try_read(2, mode=ReadMode.FIFO)

    assert [source for source, _ in buf.take_read_waits()] == [ConnId(1001), ConnId(1002)]


def test_a_fifo_read_notes_each_frame_it_took() -> None:
    buf = InputBuffer(maxlen=4)
    for value in range(3):
        buf.push(frame(value))

    buf.try_read(2, mode=ReadMode.FIFO)

    assert len(buf.take_read_waits()) == 2
