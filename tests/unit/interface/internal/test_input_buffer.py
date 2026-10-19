import asyncio
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


def test_waits_that_are_never_taken_stay_within_the_buffers_bound() -> None:
    buf = InputBuffer(maxlen=2)
    for value in range(5):
        buf.push(frame(value), ConnId(1000 + value))
        buf.try_read(1)

    assert [source for source, _ in buf.take_read_waits()] == [ConnId(1003), ConnId(1004)]


def audio(index: int, samples: int = 480) -> InputFrame:
    """One 10 ms frame of 48 kHz mono audio whose samples encode its index."""
    data = ((np.arange(samples) + index * samples) % 30_000).astype(np.int16).reshape(1, samples)
    return InputFrame(data=data, pts=float(index))


def test_the_default_bound_keeps_the_newest_128_audio_frames_untouched() -> None:
    buffer = InputBuffer()
    pushed = [audio(i) for i in range(200)]
    for item in pushed:
        buffer.push(item)

    read = buffer.try_read(128, mode=ReadMode.FIFO)

    assert read is not None
    # The same objects, in arrival order: nothing is copied, resampled, or reordered.
    assert read == pushed[72:]
    samples = np.concatenate([f.data for f in read], axis=1)
    assert np.array_equal(samples, np.concatenate([f.data for f in pushed[72:]], axis=1))
    assert buffer.available == 0


async def test_the_default_bound_refuses_a_read_above_128_frames() -> None:
    buffer = InputBuffer()
    with pytest.raises(ValueError, match="at most 128"):
        await buffer.read(129)


def test_a_latest_read_of_audio_returns_the_newest_frames_in_order() -> None:
    buffer = InputBuffer()
    pushed = [audio(i) for i in range(10)]
    for item in pushed:
        buffer.push(item)

    assert buffer.try_read(3, mode=ReadMode.LATEST) == pushed[7:]
    assert buffer.available == 0


def test_a_push_reports_the_frame_the_default_bound_evicts() -> None:
    buffer = InputBuffer(maxlen=2)

    assert [buffer.push(audio(i)) for i in range(4)] == [0, 0, 1, 1]


def test_a_push_to_a_closed_buffer_reports_no_eviction() -> None:
    buffer = InputBuffer(maxlen=1)
    buffer.push(audio(0))
    buffer.close()

    assert buffer.push(audio(1)) == 0


def test_a_sample_bound_holds_that_many_samples_of_the_newest_audio() -> None:
    # 50 ms at 48 kHz: five 10 ms frames.
    buffer = InputBuffer(max_samples=2_400)
    pushed = [audio(i) for i in range(8)]

    evicted = [buffer.push(item) for item in pushed]

    assert evicted == [0, 0, 0, 0, 0, 1, 1, 1]
    assert buffer.try_read(5, mode=ReadMode.FIFO) == pushed[3:]


def test_a_sample_bound_is_the_same_length_whatever_the_frame_size() -> None:
    # 100 ms frames, as a transport that delivers larger frames sends them.
    buffer = InputBuffer(max_samples=48_000)
    for i in range(15):
        buffer.push(audio(i, samples=4_800))

    assert buffer.available == 10


def test_a_sample_bound_lifts_the_128_frame_read_limit() -> None:
    buffer = InputBuffer(max_samples=48_000 * 120)
    pushed = [audio(i) for i in range(300)]
    for item in pushed:
        assert buffer.push(item) == 0

    assert buffer.try_read(300, mode=ReadMode.FIFO) == pushed


def test_a_sample_bound_keeps_a_frame_larger_than_the_bound() -> None:
    buffer = InputBuffer(max_samples=480)
    buffer.push(audio(0))

    assert buffer.push(audio(1, samples=960)) == 1
    assert buffer.available == 1


def test_a_sample_bound_counts_a_zero_sample_frame_as_no_audio() -> None:
    buffer = InputBuffer(max_samples=960)
    buffer.push(audio(0))
    buffer.push(InputFrame(data=np.zeros((1, 0), dtype=np.int16)))

    assert buffer.push(audio(1)) == 0
    assert buffer.available == 3


@pytest.mark.parametrize("mode", [ReadMode.FIFO, ReadMode.LATEST])
def test_a_sample_bound_frees_the_samples_a_read_takes(mode: ReadMode) -> None:
    buffer = InputBuffer(max_samples=960)
    buffer.push(audio(0))
    buffer.push(audio(1))
    buffer.try_read(2, mode=mode)

    assert buffer.push(audio(2)) == 0
    assert buffer.push(audio(3)) == 0


@pytest.mark.parametrize("empty", ["clear", "reset"])
def test_a_sample_bound_starts_over_after_the_buffer_empties(empty: str) -> None:
    buffer = InputBuffer(max_samples=960)
    buffer.push(audio(0))
    buffer.push(audio(1))
    if empty == "reset":
        buffer.close()
    getattr(buffer, empty)()

    assert buffer.push(audio(2)) == 0
    assert buffer.push(audio(3)) == 0


def test_a_sample_bound_drops_the_wait_of_each_frame_it_evicts() -> None:
    buffer = InputBuffer(max_samples=960)
    for index in range(4):
        buffer.push(audio(index), ConnId(1000 + index))

    buffer.try_read(2, mode=ReadMode.FIFO)

    assert [source for source, _ in buffer.take_read_waits()] == [ConnId(1002), ConnId(1003)]


def test_a_push_counts_only_the_frames_it_evicts_from_its_own_session() -> None:
    buffer = InputBuffer(maxlen=2)
    buffer.push(frame(1), session=1)
    buffer.push(frame(2), session=1)

    # A new session's frames push the old session's out without charging it.
    assert buffer.push(frame(3), session=2) == 0
    assert buffer.push(frame(4), session=2) == 0
    assert buffer.push(frame(5), session=2) == 1


def test_a_reset_for_an_ended_session_keeps_a_later_sessions_frames() -> None:
    buffer = InputBuffer(max_samples=48_000)
    buffer.push(audio(0), session=1)
    buffer.push(audio(1), session=1)
    buffer.push(audio(2), session=2)

    buffer.reset(through_session=1)

    assert buffer.available == 1
    assert buffer.total_received == 1
    kept = buffer.try_read(1)
    assert kept is not None
    assert kept[0].pts == 2.0


def test_a_reset_with_no_session_drops_every_frame() -> None:
    buffer = InputBuffer()
    buffer.push(frame(1), session=2)

    buffer.reset()

    assert buffer.available == 0


async def test_a_read_past_the_frame_cap_fails_at_once() -> None:
    buffer = InputBuffer(max_samples=2_400, max_frames=10)

    with pytest.raises(ValueError, match="holds at most 10"):
        await buffer.read(11, timeout=5.0)


async def test_a_read_more_frames_than_the_samples_can_hold_waits() -> None:
    # 10 ms frames under a 50 ms bound: five fit, so a read of six waits for
    # smaller frames, or for its timeout.
    buffer = InputBuffer(max_samples=2_400, max_frames=50)
    for index in range(5):
        buffer.push(audio(index))

    with pytest.raises(TimeoutError):
        await buffer.read(6, timeout=0.05)


async def test_a_read_completes_once_smaller_frames_fit() -> None:
    buffer = InputBuffer(max_samples=2_400, max_frames=50)
    buffer.push(audio(0))

    read = asyncio.create_task(buffer.read(6, timeout=5.0, mode=ReadMode.FIFO))
    await asyncio.sleep(0.05)
    assert not read.done()
    for index in range(1, 6):
        buffer.push(audio(index, samples=240))

    assert [frame.data.shape[-1] for frame in await read] == [480, 240, 240, 240, 240, 240]


async def test_a_read_after_a_larger_frame_was_evicted_waits_for_smaller_ones() -> None:
    buffer = InputBuffer(max_samples=100)
    buffer.push(audio(0, samples=80))
    buffer.push(audio(1, samples=30))

    read = asyncio.create_task(buffer.read(2, timeout=5.0, mode=ReadMode.FIFO))
    await asyncio.sleep(0.05)
    assert not read.done()
    buffer.push(audio(2, samples=30))

    assert [frame.data.shape[-1] for frame in await read] == [30, 30]


def test_frames_with_no_samples_are_held_to_the_frame_cap() -> None:
    buffer = InputBuffer(max_samples=2_400, max_frames=10)
    for _ in range(50):
        buffer.push(InputFrame(data=np.zeros((1, 0), dtype=np.int16)))

    assert buffer.available == 10
