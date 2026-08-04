import queue
import threading
import time

import numpy as np

from reactor_runtime.core.values import (
    MediaBundle,
    MediaChunk,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.transport.webrtc.pacer import (
    DEFAULT_FRAME_DIMENSIONS,
    MediaPacer,
)


def video_info(name: str = "main") -> TrackInfo:
    return TrackInfo(name=name, kind=TrackKind.VIDEO, direction=TrackDirection.OUT)


def audio_info(name: str = "audio") -> TrackInfo:
    return TrackInfo(name=name, kind=TrackKind.AUDIO, rate=48_000, direction=TrackDirection.OUT)


def video_bundle(
    data: np.ndarray, name: str = "main", metadata: bytes | list[bytes] | None = None
) -> MediaBundle:
    return MediaBundle(
        tracks={name: TrackData(info=video_info(name), data=data, metadata=metadata)}
    )


def chunk(
    bundle: MediaBundle, fps: float = 30.0, n_frames: int = 1, wait: bool = False
) -> MediaChunk:
    return MediaChunk(bundle=bundle, fps=fps, n_frames=n_frames, wait=wait)


def batch(n: int, side: int = 4) -> np.ndarray:
    return np.zeros((n, side, side, 3), dtype=np.uint8)


class Sink:
    def __init__(self) -> None:
        self.seen: list[MediaBundle] = []

    def __call__(self, bundle: MediaBundle) -> None:
        self.seen.append(bundle)


def make_pacer(name: str = "main", queue_depth: int = 10) -> tuple[MediaPacer, Sink]:
    sink = Sink()
    pacer = MediaPacer({name: video_info(name)}, sink, queue_depth=queue_depth)
    return pacer, sink


# --- submit adopts the chunk's rate and splits batches --------------------


def test_submit_splits_a_batched_chunk_into_frames() -> None:
    pacer, _ = make_pacer()
    enqueued = pacer.submit(chunk(video_bundle(batch(3)), n_frames=3))
    assert enqueued == 3


def test_submit_enqueues_a_single_frame() -> None:
    pacer, _ = make_pacer()
    assert pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)))) == 1


def test_submit_adopts_the_chunk_rate() -> None:
    pacer, _ = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)), fps=60.0))
    assert pacer._interval == 1.0 / 60.0


def test_set_rate_repaces_queued_frames_immediately() -> None:
    pacer, _ = make_pacer()
    pacer.submit(chunk(video_bundle(batch(3)), fps=30.0, n_frames=3))
    pacer.set_rate(60.0)
    assert pacer._interval == 1.0 / 60.0


def test_a_later_chunk_supersedes_a_set_rate() -> None:
    pacer, _ = make_pacer()
    pacer.set_rate(60.0)
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)), fps=24.0))
    assert pacer._interval == 1.0 / 24.0


# --- the queue bound is the depth, floored at one chunk --------------------


def test_a_drop_chunk_loses_frames_once_the_queue_is_full() -> None:
    pacer, _ = make_pacer()
    # Nothing drains (the thread is stopped), so a second batch at the bound
    # has nowhere to go.
    assert pacer.submit(chunk(video_bundle(batch(10)), n_frames=10)) == 10
    assert pacer.submit(chunk(video_bundle(batch(10)), n_frames=10)) == 0


def test_a_batch_larger_than_the_depth_is_enqueued_whole() -> None:
    # A model emitting 29 frames per inference must never lose most of every
    # batch to a bound smaller than one of its own chunks.
    pacer, _ = make_pacer()
    assert pacer.submit(chunk(video_bundle(batch(29)), n_frames=29)) == 29


def test_capacity_does_not_grow_past_the_depth() -> None:
    pacer, _ = make_pacer()
    pacer.submit(chunk(video_bundle(batch(29)), n_frames=29))
    assert pacer.submit(chunk(video_bundle(batch(29)), n_frames=29)) == 0


def test_set_depth_rebounds_the_queue() -> None:
    pacer, _ = make_pacer()
    pacer.set_depth(2)
    assert pacer.submit(chunk(video_bundle(batch(5)), n_frames=5)) == 5  # one-chunk floor
    assert pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)))) == 0


# --- backpressure: a wait chunk throttles the producer ---------------------


def test_a_wait_chunk_blocks_until_the_drain_opens_room() -> None:
    # The one-chunk floor admits any single chunk whole, so backpressure
    # engages against frames a previous chunk left in the queue.
    pacer, sink = make_pacer(queue_depth=4)
    pacer.submit(chunk(video_bundle(batch(4)), fps=200.0, n_frames=4))
    pacer.start()
    try:
        enqueued = pacer.submit(chunk(video_bundle(batch(4)), fps=200.0, n_frames=4, wait=True))
    finally:
        pacer.stop()
    assert enqueued == 4
    assert len(sink.seen) >= 4  # the wait completed by draining, not dropping


def test_flush_releases_a_blocked_wait_submit() -> None:
    pacer, _ = make_pacer(queue_depth=4)
    # 1 fps: the queue cannot drain within the test, so only a flush can
    # release the blocked submit.
    pacer.submit(chunk(video_bundle(batch(4)), fps=1.0, n_frames=4))
    pacer.start()
    result: list[int] = []

    def submit() -> None:
        result.append(pacer.submit(chunk(video_bundle(batch(4)), fps=1.0, n_frames=4, wait=True)))

    thread = threading.Thread(target=submit)
    thread.start()
    try:
        time.sleep(0.1)
        assert thread.is_alive()  # blocked on a full queue
        pacer.flush()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        pacer.stop()
        thread.join(timeout=2.0)
    assert result  # the submit returned
    assert result[0] < 4  # the rest of the chunk was abandoned


def test_stop_releases_a_blocked_wait_submit() -> None:
    pacer, _ = make_pacer(queue_depth=4)
    pacer.submit(chunk(video_bundle(batch(4)), fps=1.0, n_frames=4))
    pacer.start()
    result: list[int] = []

    def submit() -> None:
        result.append(pacer.submit(chunk(video_bundle(batch(4)), fps=1.0, n_frames=4, wait=True)))

    thread = threading.Thread(target=submit)
    thread.start()
    time.sleep(0.1)
    pacer.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert result
    assert result[0] < 4


def test_a_drop_chunk_stops_enqueueing_past_a_flush() -> None:
    # A flush mid-chunk must abandon the rest of a drop chunk too, or the
    # flushed run's tail trickles in after the black cut.
    pacer, _ = make_pacer()

    class FlushingQueue(queue.Queue[MediaBundle]):
        """Flushes the pacer as a side effect of the first enqueue."""

        puts = 0

        def put_nowait(self, item: MediaBundle) -> None:
            super().put_nowait(item)
            FlushingQueue.puts += 1
            if FlushingQueue.puts == 1:
                pacer.flush()

    pacer._queue = FlushingQueue()
    enqueued = pacer.submit(chunk(video_bundle(batch(5)), n_frames=5))
    assert enqueued == 1  # the flush landed after the first frame
    assert pacer._queue.qsize() == 0  # and drained it


def test_a_wait_chunk_does_not_block_while_pacing_is_stopped() -> None:
    # Teardown safety: with no thread draining, a wait submit must not sleep
    # on room that will never open.
    pacer, _ = make_pacer(queue_depth=4)
    pacer.submit(chunk(video_bundle(batch(10)), n_frames=10))
    enqueued = pacer.submit(chunk(video_bundle(batch(2)), n_frames=2, wait=True))
    assert enqueued == 0


# --- per-tick emission: boundary black once, then silence ------------------


def test_tick_emits_a_queued_frame() -> None:
    pacer, sink = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8))))
    pacer._emit_one_tick()
    assert len(sink.seen) == 1
    assert sink.seen[0].tracks["main"].data.shape == (4, 4, 3)


def test_first_tick_with_no_data_emits_one_black_frame() -> None:
    pacer, sink = make_pacer()
    pacer._emit_one_tick()
    assert sink.seen[0].tracks["main"].data.shape == DEFAULT_FRAME_DIMENSIONS
    assert not sink.seen[0].tracks["main"].data.any()


def test_an_empty_queue_after_the_boundary_black_sends_nothing() -> None:
    pacer, sink = make_pacer()
    pacer._emit_one_tick()  # the one boundary black
    pacer._emit_one_tick()
    pacer._emit_one_tick()
    assert len(sink.seen) == 1


def test_underrun_after_real_frames_sends_nothing() -> None:
    # The client keeps the frame it has; no gap-fill duplicates ride the wire.
    pacer, sink = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8))))
    pacer._emit_one_tick()  # consumes the real frame
    pacer._emit_one_tick()  # queue empty
    assert len(sink.seen) == 1


def test_flush_drains_the_queue_and_rearms_the_boundary_black() -> None:
    pacer, sink = make_pacer()
    pacer.submit(chunk(video_bundle(batch(5, side=8)), n_frames=5))
    pacer._emit_one_tick()  # one real frame, capturing (8, 8, 3)
    pacer.flush()
    pacer._emit_one_tick()  # the flush cut: one black frame
    pacer._emit_one_tick()  # then silence
    assert len(sink.seen) == 2
    assert not sink.seen[1].tracks["main"].data.any()


def test_each_frame_of_a_batch_is_paced_out_with_its_own_metadata() -> None:
    pacer, sink = make_pacer()
    batch = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    pacer.submit(chunk(video_bundle(batch, metadata=[b"a", b"b", b"c"]), n_frames=3))
    for _ in range(3):
        pacer._emit_one_tick()
    assert [bundle.tracks["main"].metadata for bundle in sink.seen] == [b"a", b"b", b"c"]


def test_a_gap_fill_repeat_carries_the_metadata_of_the_frame_it_repeats() -> None:
    # The repeat shows the same picture, so a client reading the metadata to
    # interpret it must be told the same thing about it.
    pacer, sink = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8), metadata=b"pose-7")))
    pacer._emit_one_tick()  # consumes the real frame
    pacer._emit_one_tick()  # queue empty -> gap-fill
    assert [bundle.tracks["main"].metadata for bundle in sink.seen] == [b"pose-7", b"pose-7"]


def test_a_black_frame_carries_no_metadata() -> None:
    pacer, sink = make_pacer()
    pacer._emit_one_tick()
    assert sink.seen[0].tracks["main"].metadata is None


def test_black_uses_the_dimensions_of_the_last_real_frame() -> None:
    pacer, sink = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((8, 8, 3), dtype=np.uint8))))
    pacer._emit_one_tick()  # captures dims (8, 8, 3)
    pacer.flush()
    pacer._emit_one_tick()  # boundary black at the captured dims
    assert sink.seen[-1].tracks["main"].data.shape == (8, 8, 3)


# --- lifecycle ------------------------------------------------------------


def test_pacing_thread_delivers_submitted_frames() -> None:
    pacer, sink = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)), fps=200.0))
    pacer.start()
    try:
        time.sleep(0.05)
    finally:
        pacer.stop()
    assert any(bundle.tracks["main"].data.shape == (4, 4, 3) for bundle in sink.seen)


def test_stop_is_idempotent_before_start() -> None:
    pacer, _ = make_pacer()
    pacer.stop()  # never started; must not raise


def test_a_batching_model_keeps_the_wire_in_fresh_frames() -> None:
    """Every frame of every batch reaches the sink, in order and exactly once.

    The symptom of a queue that cannot hold a chunk is a wire fed a handful of
    real frames and then nothing until the next batch, so this asserts on what
    actually arrives rather than on the queue's bookkeeping.
    """
    pacer, sink = make_pacer()
    frames_per_batch, batches = 8, 3
    fps = 400.0  # drains a batch in ~20ms, keeping the test quick

    for batch_index in range(batches):
        stamped = batch(frames_per_batch)
        for frame_index in range(frames_per_batch):
            stamped[frame_index, 0, 0, 0] = batch_index * frames_per_batch + frame_index + 1
        pacer.submit(chunk(video_bundle(stamped), fps=fps, n_frames=frames_per_batch))
        pacer.start()
        time.sleep(frames_per_batch / fps * 1.5)
    pacer.stop()

    stamps = [int(bundle.tracks["main"].data[0, 0, 0]) for bundle in sink.seen]
    distinct = [
        stamp for index, stamp in enumerate(stamps) if index == 0 or stamp != stamps[index - 1]
    ]
    assert distinct == list(range(1, frames_per_batch * batches + 1))
