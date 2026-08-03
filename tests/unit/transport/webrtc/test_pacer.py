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
    CHUNKS_OF_HEADROOM,
    DEFAULT_FRAME_DIMENSIONS,
    MediaPacer,
)


def video_info(name: str = "main") -> TrackInfo:
    return TrackInfo(name=name, kind=TrackKind.VIDEO, direction=TrackDirection.OUT)


def audio_info(name: str = "audio") -> TrackInfo:
    return TrackInfo(name=name, kind=TrackKind.AUDIO, rate=48_000, direction=TrackDirection.OUT)


def video_bundle(data: np.ndarray, name: str = "main") -> MediaBundle:
    return MediaBundle(tracks={name: TrackData(info=video_info(name), data=data)})


def chunk(bundle: MediaBundle, fps: float = 30.0, n_frames: int = 1) -> MediaChunk:
    return MediaChunk(bundle=bundle, fps=fps, n_frames=n_frames)


class Sink:
    def __init__(self) -> None:
        self.seen: list[MediaBundle] = []

    def __call__(self, bundle: MediaBundle) -> None:
        self.seen.append(bundle)


def make_pacer(name: str = "main") -> tuple[MediaPacer, Sink]:
    sink = Sink()
    pacer = MediaPacer({name: video_info(name)}, sink)
    return pacer, sink


# --- submit adopts the chunk's rate and splits batches --------------------


def test_submit_splits_a_batched_chunk_into_frames() -> None:
    pacer, _ = make_pacer()
    enqueued = pacer.submit(chunk(video_bundle(np.zeros((3, 4, 4, 3), dtype=np.uint8)), n_frames=3))
    assert enqueued == 3


def test_submit_enqueues_a_single_frame() -> None:
    pacer, _ = make_pacer()
    assert pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)))) == 1


def test_submit_adopts_the_chunk_rate() -> None:
    pacer, _ = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)), fps=60.0))
    assert pacer._interval == 1.0 / 60.0


def test_submit_drops_frames_once_the_queue_is_really_full() -> None:
    pacer, _ = make_pacer()
    batch = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    # Two chunks of headroom, and nothing draining because the thread is stopped,
    # so the third batch has nowhere to go.
    for _ in range(CHUNKS_OF_HEADROOM):
        assert pacer.submit(chunk(video_bundle(batch), n_frames=10)) == 10
    assert pacer.submit(chunk(video_bundle(batch), n_frames=10)) == 0


# --- capacity follows the size of the chunks a model actually emits -------


def test_a_batch_larger_than_the_floor_is_enqueued_whole() -> None:
    # The regression: a model emitting 29 frames per inference lost everything
    # past the fixed depth, so most of every batch never reached the wire.
    pacer, _ = make_pacer()
    batch = np.zeros((29, 4, 4, 3), dtype=np.uint8)
    assert pacer.submit(chunk(video_bundle(batch), n_frames=29)) == 29


def test_capacity_grows_to_hold_a_chunk_with_headroom() -> None:
    pacer, _ = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((29, 4, 4, 3), dtype=np.uint8)), n_frames=29))
    assert pacer._capacity == CHUNKS_OF_HEADROOM * 29


def test_a_single_frame_model_keeps_the_default_floor() -> None:
    pacer, _ = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8))))
    assert pacer._capacity == 10


def test_capacity_does_not_shrink_on_a_smaller_chunk() -> None:
    pacer, _ = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((29, 4, 4, 3), dtype=np.uint8)), n_frames=29))
    grown = pacer._capacity
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8))))
    assert pacer._capacity == grown


def test_successive_batches_fit_while_the_previous_one_drains() -> None:
    # Two chunks of headroom means the next batch is accepted in full even
    # though none of the previous one has been paced out yet.
    pacer, _ = make_pacer()
    first = pacer.submit(chunk(video_bundle(np.zeros((29, 4, 4, 3), dtype=np.uint8)), n_frames=29))
    second = pacer.submit(chunk(video_bundle(np.zeros((29, 4, 4, 3), dtype=np.uint8)), n_frames=29))
    assert (first, second) == (29, 29)


# --- per-tick emission logic ----------------------------------------------


def test_tick_emits_a_queued_frame() -> None:
    pacer, sink = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8))))
    pacer._emit_one_tick()
    assert len(sink.seen) == 1
    assert sink.seen[0].tracks["main"].data.shape == (4, 4, 3)


def test_tick_with_no_data_emits_black_at_default_dims() -> None:
    pacer, sink = make_pacer()
    pacer._emit_one_tick()
    assert sink.seen[0].tracks["main"].data.shape == DEFAULT_FRAME_DIMENSIONS


def test_tick_repeats_the_last_frame_video_only() -> None:
    pacer, sink = make_pacer()
    full = MediaBundle(
        tracks={
            "main": TrackData(info=video_info(), data=np.zeros((4, 4, 3), dtype=np.uint8)),
            "audio": TrackData(info=audio_info(), data=np.zeros((1, 50), dtype=np.int16)),
        }
    )
    pacer.submit(chunk(full))
    pacer._emit_one_tick()  # consumes the real frame
    pacer._emit_one_tick()  # queue empty -> gap-fill
    assert set(sink.seen[1].tracks) == {"main"}  # audio is not replayed


def test_black_uses_the_dimensions_of_the_last_real_frame() -> None:
    pacer, sink = make_pacer()
    pacer.submit(chunk(video_bundle(np.zeros((8, 8, 3), dtype=np.uint8))))
    pacer._emit_one_tick()  # captures dims (8, 8, 3)
    pacer._emit_one_tick()  # gap-fill repeats the last frame at (8, 8, 3)
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
    real frames and then the same one repeated until the next batch, so this
    asserts on what actually arrives rather than on the queue's bookkeeping.
    """
    pacer, sink = make_pacer()
    frames_per_batch, batches = 8, 3
    fps = 400.0  # drains a batch in ~20ms, keeping the test quick

    for batch_index in range(batches):
        batch = np.zeros((frames_per_batch, 4, 4, 3), dtype=np.uint8)
        for frame_index in range(frames_per_batch):
            batch[frame_index, 0, 0, 0] = batch_index * frames_per_batch + frame_index + 1
        pacer.submit(chunk(video_bundle(batch), fps=fps, n_frames=frames_per_batch))
        pacer.start()
        time.sleep(frames_per_batch / fps * 1.5)
    pacer.stop()

    stamps = [int(bundle.tracks["main"].data[0, 0, 0]) for bundle in sink.seen]
    distinct = [
        stamp for index, stamp in enumerate(stamps) if index == 0 or stamp != stamps[index - 1]
    ]
    assert distinct == list(range(1, frames_per_batch * batches + 1))
