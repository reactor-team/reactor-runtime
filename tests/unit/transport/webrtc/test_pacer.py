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
from reactor_runtime.transport.webrtc.pacer import DEFAULT_FRAME_DIMENSIONS, MediaPacer


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


def test_submit_drops_frames_that_overflow_the_queue() -> None:
    pacer, _ = make_pacer()
    pacer._queue.maxsize = 2
    # A ten-frame batch cannot fit a depth-2 queue while the thread is stopped.
    enqueued = pacer.submit(
        chunk(video_bundle(np.zeros((10, 4, 4, 3), dtype=np.uint8)), n_frames=10)
    )
    assert enqueued == 2


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
