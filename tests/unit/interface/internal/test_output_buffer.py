import time

import numpy as np
import pytest

from reactor_runtime.core.values import (
    MediaBundle,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.interface.internal.output_buffer import (
    DEFAULT_FRAME_DIMENSIONS,
    OutputBuffer,
    split_batch,
)


def video_info(name: str = "main") -> TrackInfo:
    return TrackInfo(name=name, kind=TrackKind.VIDEO, direction=TrackDirection.OUT)


def audio_info(name: str = "audio") -> TrackInfo:
    return TrackInfo(name=name, kind=TrackKind.AUDIO, rate=48_000, direction=TrackDirection.OUT)


def video_bundle(data: np.ndarray, name: str = "main") -> MediaBundle:
    return MediaBundle(tracks={name: TrackData(info=video_info(name), data=data)})


class Sink:
    def __init__(self) -> None:
        self.seen: list[tuple[MediaBundle, bool, bool]] = []

    def __call__(self, bundle: MediaBundle, duplicate: bool, is_fresh_black: bool) -> None:
        self.seen.append((bundle, duplicate, is_fresh_black))


def make_buffer(name: str = "main") -> tuple[OutputBuffer, Sink]:
    buffer = OutputBuffer(output_tracks={name: video_info(name)})
    sink = Sink()
    buffer.add_callback(sink)
    buffer.set_fps(30)
    return buffer, sink


# --- split_batch ---------------------------------------------------------


def test_split_batch_leaves_an_unbatched_bundle_unchanged() -> None:
    bundle = video_bundle(np.zeros((4, 4, 3), dtype=np.uint8))
    assert split_batch(bundle) == [bundle]


def test_split_batch_splits_batched_video_into_frames() -> None:
    bundle = video_bundle(np.zeros((3, 4, 4, 3), dtype=np.uint8))
    frames = split_batch(bundle)
    assert len(frames) == 3
    assert all(frame.tracks["main"].data.shape == (4, 4, 3) for frame in frames)


def test_split_batch_collapses_a_single_frame_batch() -> None:
    frames = split_batch(video_bundle(np.zeros((1, 4, 4, 3), dtype=np.uint8)))
    assert len(frames) == 1
    assert frames[0].tracks["main"].data.shape == (4, 4, 3)


def test_split_batch_does_not_mutate_the_caller_bundle() -> None:
    bundle = video_bundle(np.zeros((1, 4, 4, 3), dtype=np.uint8))
    split_batch(bundle)
    assert bundle.tracks["main"].data.shape == (1, 4, 4, 3)


def test_split_batch_divides_audio_proportionally() -> None:
    bundle = MediaBundle(
        tracks={
            "main": TrackData(info=video_info(), data=np.zeros((2, 4, 4, 3), dtype=np.uint8)),
            "audio": TrackData(info=audio_info(), data=np.zeros((1, 100), dtype=np.int16)),
        }
    )
    frames = split_batch(bundle)
    assert len(frames) == 2
    assert frames[0].tracks["audio"].data.shape == (1, 50)


def test_split_batch_rejects_mismatched_batch_sizes() -> None:
    bundle = MediaBundle(
        tracks={
            "a": TrackData(info=video_info("a"), data=np.zeros((3, 4, 4, 3), dtype=np.uint8)),
            "b": TrackData(info=video_info("b"), data=np.zeros((2, 4, 4, 3), dtype=np.uint8)),
        }
    )
    with pytest.raises(ValueError, match="batch size"):
        split_batch(bundle)


def test_split_batch_without_video_is_unchanged() -> None:
    bundle = MediaBundle(
        tracks={"audio": TrackData(info=audio_info(), data=np.zeros((1, 100), dtype=np.int16))}
    )
    assert split_batch(bundle) == [bundle]


# --- fps -----------------------------------------------------------------


def test_set_fps_rejects_nonpositive() -> None:
    buffer, _ = make_buffer()
    with pytest.raises(ValueError, match="positive"):
        buffer.set_fps(0)


def test_set_fps_sets_the_rate() -> None:
    buffer, _ = make_buffer()
    buffer.set_fps(60)
    assert buffer.fps == 60


# --- per-tick emission logic ---------------------------------------------


def test_tick_emits_a_real_frame_not_a_duplicate() -> None:
    buffer, sink = make_buffer()
    buffer.submit(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)), drop=True)
    buffer._emit_one_tick()
    assert len(sink.seen) == 1
    _, duplicate, is_fresh_black = sink.seen[0]
    assert duplicate is False
    assert is_fresh_black is False


def test_tick_with_no_data_emits_black_at_default_dims() -> None:
    buffer, sink = make_buffer()
    buffer._emit_one_tick()
    bundle, duplicate, _ = sink.seen[0]
    assert duplicate is True
    assert bundle.tracks["main"].data.shape == DEFAULT_FRAME_DIMENSIONS


def test_tick_repeats_the_last_frame_video_only() -> None:
    buffer, sink = make_buffer()
    full = MediaBundle(
        tracks={
            "main": TrackData(info=video_info(), data=np.zeros((4, 4, 3), dtype=np.uint8)),
            "audio": TrackData(info=audio_info(), data=np.zeros((1, 50), dtype=np.int16)),
        }
    )
    buffer.submit(full, drop=True)
    buffer._emit_one_tick()  # consumes the real frame
    buffer._emit_one_tick()  # queue empty → gap-fill
    gap_fill, duplicate, _ = sink.seen[1]
    assert duplicate is True
    assert set(gap_fill.tracks) == {"main"}  # audio is not replayed


def test_black_uses_the_dimensions_of_the_last_real_frame() -> None:
    buffer, sink = make_buffer()
    buffer.submit(video_bundle(np.zeros((8, 8, 3), dtype=np.uint8)), drop=True)
    buffer._emit_one_tick()  # captures dims (8, 8, 3)
    buffer.flush()
    buffer._emit_one_tick()  # consumes the flush marker → fresh black
    bundle, _, is_fresh_black = sink.seen[-1]
    assert is_fresh_black is True
    assert bundle.tracks["main"].data.shape == (8, 8, 3)


# --- lifecycle -----------------------------------------------------------


def test_start_emission_requires_a_callback() -> None:
    buffer = OutputBuffer(output_tracks={"main": video_info()})
    buffer.set_fps(30)
    with pytest.raises(RuntimeError, match="callback"):
        buffer.start_emission()


def test_start_emission_requires_an_fps() -> None:
    buffer = OutputBuffer(output_tracks={"main": video_info()})
    buffer.add_callback(Sink())
    with pytest.raises(RuntimeError, match="FPS"):
        buffer.start_emission()


def test_emission_thread_delivers_submitted_frames() -> None:
    buffer, sink = make_buffer()
    buffer.set_fps(200)
    buffer.start_emission()
    try:
        buffer.submit(video_bundle(np.zeros((4, 4, 3), dtype=np.uint8)))
        time.sleep(0.05)
    finally:
        buffer.stop_emission()
    assert any(duplicate is False for _, duplicate, _ in sink.seen)
