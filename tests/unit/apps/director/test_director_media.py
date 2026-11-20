"""Reading a built shot: frames, slices, last frame, PNG, and a file decoded onto tracks."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from director_harness import encode_mp4, track

from reactor_runtime import Audio, Output, TrackPayload, Video
from reactor_runtime.apps.director.media import (
    decode_file,
    frame_count,
    last_frame,
    png_bytes,
    read_file_video,
    slice_output,
)


class Clip(Output):
    main_video: Video
    main_audio: Audio


class Picture(Output):
    main_video: Video


class Sound(Output):
    main_audio: Audio


def _clip(frames: int, fps: float = 24.0) -> Clip:
    video = np.arange(frames, dtype=np.uint8)[:, None, None, None] * np.ones((1, 4, 4, 3), np.uint8)
    samples = round(frames * 48_000 / fps)
    return Clip(main_video=video, main_audio=np.arange(samples, dtype=np.int16).reshape(1, -1))


def test_a_batched_video_track_counts_its_frames() -> None:
    assert frame_count(_clip(10), 24) == 10


def test_an_unbatched_frame_is_one_frame() -> None:
    assert frame_count(Picture(main_video=np.zeros((4, 4, 3), np.uint8)), 24) == 1


def test_audio_alone_lasts_as_long_as_its_samples() -> None:
    sound = Sound(main_audio=np.zeros((1, 48_000), np.int16))
    assert frame_count(sound, 24) == 24


def test_video_tracks_that_disagree_on_length_are_refused() -> None:
    class Two(Output):
        left: Video
        right: Video

    two = Two(left=np.zeros((3, 2, 2, 3), np.uint8), right=np.zeros((4, 2, 2, 3), np.uint8))
    with pytest.raises(ValueError, match="disagree"):
        frame_count(two, 24)


def test_slices_hold_every_frame_and_every_sample_once() -> None:
    clip = _clip(10, fps=30)
    slices = [slice_output(clip, lo, min(lo + 3, 10), 30) for lo in range(0, 10, 3)]
    video = np.concatenate([track(s, "main_video") for s in slices])
    audio = np.concatenate([track(s, "main_audio") for s in slices], axis=-1)
    np.testing.assert_array_equal(video, track(clip, "main_video"))
    np.testing.assert_array_equal(audio, track(clip, "main_audio"))


def test_per_frame_metadata_is_sliced_with_its_frames() -> None:
    metadata: list[dict[str, Any] | bytes] = [{"i": i} for i in range(4)]
    clip = Picture(main_video=TrackPayload(np.zeros((4, 2, 2, 3), np.uint8), metadata=metadata))
    assert slice_output(clip, 1, 3, 24).__metadata__["main_video"] == [{"i": 1}, {"i": 2}]


def test_the_last_frame_is_a_copy_of_the_first_video_track_last_frame() -> None:
    clip = _clip(5)
    frame = last_frame(clip)
    assert frame is not None
    np.testing.assert_array_equal(frame, track(clip, "main_video")[-1])
    frame[...] = 99
    assert track(clip, "main_video")[-1].max() == 4
    assert last_frame(Sound(main_audio=np.zeros((1, 10), np.int16))) is None


def test_png_bytes_decode_back_to_the_frame() -> None:
    import av

    frame = np.random.default_rng(1).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    with av.open(io.BytesIO(png_bytes(frame)), mode="r") as container:
        decoded = next(container.decode(video=0)).to_ndarray(format="rgb24")
    np.testing.assert_array_equal(decoded, frame)


def test_a_file_reports_its_last_frame_and_length(tmp_path: Path) -> None:
    frames = np.full((6, 16, 16, 3), 200, np.uint8)
    path = encode_mp4(tmp_path / "shot.mp4", frames, fps=24)
    last, count = read_file_video(path)
    assert count == 6
    assert last is not None
    assert last.shape == (16, 16, 3)
    assert abs(int(last.mean()) - 200) < 10


def test_a_file_that_is_not_media_reports_nothing(tmp_path: Path) -> None:
    path = tmp_path / "notes.mp4"
    path.write_bytes(b"not a video")
    assert read_file_video(path) == (None, None)


def test_a_file_decodes_onto_one_video_and_one_audio_track(tmp_path: Path) -> None:
    frames = np.full((12, 16, 16, 3), 90, np.uint8)
    audio = (np.sin(np.arange(24_000) / 10) * 8000).astype(np.int16)
    path = encode_mp4(tmp_path / "shot.mp4", frames, fps=24, audio=audio)
    clip = decode_file(path, Clip, 24)
    video, sound = track(clip, "main_video"), track(clip, "main_audio")
    assert video.shape == (12, 16, 16, 3)
    assert sound.dtype == np.int16
    assert sound.shape[0] == 1
    assert abs(sound.shape[1] - 24_000) < 2_048


def test_a_file_without_sound_plays_silence_for_its_length(tmp_path: Path) -> None:
    path = encode_mp4(tmp_path / "shot.mp4", np.zeros((6, 16, 16, 3), np.uint8), fps=24)
    sound = track(decode_file(path, Clip, 24), "main_audio")
    assert sound.shape == (1, 12_000)
    assert not sound.any()


def test_decoding_onto_two_video_tracks_is_refused(tmp_path: Path) -> None:
    class Two(Output):
        left: Video
        right: Video

    path = encode_mp4(tmp_path / "shot.mp4", np.zeros((2, 16, 16, 3), np.uint8), fps=24)
    with pytest.raises(ValueError, match="more than one"):
        decode_file(path, Two, 24)
