import contextlib
import threading
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from reactor_runtime.core import RecordingConfig
from reactor_runtime.recording.chunk_encoder import (
    _X264_THREADS,
    ChunkEncoder,
    EncoderStoppedError,
    _video_options,
)

_FRAME_RATE = 30


def _encoder(tmp_path: Path, *, has_audio: bool = False, chunk_seconds: int = 1) -> ChunkEncoder:
    return ChunkEncoder(
        output_dir=tmp_path,
        config=RecordingConfig(enabled=True, chunk_seconds=chunk_seconds),
        has_audio=has_audio,
        audio_sample_rate=48_000,
        frame_rate=_FRAME_RATE,
    )


def _frame(size: int = 64) -> np.ndarray[Any, Any]:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _decode(output_dir: Path) -> av.container.InputContainer:
    """Open init plus every segment as one file, the way a player receives them."""
    parts = [output_dir / "init.mp4", *sorted(output_dir.glob("chunk_*.m4s"))]
    merged = output_dir / "merged.mp4"
    merged.write_bytes(b"".join(part.read_bytes() for part in parts))
    return av.open(str(merged))


def _frame_count(container: av.container.InputContainer) -> int:
    """Frames the decoder produces; a fragmented header carries no frame count."""
    return sum(1 for _ in container.decode(video=0))


def _seconds(stream: av.VideoStream | av.AudioStream) -> float:
    """The stream's decoded duration in seconds."""
    assert stream.duration is not None
    assert stream.time_base is not None
    return float(stream.duration * stream.time_base)


# -- the encoded timeline ---------------------------------------------------


def test_video_time_derives_from_the_grid_not_from_wall_clock(tmp_path: Path) -> None:
    # The recorder resamples each chunk onto a fixed grid and the encoder stamps
    # frames by position, so a stall between feeds records no dead air. A
    # wall-clock timeline would stretch this recording by the sleep.
    encoder = _encoder(tmp_path)
    for index in range(_FRAME_RATE):
        if index == _FRAME_RATE // 2:
            time.sleep(0.5)
        encoder.feed_video(_frame())
    encoder.stop()

    with _decode(tmp_path) as container:
        assert container.duration is not None
        assert container.duration / av.time_base == pytest.approx(1.0, abs=0.05)
        assert _frame_count(container) == _FRAME_RATE


def test_audio_time_advances_with_the_samples_fed(tmp_path: Path) -> None:
    # Each grid slot carries sample_rate / frame_rate samples, so the audio
    # timeline lands on the video timeline rather than drifting off it.
    encoder = _encoder(tmp_path, has_audio=True)
    samples = np.zeros(48_000 // _FRAME_RATE, dtype=np.int16)
    for _ in range(_FRAME_RATE):
        encoder.feed_video(_frame())
        encoder.feed_audio(samples)
    encoder.stop()

    with _decode(tmp_path) as container:
        skew = _seconds(container.streams.video[0]) - _seconds(container.streams.audio[0])
        assert abs(skew) < 0.05


def test_a_video_only_recording_still_carries_an_audio_track(tmp_path: Path) -> None:
    # Players and uploaders reject a segment with no audio, so a silent track
    # stands in for a model that emits none.
    encoder = _encoder(tmp_path)
    for _ in range(_FRAME_RATE):
        encoder.feed_video(_frame())
    encoder.stop()

    with _decode(tmp_path) as container:
        assert len(container.streams.audio) == 1


def test_segments_close_on_the_configured_boundary(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path, chunk_seconds=1)
    for _ in range(_FRAME_RATE * 3):
        encoder.feed_video(_frame())
    encoder.stop()

    # A hard GOP puts a keyframe on every boundary, so the muxer closes a segment
    # per chunk_seconds rather than carrying frames into the next one.
    assert len(list(tmp_path.glob("chunk_*.m4s"))) == 3


def test_segments_land_on_disk_before_the_recording_ends(tmp_path: Path) -> None:
    # The recorder serves a segment as soon as it is written, so segments have
    # to appear while the recording is still running.
    encoder = _encoder(tmp_path, chunk_seconds=1)
    try:
        for _ in range(_FRAME_RATE * 2):
            encoder.feed_video(_frame())
        assert list(tmp_path.glob("chunk_*.m4s"))
    finally:
        encoder.stop()


def test_a_segment_appears_under_its_final_name_only_once_whole(tmp_path: Path) -> None:
    # The muxer writes a segment file in pieces. The recorder serves any segment
    # under its final name, so every size seen there must be the final one.
    seen: dict[str, set[int]] = {}
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            for path in tmp_path.glob("chunk_*.m4s"):
                with contextlib.suppress(OSError):
                    seen.setdefault(path.name, set()).add(path.stat().st_size)

    watcher = threading.Thread(target=watch)
    watcher.start()
    encoder = _encoder(tmp_path, chunk_seconds=1)
    rng = np.random.default_rng(0)
    try:
        # Noise keeps each segment well above one write buffer.
        for _ in range(_FRAME_RATE * 3):
            encoder.feed_video(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    finally:
        encoder.stop()
        stop.set()
        watcher.join()

    assert len(seen) == 3
    for name, sizes in seen.items():
        assert sizes == {(tmp_path / name).stat().st_size}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_later_part_continues_the_numbering_and_the_timeline(tmp_path: Path) -> None:
    # A recording sealed mid-segment resumes in a fresh encoder at the next
    # boundary. Its segments are numbered from there, its timeline starts there,
    # and the first part's init decodes them, so a player reading init.mp4 and
    # every segment in order plays both parts on one timeline.
    first = _encoder(tmp_path)
    for _ in range(20):
        first.feed_video(_frame())
    size = first.frame_size
    first.stop()
    second = ChunkEncoder(
        output_dir=tmp_path,
        config=RecordingConfig(enabled=True, chunk_seconds=1),
        has_audio=False,
        audio_sample_rate=48_000,
        frame_rate=_FRAME_RATE,
        first_segment=1,
        frame_size=size,
    )
    for _ in range(_FRAME_RATE):
        second.feed_video(_frame(32))
    second.stop()

    assert sorted(p.name for p in tmp_path.glob("chunk_*.m4s")) == [
        "chunk_00000.m4s",
        "chunk_00001.m4s",
    ]
    assert (tmp_path / "init_00001.mp4").read_bytes() == (tmp_path / "init.mp4").read_bytes()
    with _decode(tmp_path) as container:
        stream = container.streams.video[0]
        assert stream.time_base is not None
        times = [
            float(frame.pts * stream.time_base)
            for frame in container.decode(stream)
            if frame.pts is not None
        ]
    assert len(times) == 20 + _FRAME_RATE
    assert times[19] < 20 / _FRAME_RATE
    assert times[20] == pytest.approx(1.0)


# -- frame handling ---------------------------------------------------------


def _gray(width: int, height: int) -> np.ndarray[Any, Any]:
    return np.full((height, width, 3), 200, dtype=np.uint8)


def _last_frame(output_dir: Path) -> np.ndarray[Any, Any]:
    """Decode the recording and return its last frame as ``(H, W, 3)`` RGB."""
    with _decode(output_dir) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    return frames[-1]


def _lit(row: np.ndarray[Any, Any]) -> bool:
    """Whether a run of pixels is the gray picture rather than a black bar."""
    return bool(row.mean() > 150)


def test_a_resized_frame_with_the_same_aspect_ratio_fills_the_opened_size(
    tmp_path: Path,
) -> None:
    encoder = _encoder(tmp_path)
    encoder.feed_video(_gray(64, 64))
    for _ in range(5):
        encoder.feed_video(_gray(32, 32))
    encoder.stop()

    with _decode(tmp_path) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (64, 64)
    last = _last_frame(tmp_path)
    assert _lit(last[2])
    assert _lit(last[:, 2])


def test_a_wider_frame_is_letterboxed_with_bars_above_and_below(tmp_path: Path) -> None:
    # A 2:1 frame on a square canvas fills the width and half the height, with
    # 16-pixel bars above and below, instead of being squeezed to a square.
    encoder = _encoder(tmp_path)
    encoder.feed_video(_gray(64, 64))
    for _ in range(5):
        encoder.feed_video(_gray(128, 64))
    encoder.stop()

    last = _last_frame(tmp_path)
    assert last.shape == (64, 64, 3)
    assert not _lit(last[4])
    assert not _lit(last[59])
    assert _lit(last[32])
    assert _lit(last[32, :4])


def test_a_taller_frame_is_pillarboxed_with_bars_left_and_right(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    encoder.feed_video(_gray(64, 64))
    for _ in range(5):
        encoder.feed_video(_gray(32, 64))
    encoder.stop()

    last = _last_frame(tmp_path)
    assert not _lit(last[:, 4])
    assert not _lit(last[:, 59])
    assert _lit(last[:, 32])
    assert _lit(last[:4, 32])


def test_a_frame_of_another_size_is_letterboxed_onto_the_target_size(tmp_path: Path) -> None:
    encoder = ChunkEncoder(
        output_dir=tmp_path,
        config=RecordingConfig(enabled=True, target_width=64, target_height=32),
        has_audio=False,
        audio_sample_rate=48_000,
        frame_rate=_FRAME_RATE,
    )
    for _ in range(5):
        encoder.feed_video(_gray(64, 64))
    encoder.stop()

    last = _last_frame(tmp_path)
    assert last.shape == (32, 64, 3)
    assert not _lit(last[:, 4])
    assert not _lit(last[:, 59])
    assert _lit(last[:, 32])


def test_an_odd_first_frame_opens_at_the_even_size_below_it(tmp_path: Path) -> None:
    # 4:2:0 needs an even width and height: an odd first frame used to fail the
    # encoder open and turn recording off for the session.
    encoder = _encoder(tmp_path)
    for _ in range(5):
        encoder.feed_video(_gray(65, 49))
    assert encoder.frame_size == (64, 48)
    encoder.stop()

    assert not encoder.failed
    with _decode(tmp_path) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (64, 48)
        assert _frame_count(container) == 5


def test_the_configured_target_size_wins_over_the_first_frame(tmp_path: Path) -> None:
    encoder = ChunkEncoder(
        output_dir=tmp_path,
        config=RecordingConfig(enabled=True, target_width=48, target_height=32),
        has_audio=False,
        audio_sample_rate=48_000,
        frame_rate=_FRAME_RATE,
    )
    encoder.feed_video(_frame(64))
    encoder.stop()

    with _decode(tmp_path) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (48, 32)


def test_an_odd_target_size_is_rounded_down_to_even(tmp_path: Path) -> None:
    encoder = ChunkEncoder(
        output_dir=tmp_path,
        config=RecordingConfig(enabled=True, target_width=49, target_height=33),
        has_audio=False,
        audio_sample_rate=48_000,
        frame_rate=_FRAME_RATE,
    )
    encoder.feed_video(_frame(64))
    assert encoder.frame_size == (48, 32)
    encoder.stop()
    assert not encoder.failed


def test_feed_video_rejects_a_frame_that_is_not_three_channel(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    with pytest.raises(ValueError, match="expects"):
        encoder.feed_video(np.zeros((8, 8), dtype=np.uint8))


# -- lifecycle --------------------------------------------------------------


def test_stop_is_safe_before_a_frame_and_when_repeated(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    encoder.stop()
    encoder.stop()

    assert not encoder.failed
    assert not list(tmp_path.glob("chunk_*.m4s"))


def test_a_feed_after_stop_is_refused(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    encoder.feed_video(_frame())
    encoder.stop()

    # Encoding into a closed container would crash, so the latch has to hold even
    # against a feed worker that has not wound down yet. The refusal has its own
    # type so the recorder can tell teardown apart from an encoder failure.
    with pytest.raises(EncoderStoppedError):
        encoder.feed_video(_frame())
    assert not encoder.failed


def test_stop_gives_up_on_a_feed_stuck_inside_libav(tmp_path: Path) -> None:
    # A feed holds the encoder's lock across the libav call, so one that never
    # returns would block stop() on that lock forever. stop() gives up after its
    # timeout instead, and the stuck feed is refused once it does return.
    encoder = _encoder(tmp_path)
    encoder.feed_video(_frame())
    held = threading.Event()
    release = threading.Event()

    def stuck_feed() -> None:
        with encoder._lock:
            held.set()
            release.wait(10.0)

    feeder = threading.Thread(target=stuck_feed, daemon=True)
    feeder.start()
    assert held.wait(5.0)
    try:
        started = time.monotonic()
        closed = encoder.stop(timeout=0.2)
        elapsed = time.monotonic() - started
    finally:
        release.set()
        feeder.join(5.0)

    assert elapsed < 1.0
    assert closed is False  # the final segment was left unclosed, and stop() says so
    with pytest.raises(EncoderStoppedError):
        encoder.feed_video(_frame())


def test_stop_reports_that_it_closed_the_output(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    encoder.feed_video(_frame())
    assert encoder.stop() is True
    assert encoder.stop() is True  # already closed
    assert _encoder(tmp_path / "unopened").stop() is True  # never opened


def test_feed_audio_is_inert_for_a_video_only_recording(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    encoder.feed_video(_frame())
    encoder.feed_audio(np.zeros(1600, dtype=np.int16))
    encoder.stop()

    with _decode(tmp_path) as container:
        # The silent track written alongside the video, not the samples above.
        assert len(container.streams.audio) == 1


# -- encoder options --------------------------------------------------------


@pytest.mark.parametrize(
    ("codec", "key"),
    [("h264", "x264-params"), ("h265", "x265-params")],
)
def test_the_gop_is_pinned_to_the_chunk_length(codec: str, key: str) -> None:
    options = _video_options(RecordingConfig(enabled=True, video_codec=codec), 60)

    assert "keyint=60:min-keyint=60" in options[key]
    assert "scenecut=0" in options[key]


def test_a_segment_lands_within_a_few_frames_of_its_boundary(tmp_path: Path) -> None:
    # Each x264 frame thread holds a frame back. Capped, a 1080p segment is on
    # disk a handful of frames after its boundary, whatever the host's cores.
    encoder = _encoder(tmp_path, chunk_seconds=1)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    landed_at = -1
    try:
        for index in range(_FRAME_RATE * 2):
            encoder.feed_video(frame)
            if (tmp_path / "chunk_00000.m4s").exists():
                landed_at = index
                break
    finally:
        encoder.stop()

    assert _FRAME_RATE <= landed_at <= _FRAME_RATE + _X264_THREADS + 2
