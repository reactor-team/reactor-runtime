import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from reactor_runtime.core import RecordingConfig
from reactor_runtime.recording.chunk_encoder import ChunkEncoder, _video_options

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
    # The recorder serves a segment as soon as its successor exists, so segments
    # have to appear while the recording is still running.
    encoder = _encoder(tmp_path, chunk_seconds=1)
    try:
        for _ in range(_FRAME_RATE * 2):
            encoder.feed_video(_frame())
        assert list(tmp_path.glob("chunk_*.m4s"))
    finally:
        encoder.stop()


# -- frame handling ---------------------------------------------------------


def test_a_resized_frame_is_scaled_onto_the_opened_size(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    encoder.feed_video(_frame(64))
    encoder.feed_video(_frame(32))
    encoder.stop()

    with _decode(tmp_path) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (64, 64)
        assert _frame_count(container) == 2


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
    # against a feed worker that has not wound down yet.
    with pytest.raises(RuntimeError, match="stopped"):
        encoder.feed_video(_frame())


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
