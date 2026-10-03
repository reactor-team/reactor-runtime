from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from reactor_runtime.core import (
    MediaBundle,
    StepResultsConfig,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.step_results import write_mp4

_CONFIG = StepResultsConfig(enabled=True)


def _video(name: str, frames: np.ndarray) -> TrackData:
    info = TrackInfo(name=name, kind=TrackKind.VIDEO, rate=0.0, direction=TrackDirection.OUT)
    return TrackData(info=info, data=frames)


def _audio(name: str, samples: np.ndarray, rate: float = 48_000.0) -> TrackData:
    info = TrackInfo(name=name, kind=TrackKind.AUDIO, rate=rate, direction=TrackDirection.OUT)
    return TrackData(info=info, data=samples)


def _frames(count: int, height: int = 48, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(count, height, width, 3), dtype=np.uint8)


def _bundle(*tracks: TrackData) -> MediaBundle:
    return MediaBundle(tracks={track.info.name: track for track in tracks})


def _decoded(path: Path) -> list[dict[str, Any]]:
    """Decode every stream of the file, in file order, and describe what it held."""
    with av.open(str(path)) as container:
        assert len(container.streams) == len(container.streams.video) + len(container.streams.audio)
        described: list[dict[str, Any]] = []
        for video in container.streams.video:
            container.seek(0)
            pictures = list(container.decode(video))
            described.append(
                {
                    "type": "video",
                    "codec": video.codec_context.name,
                    "frames": len(pictures),
                    "size": (pictures[0].width, pictures[0].height),
                    "rate": video.average_rate,
                }
            )
        for audio in container.streams.audio:
            container.seek(0)
            blocks = list(container.decode(audio))
            described.append(
                {
                    "type": "audio",
                    "codec": audio.codec_context.name,
                    "samples": sum(block.samples for block in blocks),
                    "rate": audio.rate,
                }
            )
        return described


def test_a_video_and_an_audio_track_become_two_streams(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    samples = np.zeros((1, 24_000), dtype=np.int16)
    write_mp4(
        path, _bundle(_video("main_video", _frames(10)), _audio("main_audio", samples)), 24, _CONFIG
    )

    video, audio = _decoded(path)
    assert video["type"] == "video"
    assert video["codec"] == "h264"
    assert video["frames"] == 10
    assert video["size"] == (64, 48)
    assert video["rate"] == 24
    assert audio["type"] == "audio"
    assert audio["codec"] == "aac"
    assert audio["rate"] == 48_000
    # The encoder's priming and padding make the decoded length a little longer.
    assert 24_000 <= audio["samples"] <= 24_000 + 3 * 1024


def test_a_video_only_output_has_one_stream(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    write_mp4(path, _bundle(_video("main_video", _frames(3))), 30, _CONFIG)

    (video,) = _decoded(path)
    assert video["frames"] == 3


def test_each_video_track_is_its_own_stream_at_its_own_size(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    bundle = _bundle(
        _video("main_video", _frames(4, 48, 64)),
        _video("depth", _frames(2, 32, 32)),
    )
    write_mp4(path, bundle, 24, _CONFIG)

    main, depth = _decoded(path)
    assert (main["frames"], main["size"]) == (4, (64, 48))
    assert (depth["frames"], depth["size"]) == (2, (32, 32))


def test_a_single_frame_payload_is_one_frame(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    write_mp4(path, _bundle(_video("main_video", _frames(1)[0])), 24, _CONFIG)

    (video,) = _decoded(path)
    assert video["frames"] == 1


def test_an_odd_frame_size_is_padded_to_even(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    write_mp4(path, _bundle(_video("main_video", _frames(2, 21, 33))), 24, _CONFIG)

    (video,) = _decoded(path)
    assert video["size"] == (34, 22)


def test_a_fractional_frame_rate_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    write_mp4(path, _bundle(_video("main_video", _frames(3))), 30000 / 1001, _CONFIG)

    (video,) = _decoded(path)
    assert float(video["rate"]) == pytest.approx(29.97, abs=0.01)


def test_h265_is_used_when_configured(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    config = StepResultsConfig(enabled=True, video_codec="h265")
    write_mp4(path, _bundle(_video("main_video", _frames(2))), 24, config)

    (video,) = _decoded(path)
    assert video["codec"] == "hevc"


def test_the_file_is_laid_out_to_play_while_it_downloads(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    write_mp4(path, _bundle(_video("main_video", _frames(5))), 24, _CONFIG)

    data = path.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")


@pytest.mark.parametrize(
    ("bundle", "fps", "message"),
    [
        (MediaBundle(tracks={}), 24, "no tracks"),
        (_bundle(_video("main_video", _frames(2))), 0, "fps must be positive"),
        (
            _bundle(_video("main_video", np.zeros((4, 4), dtype=np.uint8))),
            24,
            "must be \\(H, W, 3\\)",
        ),
    ],
)
def test_a_payload_it_cannot_encode_is_rejected_and_leaves_no_file(
    tmp_path: Path, bundle: MediaBundle, fps: float, message: str
) -> None:
    path = tmp_path / "output.mp4"
    with pytest.raises(ValueError, match=message):
        write_mp4(path, bundle, fps, _CONFIG)
    assert not path.exists()


def test_a_file_that_fails_while_it_is_written_is_removed(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    config = StepResultsConfig(enabled=True, audio_codec="not-a-codec")
    bundle = _bundle(
        _video("main_video", _frames(2)), _audio("main_audio", np.zeros((1, 480), np.int16))
    )
    with pytest.raises(av.codec.codec.UnknownCodecError):
        write_mp4(path, bundle, 24, config)
    assert not path.exists()
