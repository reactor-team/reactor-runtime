import json
import os
import subprocess
import sys
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
from reactor_runtime.step_results import VideoUsage, write_mp4
from reactor_runtime.step_results.mp4 import _video_options

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
    usage = write_mp4(
        path, _bundle(_video("main_video", _frames(10)), _audio("main_audio", samples)), 24, _CONFIG
    )

    # The audio stream is in the file but has no entry.
    assert usage == (
        VideoUsage(
            track="main_video",
            file="output.mp4",
            duration_s=10 / 24,
            frames=10,
            fps=24.0,
            width=64,
            height=48,
        ),
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


def test_a_long_video_and_audio_output_is_interleaved(tmp_path: Path) -> None:
    # Twenty seconds of each: longer than the muxer buffers for one stream
    # while it waits for the other.
    path = tmp_path / "output.mp4"
    frames = np.zeros((600, 48, 64, 3), dtype=np.uint8)
    samples = np.zeros((1, 48_000 * 20), dtype=np.int16)
    write_mp4(
        path, _bundle(_video("main_video", frames), _audio("main_audio", samples)), 30, _CONFIG
    )

    video, audio = _decoded(path)
    assert video["frames"] == 600
    assert 48_000 * 20 <= audio["samples"] <= 48_000 * 20 + 3 * 1024


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
    usage = write_mp4(path, bundle, 24, _CONFIG)

    main, depth = _decoded(path)
    assert (main["frames"], main["size"]) == (4, (64, 48))
    assert (depth["frames"], depth["size"]) == (2, (32, 32))
    assert [(entry.track, entry.frames, entry.width, entry.height) for entry in usage] == [
        ("main_video", 4, 64, 48),
        ("depth", 2, 32, 32),
    ]


def test_a_single_frame_payload_is_one_frame(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    (entry,) = write_mp4(path, _bundle(_video("main_video", _frames(1)[0])), 24, _CONFIG)

    (video,) = _decoded(path)
    assert video["frames"] == 1
    assert (entry.frames, entry.duration_s) == (1, 1 / 24)


def test_an_odd_frame_size_is_padded_to_even(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    (entry,) = write_mp4(path, _bundle(_video("main_video", _frames(2, 21, 33))), 24, _CONFIG)

    (video,) = _decoded(path)
    assert video["size"] == (34, 22)
    assert (entry.width, entry.height) == (34, 22)


def test_a_fractional_frame_rate_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    (entry,) = write_mp4(path, _bundle(_video("main_video", _frames(3))), 30000 / 1001, _CONFIG)

    (video,) = _decoded(path)
    assert float(video["rate"]) == pytest.approx(29.97, abs=0.01)
    # The entry reports the rate the stream holds, not the float it was given.
    assert entry.fps == float(video["rate"])
    assert entry.duration_s == pytest.approx(3 * 1001 / 30000)


def test_h265_is_used_when_configured(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    config = StepResultsConfig(enabled=True, video_codec="h265")
    write_mp4(path, _bundle(_video("main_video", _frames(2))), 24, config)

    (video,) = _decoded(path)
    assert video["codec"] == "hevc"


# Encodes one- and two-frame H.265 tracks the way write_mp4 does and prints
# each packet's (pts, dts).
_SHORT_H265_TRACKS = """
import json, sys
from fractions import Fraction
import av, numpy as np
from reactor_runtime.core import StepResultsConfig
from reactor_runtime.step_results import mp4

config = StepResultsConfig(enabled=True, video_codec="h265")
stamps = {}
for count in (1, 2):
    frames = np.zeros((count, 48, 64, 3), dtype=np.uint8)
    container = av.open(f"{sys.argv[1]}/{count}.mp4", mode="w", format="mp4")
    stream = mp4._add_video(container, frames, Fraction(24), config)
    stamps[count] = [(p.pts, p.dts) for p in mp4._encode_video(stream, frames)]
    container.close()
print(json.dumps(stamps))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="MALLOC_PERTURB_ is a glibc setting")
def test_a_short_h265_track_decodes_each_packet_at_its_presentation_time(
    tmp_path: Path,
) -> None:
    """A one- or two-frame H.265 track gets decode times that do not depend on the heap.

    With B-frames, x265 releases up to 4.3 read the decode time of a track
    shorter than three frames from memory they never set. glibc's
    ``MALLOC_PERTURB_`` fills fresh memory with junk, so a value read that way
    is wrong on every run rather than only when the memory was used before.
    """
    run = subprocess.run(
        [sys.executable, "-c", _SHORT_H265_TRACKS, str(tmp_path)],
        env={**os.environ, "MALLOC_PERTURB_": "165"},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    stamps = json.loads(run.stdout.strip().splitlines()[-1])

    assert stamps == {"1": [[0, 0]], "2": [[0, 0], [1, 1]]}


def test_an_h265_track_of_three_frames_keeps_its_b_frames(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    config = StepResultsConfig(enabled=True, video_codec="h265")
    write_mp4(path, _bundle(_video("main_video", _frames(3))), 24, config)

    with av.open(str(path)) as container:
        stamps = [(p.pts, p.dts) for p in container.demux(video=0) if p.size]
    pts, dts = stamps[0]
    assert pts is not None
    assert dts is not None
    # A reordered stream decodes its first packet before it is presented.
    assert dts < pts


@pytest.mark.parametrize(("frames", "bframes_off"), [(1, True), (2, True), (3, False), (24, False)])
def test_only_an_h265_track_shorter_than_three_frames_turns_b_frames_off(
    frames: int, bframes_off: bool
) -> None:
    h265 = _video_options(StepResultsConfig(enabled=True, video_codec="h265"), frames)
    h264 = _video_options(StepResultsConfig(enabled=True, video_codec="h264"), frames)

    assert ("bframes=0" in h265["x265-params"].split(":")) is bframes_off
    assert "x264-params" not in h264


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
