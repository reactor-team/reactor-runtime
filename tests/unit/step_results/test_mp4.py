from pathlib import Path
from typing import Any

import av
import av.stream
import numpy as np
import pytest

from reactor_runtime.core import MediaBundle, StepResultsConfig, TrackData, TrackInfo, TrackKind
from reactor_runtime.step_results.mp4 import StreamInfo, encode_mp4
from reactor_runtime.step_results.result import StepResultCancelledError
from reactor_runtime.step_results.validate import prepare_tracks

_CONFIG = StepResultsConfig(enabled=True)


def _video(name: str, frames: int, height: int = 32, width: int = 48) -> TrackData:
    data = np.random.default_rng(0).integers(0, 255, (frames, height, width, 3), dtype=np.uint8)
    return TrackData(info=TrackInfo(name=name, kind=TrackKind.VIDEO), data=data)


def _audio(name: str, samples: int, rate: int = 48_000) -> TrackData:
    data = np.full((1, samples), 1000, dtype=np.int16)
    return TrackData(info=TrackInfo(name=name, kind=TrackKind.AUDIO, rate=float(rate)), data=data)


def _bundle(*tracks: TrackData) -> MediaBundle:
    return MediaBundle(tracks={track.info.name: track for track in tracks})


def _encode(path: Path, fps: float, *tracks: TrackData, **kwargs: Any) -> list[StreamInfo]:
    return encode_mp4(path, prepare_tracks(_bundle(*tracks), fps), fps, _CONFIG, **kwargs)


def _decoded_frames(path: Path, index: int) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams[index]))


def test_every_track_becomes_a_stream_in_declaration_order(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"

    infos = _encode(
        path,
        24.0,
        _video("main_video", 6),
        _video("depth", 6, height=16, width=24),
        _audio("main_audio", 12_000),
        _audio("narration", 4_000, rate=16_000),
    )

    assert [(info.name, info.kind, info.stream) for info in infos] == [
        ("main_video", TrackKind.VIDEO, 0),
        ("depth", TrackKind.VIDEO, 1),
        ("main_audio", TrackKind.AUDIO, 2),
        ("narration", TrackKind.AUDIO, 3),
    ]
    with av.open(str(path)) as container:
        assert [stream.type for stream in container.streams] == ["video", "video", "audio", "audio"]
        video = container.streams.video
        audio = container.streams.audio
        assert (video[0].width, video[0].height) == (48, 32)
        assert (video[1].width, video[1].height) == (24, 16)
        assert audio[0].rate == 48_000
        assert audio[1].rate == 16_000


def test_the_first_stream_of_each_kind_is_the_default(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"

    infos = _encode(
        path,
        24.0,
        _video("main_video", 3),
        _video("depth", 3),
        _audio("main_audio", 6_000),
        _audio("narration", 6_000),
    )

    assert [info.default for info in infos] == [True, False, True, False]
    with av.open(str(path)) as container:
        assert [
            bool(stream.disposition & av.stream.Disposition.default) for stream in container.streams
        ] == [
            True,
            False,
            True,
            False,
        ]


def test_result_entries_describe_each_stream_by_kind(tmp_path: Path) -> None:
    infos = _encode(
        tmp_path / "output.mp4", 24.0, _video("main_video", 6), _audio("main_audio", 12_000)
    )

    assert infos[0].to_dict() == {
        "name": "main_video",
        "kind": "video",
        "file": "output.mp4",
        "stream": 0,
        "width": 48,
        "height": 32,
        "fps": 24.0,
        "frames": 6,
        "default": True,
    }
    assert infos[1].to_dict() == {
        "name": "main_audio",
        "kind": "audio",
        "file": "output.mp4",
        "stream": 1,
        "sample_rate": 48_000,
        "samples": 12_000,
        "default": True,
    }


def test_every_video_frame_is_in_the_file_at_the_asked_rate(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"

    _encode(path, 12.0, _video("main_video", 9), _video("depth", 9))

    assert _decoded_frames(path, 0) == 9
    assert _decoded_frames(path, 1) == 9
    with av.open(str(path)) as container:
        assert float(container.streams.video[0].average_rate or 0) == pytest.approx(12.0)


def test_a_fractional_rate_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"

    infos = _encode(path, 23.976, _video("main_video", 4))

    assert infos[0].fps == pytest.approx(23.976, rel=1e-4)
    with av.open(str(path)) as container:
        rate = container.streams.video[0].average_rate
        assert float(rate or 0) == pytest.approx(23.976, rel=1e-3)


@pytest.mark.parametrize("samples", [1_000, 48_000])
def test_audio_is_padded_or_trimmed_to_the_video(tmp_path: Path, samples: int) -> None:
    # 12 frames at 24 fps is half a second: 24 000 samples at 48 kHz, whether
    # the model handed over fewer (padded with silence) or more (trimmed).
    path = tmp_path / "output.mp4"

    infos = _encode(path, 24.0, _video("main_video", 12), _audio("main_audio", samples))

    assert infos[1].samples == 24_000
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        assert stream.duration is not None
        assert stream.time_base is not None
        # AAC adds a priming window; the duration lands within one frame of it.
        assert float(stream.duration * stream.time_base) == pytest.approx(0.5, abs=0.05)


def test_a_video_only_output_writes_no_audio_stream(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"

    infos = _encode(path, 24.0, _video("main_video", 3))

    assert [info.kind for info in infos] == [TrackKind.VIDEO]
    with av.open(str(path)) as container:
        assert [stream.type for stream in container.streams] == ["video"]


def test_an_audio_only_output_keeps_every_sample(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"

    infos = _encode(path, 24.0, _audio("main_audio", 48_000))

    assert infos[0].samples == 48_000
    with av.open(str(path)) as container:
        assert [stream.type for stream in container.streams] == ["audio"]


def test_mismatched_video_frame_counts_are_refused() -> None:
    with pytest.raises(ValueError, match="same number of frames"):
        prepare_tracks(_bundle(_video("main_video", 6), _video("depth", 5)), 24.0)


def test_a_single_frame_is_a_batch_of_one(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    single = TrackData(
        info=TrackInfo(name="main_video", kind=TrackKind.VIDEO),
        data=np.zeros((32, 48, 3), dtype=np.uint8),
    )

    infos = _encode(path, 24.0, single)

    assert infos[0].frames == 1
    assert _decoded_frames(path, 0) == 1


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (np.zeros((0, 32, 48, 3), dtype=np.uint8), "non-empty uint8 RGB"),
        (np.zeros((4, 32, 48, 4), dtype=np.uint8), "non-empty uint8 RGB"),
        (np.zeros((4, 32, 48, 3), dtype=np.float32), "non-empty uint8 RGB"),
        (np.zeros((32, 48), dtype=np.uint8), "non-empty uint8 RGB"),
    ],
)
def test_a_video_payload_the_encoder_cannot_take_is_refused(data: Any, match: str) -> None:
    track = TrackData(info=TrackInfo(name="main_video", kind=TrackKind.VIDEO), data=data)

    with pytest.raises(ValueError, match=match):
        prepare_tracks(_bundle(track), 24.0)


@pytest.mark.parametrize(
    ("data", "rate", "match"),
    [
        (np.zeros((2, 100), dtype=np.int16), 48_000.0, "mono int16"),
        (np.zeros((1, 100), dtype=np.float32), 48_000.0, "mono int16"),
        (np.zeros((1, 100), dtype=np.int16), 0.0, "integer sample rate"),
        (np.zeros((1, 100), dtype=np.int16), 44_100.5, "integer sample rate"),
    ],
)
def test_an_audio_payload_the_encoder_cannot_take_is_refused(
    data: Any, rate: float, match: str
) -> None:
    track = TrackData(info=TrackInfo(name="main_audio", kind=TrackKind.AUDIO, rate=rate), data=data)

    with pytest.raises(ValueError, match=match):
        prepare_tracks(_bundle(track), 24.0)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("inf"), float("nan")])
def test_a_bad_frame_rate_is_refused(fps: float) -> None:
    with pytest.raises(ValueError, match="fps must be finite and positive"):
        prepare_tracks(_bundle(_video("main_video", 2)), fps)


def test_nothing_to_encode_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one track"):
        encode_mp4(tmp_path / "output.mp4", [], 24.0, _CONFIG)


def test_a_cancel_mid_encode_stops_the_file(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    polls = iter([False, False, True])

    with pytest.raises(StepResultCancelledError):
        _encode(path, 24.0, _video("main_video", 10), cancelled=lambda: next(polls, True))
