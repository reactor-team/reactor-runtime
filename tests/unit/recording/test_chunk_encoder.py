import subprocess
from pathlib import Path
from typing import cast

import pytest

from reactor_runtime.core import RecordingConfig
from reactor_runtime.recording.chunk_encoder import ChunkEncoder, _build_argv


class _ExitedProc:
    """A stand-in for a finished ffmpeg subprocess with a known exit code."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode


class _RecordingLogger:
    """A logger stub that records the warnings emitted against it."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, object]]] = []

    def warning(self, message: str, **fields: object) -> None:
        self.warnings.append((message, fields))


def _encoder(tmp_path: Path) -> ChunkEncoder:
    return ChunkEncoder(
        output_dir=tmp_path,
        config=RecordingConfig(enabled=True),
        has_audio=False,
        audio_sample_rate=48_000,
    )


def test_argv_stamps_pts_from_a_fixed_input_framerate(tmp_path: Path) -> None:
    argv = _build_argv(
        output_dir=tmp_path,
        width=64,
        height=48,
        has_audio=False,
        audio_sample_rate=48_000,
        frame_rate=24,
        config=RecordingConfig(enabled=True),
        video_read_fd=7,
        audio_read_fd=None,
    )
    # PTS derives from the declared input frame rate, not from wall-clock arrival.
    assert "-framerate" in argv
    assert argv[argv.index("-framerate") + 1] == "24"
    assert "-use_wallclock_as_timestamps" not in argv


def test_stop_logs_a_non_zero_ffmpeg_self_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr("reactor_runtime.recording.chunk_encoder.logger", logger)
    encoder = _encoder(tmp_path)
    encoder._proc = cast("subprocess.Popen[bytes]", _ExitedProc(1))

    encoder.stop()

    assert logger.warnings
    assert logger.warnings[0][1]["returncode"] == 1


def test_stop_is_quiet_on_a_clean_self_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr("reactor_runtime.recording.chunk_encoder.logger", logger)
    encoder = _encoder(tmp_path)
    encoder._proc = cast("subprocess.Popen[bytes]", _ExitedProc(0))

    encoder.stop()

    assert logger.warnings == []
