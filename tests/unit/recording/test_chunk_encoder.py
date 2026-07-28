import fcntl
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from reactor_runtime.core import RecordingConfig
from reactor_runtime.recording import chunk_encoder
from reactor_runtime.recording.chunk_encoder import (
    _F_GETPIPE_SZ,
    _VIDEO_QUEUE_DEPTH,
    ChunkEncoder,
    EncoderBusyError,
    _build_argv,
    _enlarge_pipe,
    _pipe_size_ceiling,
)

_DEFAULT_PIPE_BYTES = 65536


class _ExitedProc:
    """A stand-in for a finished ffmpeg subprocess with a known exit code."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode


class _ParkedFfmpeg:
    """A stand-in for ffmpeg that holds its input pipes open but never reads them.

    Duplicating the read ends is what the real child process does implicitly, and
    it is what keeps a write to the pipe blocking rather than failing once the
    encoder closes its own copy. Reading nothing reproduces a demuxer parked on
    one input, which is the state the encoder has to survive.
    """

    instance: "_ParkedFfmpeg | None" = None

    def __init__(self, argv: list[str], pass_fds: tuple[int, ...], **_: object) -> None:
        self.pid = -1
        self.returncode: int | None = None
        self.read_fds = [os.dup(fd) for fd in pass_fds]
        _ParkedFfmpeg.instance = self

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, _signal: int) -> None:
        self._exit()

    def kill(self) -> None:
        self._exit()

    def wait(self, timeout: float | None = None) -> int:
        self._exit()
        return 0

    def _exit(self) -> None:
        """Release the read ends, so a parked write to the pipe now fails."""
        if self.returncode is not None:
            return
        self.returncode = 0
        for fd in self.read_fds:
            os.close(fd)


class _RecordingLogger:
    """A logger stub that records the warnings emitted against it."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, object]]] = []

    def info(self, message: str, **fields: object) -> None:
        pass

    def warning(self, message: str, **fields: object) -> None:
        self.warnings.append((message, fields))


def _encoder(tmp_path: Path, *, has_audio: bool = False) -> ChunkEncoder:
    return ChunkEncoder(
        output_dir=tmp_path,
        config=RecordingConfig(enabled=True),
        has_audio=has_audio,
        audio_sample_rate=48_000,
    )


def _park_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spawn a parked stand-in instead of the real ffmpeg."""
    _ParkedFfmpeg.instance = None
    monkeypatch.setattr(subprocess, "Popen", _ParkedFfmpeg)


def _read_exactly(fd: int, count: int, timeout: float) -> bytes:
    """Read *count* bytes from *fd*, or as many as arrive before *timeout*."""
    deadline = time.monotonic() + timeout
    out = bytearray()
    while len(out) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            break
        out += os.read(fd, count - len(out))
    return bytes(out)


def _oversized_frame() -> np.ndarray[Any, Any]:
    """A frame too large for any pipe buffer the kernel will hand out."""
    return np.zeros((1024, 1024, 3), dtype=np.uint8)


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


# -- pipe independence -----------------------------------------------------


def test_a_stalled_video_pipe_still_lets_audio_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The deadlock this guards against: a frame larger than the pipe buffer only
    # streams through as ffmpeg reads it, so a video write against a parked
    # demuxer never returns. Audio has to reach ffmpeg anyway, because ffmpeg
    # holds the video packets back until the audio stream advances past them.
    _park_ffmpeg(monkeypatch)
    encoder = _encoder(tmp_path, has_audio=True)
    samples = np.arange(800, dtype=np.int16)
    try:
        started = time.monotonic()
        for _ in range(_VIDEO_QUEUE_DEPTH):
            encoder.feed_video(_oversized_frame())
            encoder.feed_audio(samples)
        assert time.monotonic() - started < 5.0

        proc = _ParkedFfmpeg.instance
        assert proc is not None
        audio_fd = proc.read_fds[1]
        expected = samples.tobytes() * _VIDEO_QUEUE_DEPTH
        assert _read_exactly(audio_fd, len(expected), timeout=5.0) == expected
    finally:
        # The video pipe never drains here, so the graceful flush can only run
        # out its budget; a short one keeps teardown from padding the suite.
        encoder.stop(timeout=0.5)


def test_a_saturated_queue_drops_a_frame_without_failing_the_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A queue that stays full means the encoder is behind, which a busy host can
    # cause on its own. Recording has to survive it, so the feed reports the lost
    # frame and leaves the encoder usable rather than ending the recording.
    _park_ffmpeg(monkeypatch)
    monkeypatch.setattr(chunk_encoder, "_FEED_TIMEOUT_SECONDS", 0.1)
    encoder = _encoder(tmp_path, has_audio=True)
    try:
        busy = False
        for _ in range(_VIDEO_QUEUE_DEPTH + 4):
            try:
                encoder.feed_video(_oversized_frame())
            except EncoderBusyError:
                busy = True
                break

        assert busy
        assert not encoder.failed
    finally:
        encoder.stop(timeout=0.5)


def test_stop_returns_while_a_video_write_is_parked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _park_ffmpeg(monkeypatch)
    encoder = _encoder(tmp_path, has_audio=True)
    for _ in range(_VIDEO_QUEUE_DEPTH):
        encoder.feed_video(_oversized_frame())
    writer = encoder._video
    assert writer is not None

    started = time.monotonic()
    encoder.stop()

    # Killing ffmpeg is what frees the parked write, so teardown stays bounded
    # and the writer releases the pipe instead of leaking the thread and its fd.
    assert time.monotonic() - started < 10.0
    assert not writer._thread.is_alive()


def test_stop_flushes_queued_bytes_into_the_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _park_ffmpeg(monkeypatch)
    encoder = _encoder(tmp_path, has_audio=False)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    encoder.feed_video(frame)
    proc = _ParkedFfmpeg.instance
    assert proc is not None
    video_fd = os.dup(proc.read_fds[0])

    encoder.stop()

    try:
        assert _read_exactly(video_fd, frame.nbytes, timeout=5.0) == frame.tobytes()
    finally:
        os.close(video_fd)


def test_a_broken_pipe_fails_the_encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _park_ffmpeg(monkeypatch)
    encoder = _encoder(tmp_path, has_audio=False)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    encoder.feed_video(frame)
    proc = _ParkedFfmpeg.instance
    assert proc is not None
    # ffmpeg dying mid-recording leaves the writer with nowhere to write.
    proc._exit()

    # The write happens on the writer thread, so the failure surfaces on a later
    # feed rather than on the one that queued the doomed payload.
    deadline = time.monotonic() + 5.0
    raised = False
    while time.monotonic() < deadline and not raised:
        try:
            encoder.feed_video(frame)
        except RuntimeError:
            raised = True
        time.sleep(0.02)
    assert raised
    assert encoder.failed
    encoder.stop()


# -- pipe sizing -----------------------------------------------------------


@pytest.mark.skipif(sys.platform != "linux", reason="pipe sizing is Linux-only")
def test_enlarge_pipe_grows_to_the_node_ceiling() -> None:
    ceiling = _pipe_size_ceiling()
    if ceiling is None or ceiling <= _DEFAULT_PIPE_BYTES:
        pytest.skip("the node's pipe-max-size leaves no room to grow")
    read_fd, write_fd = os.pipe()
    try:
        # A request the node forbids is clamped rather than refused: asking for
        # more than fs.pipe-max-size answers EPERM without CAP_SYS_RESOURCE, which
        # would leave the pipe at its default size.
        capacity = _enlarge_pipe(write_fd, ceiling * 4)
        assert capacity > _DEFAULT_PIPE_BYTES
        assert capacity <= ceiling
        assert int(fcntl.fcntl(write_fd, _F_GETPIPE_SZ)) == capacity
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_enlarge_pipe_is_inert_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    # `_enlarge_pipe` reads sys.platform when it is called, so patching it covers
    # the branch a Linux runner never takes, and pins the 0 return `_spawn` logs.
    on_linux = sys.platform == "linux"
    read_fd, write_fd = os.pipe()
    try:
        before = int(fcntl.fcntl(write_fd, _F_GETPIPE_SZ)) if on_linux else 0
        monkeypatch.setattr(sys, "platform", "darwin")

        assert _enlarge_pipe(write_fd, 1 << 24) == 0

        if on_linux:
            assert int(fcntl.fcntl(write_fd, _F_GETPIPE_SZ)) == before
    finally:
        os.close(read_fd)
        os.close(write_fd)


# -- teardown --------------------------------------------------------------


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
