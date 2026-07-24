"""Continuous fMP4 chunk encoder backed by a single ``ffmpeg`` subprocess.

The encoder is lazy: ffmpeg spawns on the first :meth:`ChunkEncoder.feed_video`
call so the frame's shape derives the ``-s WxH`` argument. Output is HLS-fMP4
(``init.mp4`` + ``chunk_NNNNN.m4s``) written straight into the served directory;
:func:`_build_argv` documents the full flag rationale. The audio pipe, second
input, and ``-map 1:a`` are omitted when ``has_audio`` is ``False`` — a silent
synthetic track stands in so every segment still carries audio.

Each pipe is drained by its own :class:`_PipeWriter` thread, so the video and
audio for a recorded frame reach ffmpeg concurrently whatever the pipe buffers
hold.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from reactor_runtime.core import RecordingConfig
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

# Linux fcntl commands to read and resize a pipe's kernel buffer (absent from the
# fcntl module), the floor a resized pipe is grown to, and the node-wide ceiling
# the kernel refuses to exceed without ``CAP_SYS_RESOURCE``.
_F_SETPIPE_SZ = 1031
_F_GETPIPE_SZ = 1032
_MIN_PIPE_BYTES = 1 << 20
_PIPE_MAX_SIZE_PATH = Path("/proc/sys/fs/pipe-max-size")

# How many payloads each writer queues before a feed call blocks. A video frame
# is megabytes, so its queue stays shallow; one audio slot is a few kilobytes.
_VIDEO_QUEUE_DEPTH = 4
_AUDIO_QUEUE_DEPTH = 64
# How long a feed waits for queue room before it reports the pipe unusable, how
# long a writer is given to flush and release its pipe, and how often a writer
# wakes to notice a close request.
_FEED_TIMEOUT_SECONDS = 10.0
_CLOSE_TIMEOUT_SECONDS = 1.0
_WRITER_POLL_SECONDS = 0.05


class _PipeWriter:
    """Drains one of ffmpeg's input pipes from a dedicated thread.

    ffmpeg interleaves its piped inputs by DTS: an encoded video packet is held
    back until the audio stream has advanced past it. Writing both pipes from one
    thread couples them, because a frame larger than the pipe buffer only streams
    through as ffmpeg reads it and the audio ffmpeg is waiting for cannot be
    written until that write returns. A writer per pipe removes the coupling —
    every stream keeps moving as soon as ffmpeg reads it, at any frame size and
    any pipe buffer size.

    The writer owns *write_fd* for its whole life and closes it on exit, which is
    the EOF ffmpeg needs to write its trailer.
    """

    def __init__(self, name: str, write_fd: int, depth: int) -> None:
        """Take ownership of *write_fd* and start the writer thread."""
        self._name = name
        self._fd = write_fd
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=depth)
        self._closing = threading.Event()
        self._failed = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"recording-{name}-writer", daemon=True
        )
        self._thread.start()

    @property
    def failed(self) -> bool:
        """Whether a write to the pipe failed, which is terminal for the pipe."""
        return self._failed.is_set()

    def write(self, payload: bytes) -> bool:
        """Hand *payload* to the writer thread.

        Blocks while the queue is full, which is how encoder back-pressure
        reaches the caller.

        Args:
            payload: The bytes to write to the pipe, in order.

        Returns:
            Whether the payload was accepted. ``False`` once the pipe is closing,
            its writes have failed, or the queue stayed full for the whole feed
            timeout.
        """
        if self._closing.is_set() or self._failed.is_set():
            return False
        try:
            self._queue.put(payload, timeout=_FEED_TIMEOUT_SECONDS)
        except queue.Full:
            return False
        return not self._failed.is_set()

    def close(self, timeout: float) -> None:
        """Flush what is queued, close the pipe, and join the writer thread.

        Returns once the thread has exited or *timeout* elapses. A writer parked
        in a write to a pipe ffmpeg stopped reading only leaves that write when it
        fails, which the caller forces by killing ffmpeg; calling ``close`` again
        afterwards collects it. Safe to call repeatedly.
        """
        self._closing.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        """Write queued payloads to the pipe until closed, then release the fd."""
        try:
            while True:
                try:
                    payload = self._queue.get(timeout=_WRITER_POLL_SECONDS)
                except queue.Empty:
                    if self._closing.is_set():
                        return
                    continue
                # A failed pipe keeps draining its queue, so a feed call never
                # blocks against a writer with nowhere left to write.
                if self._failed.is_set():
                    continue
                try:
                    self._write_all(payload)
                except OSError as exc:
                    self._failed.set()
                    logger.warning("recorder pipe write failed", pipe=self._name, error=str(exc))
        finally:
            with contextlib.suppress(OSError):
                os.close(self._fd)

    def _write_all(self, payload: bytes) -> None:
        """Write every byte of *payload*, however many writes the pipe takes."""
        view = memoryview(payload)
        while view:
            view = view[os.write(self._fd, view) :]


def _pipe_size_ceiling() -> int | None:
    """Return the node's ``fs.pipe-max-size``, or ``None`` when unreadable."""
    try:
        return int(_PIPE_MAX_SIZE_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def _enlarge_pipe(write_fd: int, want_bytes: int) -> int:
    """Grow a feed pipe's buffer towards *want_bytes* and report its capacity.

    A roomier buffer lets ffmpeg absorb a burst without the writer thread having
    to stream the bytes through in step with ffmpeg's reads. The request is
    clamped to ``fs.pipe-max-size``, because the kernel answers ``EPERM`` rather
    than clamping for a process without ``CAP_SYS_RESOURCE`` — asking for more
    than the node allows would leave the pipe at its default size.

    Args:
        write_fd: The pipe's write end.
        want_bytes: The buffer size the caller would like.

    Returns:
        The pipe's capacity in bytes, or ``0`` where the platform does not expose
        pipe sizing.
    """
    if sys.platform != "linux":
        return 0
    desired = max(_MIN_PIPE_BYTES, want_bytes)
    ceiling = _pipe_size_ceiling()
    if ceiling is not None:
        desired = min(desired, ceiling)
    with contextlib.suppress(OSError):
        fcntl.fcntl(write_fd, _F_SETPIPE_SZ, desired)
    try:
        return int(fcntl.fcntl(write_fd, _F_GETPIPE_SZ))
    except OSError:
        return 0


def _resize_nearest(frame: npt.NDArray[Any], target_w: int, target_h: int) -> npt.NDArray[Any]:
    """Nearest-neighbour resize using only numpy; a no-op when already sized."""
    h, w = frame.shape[:2]
    if h == target_h and w == target_w:
        return frame
    y_idx = (np.arange(target_h) * (h / target_h)).astype(np.int64)
    x_idx = (np.arange(target_w) * (w / target_w)).astype(np.int64)
    resized: npt.NDArray[Any] = frame[y_idx[:, None], x_idx[None, :]]
    return resized


def _build_argv(
    output_dir: Path,
    width: int,
    height: int,
    has_audio: bool,
    audio_sample_rate: int,
    frame_rate: int,
    config: RecordingConfig,
    video_read_fd: int,
    audio_read_fd: int | None,
) -> list[str]:
    """Construct the ffmpeg argv.

    Video input declares a fixed ``-framerate`` (*frame_rate*): ffmpeg assigns
    each frame a PTS of ``index / frame_rate``, so the recorded timeline is
    media time derived from the model's declared cadence, not the wall-clock
    moment a frame was read. The recorder resamples each chunk's frames onto that
    grid before feeding, so a model running faster or slower than real time still
    records at true duration. Audio derives its DTS from byte count and sample
    rate; it is fed ``sample_rate / frame_rate`` samples per recorded frame, so
    its DTS tracks the video PTS one-to-one.

    Compatibility choices:

    * ``-pix_fmt yuv420p`` and ``-profile:v main`` on the output. With rgb24
      input and no output pixel format, libx264 selects yuv444p (Hi444PP), which
      many decoders and uploaders reject; yuv420p with the Main profile is the
      universally-compatible sub-profile.
    * A synthetic silent AAC track for video-only models via
      ``-f lavfi -i anullsrc``, terminated with ``-shortest`` on the video pipe's
      EOF, so every output carries an audio track without a second subprocess or
      audio pipe.
    """
    # ``-probesize 32 -analyzeduration 0`` on every input: rawvideo and s16le are
    # fully described by their format flags, so the default multi-megabyte probe
    # is pure cost and can deadlock the encoder when one pipe fills before the
    # other has enough bytes for the probe to complete.
    argv: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(frame_rate),
        # A deep packet queue lets a demuxer keep reading its pipe across the
        # jitter of encoder startup and of a busy host, so the pipe rarely fills.
        "-thread_queue_size",
        "1024",
        "-i",
        f"pipe:{video_read_fd}",
    ]
    if has_audio:
        assert audio_read_fd is not None
        argv += [
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "s16le",
            "-ar",
            str(audio_sample_rate),
            "-ac",
            "1",
            "-thread_queue_size",
            "1024",
            "-i",
            f"pipe:{audio_read_fd}",
        ]
    else:
        # ``lavfi anullsrc`` is a virtual generator inside libavfilter — no extra
        # subprocess, no audio pipe. ``-shortest`` below terminates the otherwise
        # infinite stream on the video EOF; the fixed-framerate video owns the
        # timeline, so the silent track needs no real-time pacing.
        argv += [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=cl=mono:r=48000",
        ]
    argv += ["-map", "0:v", "-map", "1:a"]
    vcodec = "libx264" if config.video_codec == "h264" else "libx265"
    argv += [
        "-c:v",
        vcodec,
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "main",
        "-preset",
        config.video_preset,
        "-crf",
        str(config.video_crf),
        "-tune",
        "zerolatency",
        # x264 sliced threads (auto-enabled by ``-tune zerolatency``) can deadlock
        # at higher resolutions; frame threads avoid it and the recorder does not
        # need per-frame latency.
        "-x264-params",
        "sliced-threads=0",
        "-force_key_frames",
        f"expr:gte(t,n_forced*{config.chunk_seconds})",
    ]
    argv += ["-c:a", config.audio_codec, "-b:a", f"{config.audio_bitrate_kbps}k"]
    if not has_audio:
        argv += ["-shortest"]
    # The HLS fMP4 muxer numbers segments from zero, which the marker math and the
    # manifest endpoint already assume. The generated ``.m3u8`` is ignored — the
    # ``/clips`` endpoint composes its own manifest.
    argv += [
        "-f",
        "hls",
        "-hls_segment_type",
        "fmp4",
        "-hls_time",
        str(config.chunk_seconds),
        "-hls_list_size",
        "0",
        "-hls_flags",
        "independent_segments",
        "-hls_fmp4_init_filename",
        "init.mp4",
        "-hls_segment_filename",
        str(output_dir / "chunk_%05d.m4s"),
        str(output_dir / "manifest.m3u8"),
    ]
    return argv


class ChunkEncoder:
    """Drives a long-lived ``ffmpeg`` subprocess for one recording.

    Lifecycle: construction records the output directory; the first
    :meth:`feed_video` spawns ffmpeg from the frame's shape and starts a writer
    per pipe; subsequent ``feed_video`` / ``feed_audio`` calls queue raw bytes for
    those writers; and :meth:`stop` flushes and closes the pipes, sends
    ``SIGTERM``, and escalates to ``SIGKILL`` if ffmpeg refuses to exit.
    """

    def __init__(
        self,
        output_dir: Path,
        config: RecordingConfig,
        has_audio: bool,
        audio_sample_rate: int,
        frame_rate: int = 30,
    ) -> None:
        """Record the output directory and encode settings; spawn ffmpeg lazily."""
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._has_audio = has_audio
        self._audio_sample_rate = audio_sample_rate
        self._frame_rate = frame_rate

        self._proc: subprocess.Popen[bytes] | None = None
        self._video: _PipeWriter | None = None
        self._audio: _PipeWriter | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._lock = threading.Lock()
        self._failed = False
        # A one-way latch flipped by ``stop()`` so the feed worker cannot
        # lazy-respawn ffmpeg after teardown began.
        self._stopped = False

    @property
    def output_dir(self) -> Path:
        """The directory ffmpeg writes init and chunk segments into."""
        return self._output_dir

    @property
    def failed(self) -> bool:
        """Return whether the encoder is dead and unrecoverable for this session."""
        if self._failed:
            return True
        if any(writer is not None and writer.failed for writer in (self._video, self._audio)):
            return True
        return self._proc is not None and self._proc.poll() is not None

    def feed_video(self, frame: npt.NDArray[Any]) -> None:
        """Push a single ``(H, W, 3)`` ``uint8`` frame into the encoder.

        Frames at a size other than the encoder is locked to are nearest-neighbour
        resized, so a changing resolution never breaks the rawvideo pipe.

        Raises:
            ValueError: If *frame* is not a three-channel image.
            RuntimeError: If the encoder is stopped or in a failed state.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"feed_video expects (H, W, 3); got shape {frame.shape}")
        if self._proc is None:
            with self._lock:
                if self._stopped:
                    raise RuntimeError("ChunkEncoder is stopped")
                if self._proc is None:
                    tw = self._config.target_width
                    th = self._config.target_height
                    if tw and th:
                        self._spawn(width=int(tw), height=int(th))
                    else:
                        h, w, _ = frame.shape
                        self._spawn(width=int(w), height=int(h))
        if self._failed:
            raise RuntimeError("ChunkEncoder is in a failed state")
        if (
            self._width is not None
            and self._height is not None
            and (frame.shape[1] != self._width or frame.shape[0] != self._height)
        ):
            frame = _resize_nearest(frame, self._width, self._height)
        writer = self._video
        if writer is None or not writer.write(np.ascontiguousarray(frame).tobytes()):
            self._failed = True
            raise RuntimeError("ffmpeg stopped accepting video")

    def feed_audio(self, samples: npt.NDArray[Any]) -> None:
        """Push ``int16`` PCM samples into the audio pipe.

        A no-op when the encoder has no audio pipe.

        Raises:
            RuntimeError: If the encoder is in a failed state.
        """
        writer = self._audio
        if not self._has_audio or writer is None:
            return
        if self._failed:
            raise RuntimeError("ChunkEncoder is in a failed state")
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        if not writer.write(np.ascontiguousarray(samples).tobytes()):
            self._failed = True
            raise RuntimeError("ffmpeg stopped accepting audio")

    def _spawn(self, width: int, height: int) -> None:
        """Open the pipes and launch ffmpeg for the given frame dimensions."""
        video_r, video_w = os.pipe()
        audio_r: int | None
        audio_w: int | None
        if self._has_audio:
            audio_r, audio_w = os.pipe()
        else:
            audio_r, audio_w = None, None

        frame_bytes = width * height * 3
        video_pipe_bytes = _enlarge_pipe(video_w, frame_bytes)
        if audio_w is not None:
            _enlarge_pipe(audio_w, _MIN_PIPE_BYTES)

        argv = _build_argv(
            output_dir=self._output_dir,
            width=width,
            height=height,
            has_audio=self._has_audio,
            audio_sample_rate=self._audio_sample_rate,
            frame_rate=self._frame_rate,
            config=self._config,
            video_read_fd=video_r,
            audio_read_fd=audio_r,
        )
        pass_fds: tuple[int, ...]
        if self._has_audio:
            assert audio_r is not None
            pass_fds = (video_r, audio_r)
        else:
            pass_fds = (video_r,)
        try:
            self._proc = subprocess.Popen(argv, pass_fds=pass_fds, stdin=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            for fd in (video_r, video_w, audio_r, audio_w):
                if fd is not None:
                    with contextlib.suppress(OSError):
                        os.close(fd)
            self._failed = True
            raise RuntimeError(
                "ffmpeg binary not found on PATH; install ffmpeg to enable recording"
            ) from exc
        # The parent never reads; the child owns the read ends.
        os.close(video_r)
        if audio_r is not None:
            os.close(audio_r)

        self._video = _PipeWriter("video", video_w, _VIDEO_QUEUE_DEPTH)
        if audio_w is not None:
            self._audio = _PipeWriter("audio", audio_w, _AUDIO_QUEUE_DEPTH)
        self._width = width
        self._height = height
        logger.info(
            "recorder encoder spawned ffmpeg",
            pid=self._proc.pid,
            width=width,
            height=height,
            audio=self._has_audio,
            video_pipe_bytes=video_pipe_bytes,
            frame_bytes=frame_bytes,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Close the input pipes and shut ffmpeg down, escalating to a kill.

        Always safe to call; a no-op when ffmpeg never spawned.
        """
        with self._lock:
            # Latch first so a concurrent ``feed_video`` bails out instead of
            # spawning over the pipes about to be closed.
            self._stopped = True
            writers = [writer for writer in (self._video, self._audio) if writer is not None]
            self._video = None
            self._audio = None
            proc = self._proc
            self._proc = None
        # Flushing the queues and closing the pipes is the EOF ffmpeg needs to
        # write its trailer, so it comes before the signal.
        for writer in writers:
            writer.close(timeout=_CLOSE_TIMEOUT_SECONDS)
        if proc is not None:
            self._shutdown(proc, timeout)
        # ffmpeg is gone, so a writer still parked in a write to its pipe now
        # fails and releases the fd.
        for writer in writers:
            writer.close(timeout=_CLOSE_TIMEOUT_SECONDS)

    @staticmethod
    def _shutdown(proc: subprocess.Popen[bytes], timeout: float) -> None:
        """Signal ffmpeg to finish, escalating to a kill if it stays alive."""
        if proc.poll() is not None:
            # ffmpeg exited on its own before teardown — the feed worker would
            # have seen a broken pipe and disabled recording. A non-zero code is
            # a silent encode failure, so surface it rather than let it vanish.
            if proc.returncode:
                logger.warning(
                    "recorder ffmpeg exited before teardown with a non-zero code",
                    returncode=proc.returncode,
                )
            return
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
