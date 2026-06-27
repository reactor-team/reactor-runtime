"""Continuous fMP4 chunk encoder backed by a single ``ffmpeg`` subprocess.

The encoder is lazy: ffmpeg spawns on the first :meth:`ChunkEncoder.feed_video`
call so the frame's shape derives the ``-s WxH`` argument. Output is HLS-fMP4
(``init.mp4`` + ``chunk_NNNNN.m4s``) written straight into the served directory;
:func:`_build_argv` documents the full flag rationale. The audio pipe, second
input, and ``-map 1:a`` are omitted when ``has_audio`` is ``False`` — a silent
synthetic track stands in so every segment still carries audio.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from reactor_runtime.core import RecordingConfig
from reactor_runtime.log import get_logger

logger = get_logger(__name__)


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
    config: RecordingConfig,
    video_read_fd: int,
    audio_read_fd: int | None,
) -> list[str]:
    """Construct the ffmpeg argv.

    Video input uses ``-use_wallclock_as_timestamps 1``: ffmpeg stamps each frame
    at the moment it reads it off the pipe, so the recorded PTS reflects the
    model's actual emission cadence (dynamic on most models). The feed worker
    therefore does not pace itself; it writes frames straight through and ffmpeg
    owns the timeline. Audio derives its DTS from byte count and sample rate (no
    wallclock flag) because wallclock-stamping a sample-rate-paced batch produces
    non-monotonic per-sample DTS in AAC and stalls the encoder; audio is fed at
    ``sample_rate * dt`` samples per video frame so its DTS still tracks
    wall-clock one-to-one with the video PTS.

    Compatibility choices:

    * ``-pix_fmt yuv420p`` and ``-profile:v main`` on the output. With rgb24
      input and no output pixel format, libx264 selects yuv444p (Hi444PP), which
      many decoders and uploaders reject; yuv420p with the Main profile is the
      universally-compatible sub-profile.
    * ``-vf setpts=PTS-STARTPTS`` re-anchors the wallclock-stamped video PTS to
      session-relative, so segment timestamps start near zero rather than at a
      Unix-epoch value some players treat as the playback start time.
    * A synthetic silent AAC track for video-only models via
      ``-f lavfi -i anullsrc``, paced with ``-re`` and terminated with
      ``-shortest`` on the video pipe's EOF, so every output carries an audio
      track without a second subprocess or audio pipe.
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
        "-use_wallclock_as_timestamps",
        "1",
        # A large packet queue keeps the demuxer from parking when audio writes
        # outpace video at higher resolutions, which would otherwise wedge the
        # worker's blocking write to the pipe.
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
        # subprocess, no audio pipe. ``-re`` paces it at native rate so the silent
        # stream tracks wall-clock the way the real pipe-fed path does; ``-shortest``
        # below terminates the otherwise-infinite stream on the video EOF.
        argv += [
            "-re",
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
        "-vf",
        "setpts=PTS-STARTPTS",
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
    :meth:`feed_video` spawns ffmpeg from the frame's shape; subsequent
    ``feed_video`` / ``feed_audio`` calls write raw bytes into the pipes; and
    :meth:`stop` closes the write ends, sends ``SIGTERM``, and escalates to
    ``SIGKILL`` if ffmpeg refuses to exit.
    """

    def __init__(
        self,
        output_dir: Path,
        config: RecordingConfig,
        has_audio: bool,
        audio_sample_rate: int,
    ) -> None:
        """Record the output directory and encode settings; spawn ffmpeg lazily."""
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._has_audio = has_audio
        self._audio_sample_rate = audio_sample_rate

        self._proc: subprocess.Popen[bytes] | None = None
        self._video_w: int | None = None
        self._audio_w: int | None = None
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
        return self._failed or (self._proc is not None and self._proc.poll() is not None)

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
        assert self._video_w is not None
        try:
            os.write(self._video_w, np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            self._failed = True
            raise RuntimeError("ffmpeg video pipe broke") from exc

    def feed_audio(self, samples: npt.NDArray[Any]) -> None:
        """Push ``int16`` PCM samples into the audio pipe.

        A no-op when the encoder has no audio pipe.

        Raises:
            RuntimeError: If the encoder is in a failed state.
        """
        if not self._has_audio or self._audio_w is None:
            return
        if self._failed:
            raise RuntimeError("ChunkEncoder is in a failed state")
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        try:
            os.write(self._audio_w, np.ascontiguousarray(samples).tobytes())
        except BrokenPipeError as exc:
            self._failed = True
            raise RuntimeError("ffmpeg audio pipe broke") from exc

    def _spawn(self, width: int, height: int) -> None:
        """Open the pipes and launch ffmpeg for the given frame dimensions."""
        video_r, video_w = os.pipe()
        audio_r: int | None
        audio_w: int | None
        if self._has_audio:
            audio_r, audio_w = os.pipe()
        else:
            audio_r, audio_w = None, None

        argv = _build_argv(
            output_dir=self._output_dir,
            width=width,
            height=height,
            has_audio=self._has_audio,
            audio_sample_rate=self._audio_sample_rate,
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

        self._video_w = video_w
        self._audio_w = audio_w
        self._width = width
        self._height = height
        logger.info(
            "recorder encoder spawned ffmpeg",
            pid=self._proc.pid,
            width=width,
            height=height,
            audio=self._has_audio,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Close the input pipes and shut ffmpeg down, escalating to a kill.

        Always safe to call; a no-op when ffmpeg never spawned.
        """
        with self._lock:
            # Latch first so a concurrent ``feed_video`` bails out instead of
            # spawning over the fds about to be closed.
            self._stopped = True
            for fd in (self._video_w, self._audio_w):
                if fd is not None:
                    with contextlib.suppress(OSError):
                        os.close(fd)
            self._video_w = None
            self._audio_w = None
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
