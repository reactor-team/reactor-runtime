"""Continuous fMP4 chunk encoder built on the libav bindings.

The encoder is lazy: the first :meth:`ChunkEncoder.feed_video` opens the output
from the frame's shape. Output is HLS-fMP4 (``init.mp4`` + ``chunk_NNNNN.m4s``)
written straight into the served directory. Every recording carries an audio
track — a video-only model gets a silent one, so each segment holds both streams
whatever the model emits.
"""

from __future__ import annotations

import contextlib
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import numpy.typing as npt

from reactor_runtime.core import RecordingConfig
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_INIT_FILENAME = "init.mp4"
_SEGMENT_PATTERN = "chunk_%05d.m4s"
_MANIFEST_FILENAME = "manifest.m3u8"

# The output pixel format and profile. With rgb24 input and no explicit choice,
# libx264 selects yuv444p (Hi444PP), which many decoders and uploaders reject;
# yuv420p with the Main profile is the universally-compatible sub-profile.
_PIXEL_FORMAT = "yuv420p"
_PROFILE = "Main"


def _video_options(config: RecordingConfig, keyframe_interval: int) -> dict[str, str]:
    """Build the private encoder options for the configured video codec.

    A hard GOP — ``keyint`` equal to ``min-keyint``, scene cuts off — puts a
    keyframe on every chunk boundary, which is what the muxer needs to close a
    segment there and what ``independent_segments`` promises a player.

    Args:
        config: The recording settings the preset and quality target come from.
        keyframe_interval: Frames per chunk, so the GOP matches a segment.
    """
    gop = f"keyint={keyframe_interval}:min-keyint={keyframe_interval}:scenecut=0"
    options = {
        "preset": config.video_preset,
        "crf": str(config.video_crf),
        "tune": "zerolatency",
    }
    if config.video_codec == "h264":
        # x264 sliced threads (auto-enabled by the zerolatency tune) can deadlock
        # at higher resolutions; frame threads avoid it and the recorder does not
        # need per-frame latency.
        options["x264-params"] = f"sliced-threads=0:{gop}"
    else:
        # x265 prints a configuration banner to stderr at its default verbosity,
        # which libav's own log level does not reach.
        options["x265-params"] = f"log-level=warning:{gop}"
    return options


class ChunkEncoder:
    """Encodes one recording into HLS fMP4 segments.

    Lifecycle: construction records the output directory; the first
    :meth:`feed_video` opens the output from the frame's shape; each subsequent
    feed encodes and muxes; :meth:`stop` drains the encoders and writes the
    trailer, which is what closes the final segment.

    The recorded timeline is media time, not wall-clock time: a frame is stamped
    at the next position on a fixed *frame_rate* grid and an audio block at the
    sample that follows the last one. The recorder resamples each chunk onto that
    grid before feeding, so a model running faster or slower than real time still
    records at true duration and a stall between feeds records no dead air.

    Encoding happens on the calling thread. The recorder feeds from a worker of
    its own and libav releases the GIL, so the model thread is never blocked by
    it. Feeds and :meth:`stop` are serialised against each other, because an
    output container cannot be used from two threads at once.
    """

    def __init__(
        self,
        output_dir: Path,
        config: RecordingConfig,
        has_audio: bool,
        audio_sample_rate: int,
        frame_rate: int = 30,
    ) -> None:
        """Record the output directory and encode settings; open the output lazily."""
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._has_audio = has_audio
        self._audio_sample_rate = audio_sample_rate
        self._frame_rate = frame_rate
        # One grid slot of audio: what the recorder pairs with each frame, and
        # what a video-only recording gets as silence in its place.
        self._samples_per_frame = round(audio_sample_rate / frame_rate)

        self._container: av.container.OutputContainer | None = None
        self._video: av.VideoStream | None = None
        self._audio: av.AudioStream | None = None
        # The next grid position for each stream. Video counts frames, audio
        # counts samples, and both advance only by what was actually encoded, so
        # the two timelines stay locked to each other.
        self._video_pts = 0
        self._audio_pts = 0
        self._lock = threading.Lock()
        self._failed = False
        self._stopped = False

    @property
    def output_dir(self) -> Path:
        """The directory the init and chunk segments are written into."""
        return self._output_dir

    @property
    def failed(self) -> bool:
        """Return whether the encoder is dead and unrecoverable for this session."""
        return self._failed

    def feed_video(self, frame: npt.NDArray[Any]) -> None:
        """Encode a single ``(H, W, 3)`` ``uint8`` frame at the next grid position.

        A frame sized differently from the one the encoder opened with is scaled
        onto that size, so a changing resolution never breaks the recording. A
        video-only recording gets its slot of silence here too, so the audio
        timeline advances in step with the video whether or not the model emits
        any sound.

        Raises:
            ValueError: If *frame* is not a three-channel image.
            RuntimeError: If the encoder is stopped or in a failed state, or if
                libav rejected the frame.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"feed_video expects (H, W, 3); got shape {frame.shape}")
        with self._lock:
            if self._stopped:
                raise RuntimeError("ChunkEncoder is stopped")
            if self._failed:
                raise RuntimeError("ChunkEncoder is in a failed state")
            if self._container is None:
                width, height = self._config.target_width, self._config.target_height
                if width and height:
                    self._open(int(width), int(height))
                else:
                    self._open(int(frame.shape[1]), int(frame.shape[0]))
            try:
                self._write_video(frame)
                if not self._has_audio:
                    self._write_audio(np.zeros(self._samples_per_frame, dtype=np.int16))
            except av.FFmpegError as exc:
                self._failed = True
                raise RuntimeError("the encoder stopped accepting video") from exc

    def feed_audio(self, samples: npt.NDArray[Any]) -> None:
        """Encode ``int16`` PCM samples at the next audio position.

        A no-op for a video-only recording, whose silent track is written
        alongside the video in :meth:`feed_video`, and before the first frame has
        opened the output.

        Raises:
            RuntimeError: If the encoder is in a failed state, or if libav
                rejected the samples.
        """
        with self._lock:
            if not self._has_audio or self._container is None:
                return
            if self._failed:
                raise RuntimeError("ChunkEncoder is in a failed state")
            try:
                self._write_audio(samples)
            except av.FFmpegError as exc:
                self._failed = True
                raise RuntimeError("the encoder stopped accepting audio") from exc

    def stop(self) -> None:
        """Drain the encoders and write the trailer, closing the final segment.

        Always safe to call, and a no-op when the output never opened or has
        already been closed.
        """
        with self._lock:
            # Latch first so a concurrent feed bails out instead of encoding into
            # the container about to be closed.
            self._stopped = True
            container, video, audio = self._container, self._video, self._audio
            self._container = None
            self._video = None
            self._audio = None
        if container is None:
            return
        try:
            # A failed encoder has no coherent state left to drain; closing the
            # container still writes what already reached the muxer.
            if not self._failed:
                for stream in (video, audio):
                    if stream is not None:
                        container.mux(stream.encode(None))
        except av.FFmpegError:
            logger.exception("recorder failed to drain the encoder")
        finally:
            with contextlib.suppress(av.FFmpegError):
                container.close()

    def _open(self, width: int, height: int) -> None:
        """Open the HLS output and add the video and audio streams.

        The muxer numbers segments from zero, which the marker math and the
        manifest endpoint already assume. The ``.m3u8`` it writes alongside them
        is ignored — the ``/clips`` endpoint composes its own manifest.
        """
        config = self._config
        container = av.open(
            str(self._output_dir / _MANIFEST_FILENAME),
            mode="w",
            format="hls",
            options={
                "hls_segment_type": "fmp4",
                "hls_time": str(config.chunk_seconds),
                "hls_list_size": "0",
                "hls_flags": "independent_segments",
                "hls_fmp4_init_filename": _INIT_FILENAME,
                "hls_segment_filename": str(self._output_dir / _SEGMENT_PATTERN),
            },
        )
        video = container.add_stream(
            "libx264" if config.video_codec == "h264" else "libx265",
            rate=Fraction(self._frame_rate, 1),
            options=_video_options(config, self._frame_rate * config.chunk_seconds),
        )
        video.width = width
        video.height = height
        video.pix_fmt = _PIXEL_FORMAT
        video.profile = _PROFILE
        video.time_base = Fraction(1, self._frame_rate)

        # ``add_stream`` is overloaded on a literal set of codec names, so the
        # configured codec resolves to the catch-all return type.
        audio = cast(
            "av.AudioStream",
            container.add_stream(config.audio_codec, rate=self._audio_sample_rate, layout="mono"),
        )
        audio.bit_rate = config.audio_bitrate_kbps * 1000

        self._container = container
        self._video = video
        self._audio = audio
        logger.info(
            "recorder encoder opened",
            width=width,
            height=height,
            frame_rate=self._frame_rate,
            audio=self._has_audio,
        )

    def _write_video(self, frame: npt.NDArray[Any]) -> None:
        """Encode one frame at the next grid position and mux what comes out."""
        assert self._container is not None
        assert self._video is not None
        picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        picture.pts = self._video_pts
        picture.time_base = Fraction(1, self._frame_rate)
        self._video_pts += 1
        self._container.mux(self._video.encode(picture))

    def _write_audio(self, samples: npt.NDArray[Any]) -> None:
        """Encode PCM samples at the next audio position and mux what comes out."""
        assert self._container is not None
        assert self._audio is not None
        block = np.ascontiguousarray(samples, dtype=np.int16).reshape(1, -1)
        chunk = av.AudioFrame.from_ndarray(block, format="s16", layout="mono")
        chunk.rate = self._audio_sample_rate
        chunk.pts = self._audio_pts
        chunk.time_base = Fraction(1, self._audio_sample_rate)
        self._audio_pts += block.shape[1]
        self._container.mux(self._audio.encode(chunk))
