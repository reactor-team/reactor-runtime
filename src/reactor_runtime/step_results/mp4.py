"""One MP4 per step, with every ``Output`` track as its own stream.

A step's media is written as a single ``output.mp4``: one video stream per
video track at that track's own size, one mono audio stream per audio track at
its own sample rate, all on one timeline. The first stream of each kind carries
the default disposition, so a player that shows one picture and one sound shows
the first-declared pair; a reader that wants the rest finds each stream by its
index, which the step's ``result.json`` maps to the track name.

Streams are interleaved frame by frame as they are encoded, with the audio cut
into the slice of samples that belongs to each frame, so the muxer never has to
hold a whole stream before it can write.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any, cast

import numpy as np

from reactor_runtime.core.service import StepResultsConfig
from reactor_runtime.core.values import TrackKind
from reactor_runtime.step_results.result import MEDIA_FILENAME, StepResultCancelledError
from reactor_runtime.step_results.validate import PreparedTrack

# The output pixel format and profile, shared with the session recorder: with
# rgb24 input and no explicit choice libx264 picks a 4:4:4 profile many
# decoders reject; yuv420p with the Main profile plays everywhere.
_PIXEL_FORMAT = "yuv420p"
_PROFILE = "Main"
_VIDEO_ENCODERS = {"h264": "libx264", "h265": "libx265"}

# Bounds the rational the float frame rate is turned into, wide enough that
# 23.976 and 29.97 round-trip as 24000/1001 and 30000/1001.
_RATE_DENOMINATOR = 100_000


@dataclass(frozen=True)
class StreamInfo:
    """One stream of the step's media file, as ``result.json`` describes it.

    Attributes:
        name: The track the stream was encoded from.
        kind: Video or audio.
        stream: The stream's index in the file.
        default: Whether a player picks this stream by default.
        width: Frame width, for video.
        height: Frame height, for video.
        fps: Frames per second, for video.
        frames: How many frames the stream holds, for video.
        sample_rate: Samples per second, for audio.
        samples: How many samples the stream holds, for audio.
    """

    name: str
    kind: TrackKind
    stream: int
    default: bool = False
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frames: int | None = None
    sample_rate: int | None = None
    samples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render the stream for ``result.json``, leaving out the fields of the other kind."""
        entry: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "file": MEDIA_FILENAME,
            "stream": self.stream,
        }
        for key in ("width", "height", "fps", "frames", "sample_rate", "samples"):
            value = getattr(self, key)
            if value is not None:
                entry[key] = value
        if self.default:
            entry["default"] = True
        return entry


def _video_options(config: StepResultsConfig) -> dict[str, str]:
    """The private encoder options for the configured video codec."""
    options = {"preset": config.video_preset, "crf": str(config.video_crf)}
    if config.video_codec != "h264":
        # x265 prints a configuration banner at its default verbosity, which
        # libav's own log level does not reach.
        options["x265-params"] = "log-level=warning"
    return options


def encode_mp4(
    path: Path,
    tracks: list[PreparedTrack],
    fps: float,
    config: StepResultsConfig,
    cancelled: Callable[[], bool] = lambda: False,
) -> list[StreamInfo]:
    """Encode a step's tracks into one MP4 at *path*.

    Every video track plays at *fps*; every audio track is padded with silence
    or trimmed so it ends with the video. A file with no video track holds the
    audio at its full length. The file is written with its index at the front,
    so a download can start playing before it finishes.

    Args:
        path: Where the file is written. Overwritten if present.
        tracks: The step's tracks, validated and in declaration order.
        fps: The frame rate every video stream plays at.
        config: The codec settings.
        cancelled: Polled between frames; a true reading stops the encode.

    Returns:
        One record per stream, in stream order.

    Raises:
        ValueError: If there is nothing to encode.
        StepResultCancelledError: If *cancelled* turned true before the end.
        RuntimeError: If the encoder rejected the media.
    """
    # Imported here so the package stays importable without the media library,
    # which rendering a model's schema never needs.
    import av
    from av.stream import Disposition

    if not tracks:
        raise ValueError("a step result needs at least one track to encode")
    videos = [track for track in tracks if track.kind is TrackKind.VIDEO]
    audios = [track for track in tracks if track.kind is TrackKind.AUDIO]
    rate = Fraction(fps).limit_denominator(_RATE_DENOMINATOR)
    if videos:
        n_frames = videos[0].frames
    else:
        # Audio alone still rides the frame grid, long enough to hold every sample.
        n_frames = max(ceil(track.data.size * fps / track.rate) for track in audios)

    container = av.open(str(path), mode="w", format="mp4", options={"movflags": "+faststart"})
    infos: list[StreamInfo] = []
    try:
        video_streams: list[tuple[av.VideoStream, PreparedTrack]] = []
        for index, track in enumerate(videos):
            # ``add_stream`` is overloaded on a literal set of codec names, so
            # the configured codec resolves to the catch-all return type.
            stream = cast(
                "av.VideoStream",
                container.add_stream(
                    _VIDEO_ENCODERS.get(config.video_codec, "libx265"),
                    rate=rate,
                    options=_video_options(config),
                ),
            )
            stream.width = int(track.data.shape[2])
            stream.height = int(track.data.shape[1])
            stream.pix_fmt = _PIXEL_FORMAT
            stream.profile = _PROFILE
            stream.time_base = 1 / rate
            if index == 0:
                stream.disposition = Disposition.default
            video_streams.append((stream, track))
            infos.append(
                StreamInfo(
                    name=track.name,
                    kind=TrackKind.VIDEO,
                    stream=stream.index,
                    default=index == 0,
                    width=stream.width,
                    height=stream.height,
                    fps=float(rate),
                    frames=n_frames,
                )
            )
        audio_streams: list[tuple[av.AudioStream, PreparedTrack]] = []
        for index, track in enumerate(audios):
            stream = cast(
                "av.AudioStream",
                container.add_stream(config.audio_codec, rate=track.rate, layout="mono"),
            )
            stream.bit_rate = config.audio_bitrate_kbps * 1000
            if index == 0:
                stream.disposition = Disposition.default
            audio_streams.append((stream, track))
            infos.append(
                StreamInfo(
                    name=track.name,
                    kind=TrackKind.AUDIO,
                    stream=stream.index,
                    default=index == 0,
                    sample_rate=track.rate,
                    samples=round(n_frames * track.rate / fps),
                )
            )

        for frame_index in range(n_frames):
            if cancelled():
                raise StepResultCancelledError("step result save cancelled")
            for stream, track in video_streams:
                picture = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(track.data[frame_index]), format="rgb24"
                )
                picture.pts = frame_index
                picture.time_base = 1 / rate
                container.mux(stream.encode(picture))
            for stream, track in audio_streams:
                lo = round(frame_index * track.rate / fps)
                hi = round((frame_index + 1) * track.rate / fps)
                block = np.zeros(hi - lo, dtype=np.int16)
                available = track.data[lo:hi]
                block[: available.size] = available
                if not block.size:
                    continue
                samples = av.AudioFrame.from_ndarray(
                    block.reshape(1, -1), format="s16", layout="mono"
                )
                samples.rate = track.rate
                samples.pts = lo
                samples.time_base = Fraction(1, track.rate)
                container.mux(stream.encode(samples))
        for stream, _ in (*video_streams, *audio_streams):
            container.mux(stream.encode(None))
    except av.FFmpegError as exc:
        raise RuntimeError("the step result encoder rejected the media") from exc
    finally:
        container.close()
    return infos
