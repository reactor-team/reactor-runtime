"""Writing one step's output as a single MP4 file, :func:`write_mp4`.

Every track of the step's output becomes its own stream in the file: the video
tracks first, as H.264 or H.265 streams, then the audio tracks, each in the
order the output declares them. The file is written once, from a whole step,
so it needs none of the recorder's segmenting or timeline work: each video
stream plays its frames at the step's frame rate from the start of the file,
and each audio stream plays its samples at the track's own rate.
"""

from __future__ import annotations

import contextlib
import heapq
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import numpy.typing as npt

from reactor_runtime.core import MediaBundle, StepResultsConfig, TrackData, TrackKind
from reactor_runtime.recording.chunk_encoder import _PIXEL_FORMAT, _PROFILE

# The rate an audio track plays at when its track declares none.
_DEFAULT_SAMPLE_RATE = 48_000


def write_mp4(path: Path, bundle: MediaBundle, fps: float, config: StepResultsConfig) -> None:
    """Encode every track of *bundle* into one MP4 file at *path*.

    A video track's payload is one ``(H, W, 3)`` ``uint8`` frame or a batch of
    them, ``(N, H, W, 3)``, played at *fps*. A frame with an odd width or
    height is padded by one row or column, repeating the edge, because the
    encoder's pixel format needs even dimensions. An audio track's payload is
    ``int16`` mono samples, played at the track's rate. The file is moved to
    a progressive layout, so it plays while it downloads.

    A file that fails part-way is removed, so *path* holds either a whole file
    or nothing.

    Args:
        path: Where to write the file.
        bundle: The step's output, one payload per track.
        fps: The rate the video frames play at, in frames per second.
        config: The codec and quality settings.

    Raises:
        ValueError: If *fps* is not positive, *bundle* has no tracks, or a
            video payload is not an RGB frame or a batch of them.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not bundle.tracks:
        raise ValueError("a step output with no tracks has nothing to encode")
    # Read every payload before the file is opened, so a payload it cannot
    # encode is refused without leaving a file behind.
    videos = [
        (track, _video_frames(track))
        for track in bundle.tracks.values()
        if track.info.kind is TrackKind.VIDEO
    ]
    audios = [track for track in bundle.tracks.values() if track.info.kind is TrackKind.AUDIO]
    container = av.open(str(path), mode="w", format="mp4", options={"movflags": "+faststart"})
    try:
        # The muxer writes the file header with the first packet, so every
        # stream is added before any is encoded.
        video_streams = [_add_video(container, frames, fps, config) for _, frames in videos]
        audio_streams = [_add_audio(container, track, config) for track in audios]
        encoded = [
            *(
                _encode_video(stream, frames)
                for (_, frames), stream in zip(videos, video_streams, strict=True)
            ),
            *(
                _encode_audio(stream, track)
                for track, stream in zip(audios, audio_streams, strict=True)
            ),
        ]
        # The muxer buffers only a bounded span of one stream while it waits
        # for the others, so every stream's packets go in by decode time.
        for packet in heapq.merge(*encoded, key=_decode_time):
            container.mux(packet)
        container.close()
    except BaseException:
        with contextlib.suppress(Exception):
            container.close()
        path.unlink(missing_ok=True)
        raise


def _add_video(
    container: av.container.OutputContainer,
    frames: npt.NDArray[Any],
    fps: float,
    config: StepResultsConfig,
) -> av.VideoStream:
    rate = Fraction(fps).limit_denominator(1001)
    stream = container.add_stream(
        "libx264" if config.video_codec == "h264" else "libx265",
        rate=rate,
        options=_video_options(config),
    )
    stream.width = frames.shape[2] + frames.shape[2] % 2
    stream.height = frames.shape[1] + frames.shape[1] % 2
    stream.pix_fmt = _PIXEL_FORMAT
    stream.profile = _PROFILE
    stream.time_base = 1 / rate
    return stream


def _add_audio(
    container: av.container.OutputContainer, track: TrackData, config: StepResultsConfig
) -> av.AudioStream:
    rate = int(track.info.rate) or _DEFAULT_SAMPLE_RATE
    # ``add_stream`` is overloaded on a literal set of codec names, so the
    # configured codec resolves to the catch-all return type.
    stream = cast(
        "av.AudioStream", container.add_stream(config.audio_codec, rate=rate, layout="mono")
    )
    stream.bit_rate = config.audio_bitrate_kbps * 1000
    return stream


def _encode_video(stream: av.VideoStream, frames: npt.NDArray[Any]) -> list[av.Packet]:
    packets: list[av.Packet] = []
    for index, frame in enumerate(frames):
        picture = av.VideoFrame.from_ndarray(_even(frame), format="rgb24")
        picture.pts = index
        picture.time_base = stream.time_base
        packets.extend(stream.encode(picture))
    packets.extend(stream.encode(None))
    return packets


def _encode_audio(stream: av.AudioStream, track: TrackData) -> list[av.Packet]:
    rate = stream.rate
    samples = np.ascontiguousarray(track.data, dtype=np.int16).reshape(1, -1)
    block = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    block.rate = rate
    block.pts = 0
    block.time_base = Fraction(1, rate)
    return [*stream.encode(block), *stream.encode(None)]


def _decode_time(packet: av.Packet) -> Fraction:
    """Return when *packet* is decoded, in seconds, so streams can be merged in order."""
    stamp = packet.dts if packet.dts is not None else packet.pts
    if stamp is None or packet.time_base is None:
        return Fraction(0)
    return stamp * packet.time_base


def _video_frames(track: TrackData) -> npt.NDArray[Any]:
    """Return a video payload as a batch of ``(H, W, 3)`` frames."""
    data = track.data
    frames = data[np.newaxis] if data.ndim == 3 else data
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.shape[0] == 0:
        raise ValueError(
            f"video track {track.info.name!r} must be (H, W, 3) or (N, H, W, 3); "
            f"got shape {data.shape}"
        )
    return frames


def _even(frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Pad a frame with an odd width or height by one edge row or column."""
    pad_rows, pad_cols = frame.shape[0] % 2, frame.shape[1] % 2
    if pad_rows or pad_cols:
        frame = np.pad(frame, ((0, pad_rows), (0, pad_cols), (0, 0)), mode="edge")
    return np.ascontiguousarray(frame, dtype=np.uint8)


def _video_options(config: StepResultsConfig) -> dict[str, str]:
    """Build the private encoder options for the configured video codec."""
    options = {"preset": config.video_preset, "crf": str(config.video_crf)}
    if config.video_codec != "h264":
        # x265 prints a configuration banner to stderr at its default
        # verbosity, which libav's own log level does not reach.
        options["x265-params"] = "log-level=warning"
    return options
