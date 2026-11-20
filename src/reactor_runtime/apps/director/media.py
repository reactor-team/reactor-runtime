"""Reading a built shot: its frame count, its slices, its last frame, and a file it came as."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from reactor_runtime.core.values import TrackKind
from reactor_runtime.interface.tracks import Output, TrackPayload


def _video_tracks(output_cls: type[Output]) -> list[str]:
    return [name for name, info in output_cls.__tracks__.items() if info.kind is TrackKind.VIDEO]


def _audio_tracks(output_cls: type[Output]) -> list[str]:
    return [name for name, info in output_cls.__tracks__.items() if info.kind is TrackKind.AUDIO]


def frame_count(output: Output, fps: float) -> int:
    """Return how many frames *output* plays for at *fps*.

    A batched video track, ``(N, H, W, 3)``, has ``N`` frames and an unbatched
    one has one. An output with no video track lasts as long as its longest
    audio track, rounded up to whole frames.

    Raises:
        ValueError: Two batched video tracks disagree on their frame count.
    """
    counts = {
        getattr(output, name).shape[0]
        for name in _video_tracks(type(output))
        if getattr(output, name).ndim == 4
    }
    if len(counts) > 1:
        raise ValueError(f"the video tracks of a shot disagree on its length: {sorted(counts)}")
    if counts:
        return counts.pop()
    if _video_tracks(type(output)):
        return 1
    longest = 0.0
    for name in _audio_tracks(type(output)):
        rate = type(output).__tracks__[name].rate
        longest = max(longest, getattr(output, name).shape[-1] / rate)
    return max(1, int(np.ceil(longest * fps)))


def slice_output(output: Output, lo: int, hi: int, fps: float) -> Output:
    """Return frames ``lo`` to ``hi`` of *output*, with the audio that plays alongside them.

    Each audio track is cut at the sample its frame boundary falls on, so
    consecutive slices hold every sample once.
    """
    output_cls = type(output)
    payloads: dict[str, Any] = {}
    for name, info in output_cls.__tracks__.items():
        data = getattr(output, name)
        metadata = output.__metadata__.get(name)
        if info.kind is TrackKind.VIDEO:
            if data.ndim == 4:
                data = data[lo:hi]
                if isinstance(metadata, list):
                    metadata = metadata[lo:hi]
        else:
            per_frame = info.rate / fps
            data = data[..., round(lo * per_frame) : round(hi * per_frame)]
        payloads[name] = data if metadata is None else TrackPayload(data, metadata=metadata)
    return output_cls(**payloads)


def last_frame(output: Output) -> npt.NDArray[np.uint8] | None:
    """Return a copy of the last frame of *output*'s first video track; ``None`` without video."""
    names = _video_tracks(type(output))
    if not names:
        return None
    data = getattr(output, names[0])
    return np.array(data[-1] if data.ndim == 4 else data, dtype=np.uint8, copy=True)


def png_bytes(frame: npt.NDArray[np.uint8]) -> bytes:
    """Encode one ``(H, W, 3)`` ``uint8`` RGB frame as a PNG image."""
    import av

    height, width = frame.shape[:2]
    codec = av.CodecContext.create("png", "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = "rgb24"
    packets = codec.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24"))
    packets += codec.encode(None)
    return b"".join(bytes(packet) for packet in packets)


def _rgb(frame: Any) -> npt.NDArray[np.uint8]:
    """One decoded video frame as ``(H, W, 3)`` ``uint8`` RGB."""
    return np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)


def read_file_video(path: Path) -> tuple[npt.NDArray[np.uint8] | None, int | None]:
    """Return the last video frame of the file at *path* and its number of frames.

    A file with no video, or one that cannot be read as media, returns
    ``(None, None)``: the shot is still kept, only no later shot can continue
    from its picture.
    """
    import av

    last: npt.NDArray[np.uint8] | None = None
    count = 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None, None
            for frame in container.decode(video=0):
                last = _rgb(frame)
                count += 1
    except (av.FFmpegError, OSError):
        return None, None
    return last, (count or None)


def decode_file(path: Path, output_cls: type[Output], fps: float) -> Output:
    """Decode the file at *path* onto *output_cls*'s tracks, ready to play at *fps*.

    The first video stream goes on the first video track, and the first audio
    stream, resampled to mono ``int16`` at the track's rate, on the first audio
    track. A track the file has no stream for plays black or silence for the
    length of the other.

    Raises:
        ValueError: The class declares more than one video or audio track,
            or the file holds neither video nor audio.
        av.FFmpegError: The file cannot be decoded.
    """
    import av

    video_names, audio_names = _video_tracks(output_cls), _audio_tracks(output_cls)
    if len(video_names) > 1 or len(audio_names) > 1:
        raise ValueError(
            f"{output_cls.__name__} declares more than one video or audio track; a decoded "
            "file plays on one of each"
        )
    frames: list[npt.NDArray[np.uint8]] = []
    chunks: list[npt.NDArray[Any]] = []
    with av.open(str(path)) as container:
        if video_names and container.streams.video:
            frames = [_rgb(frame) for frame in container.decode(video=0)]
    if audio_names:
        rate = int(output_cls.__tracks__[audio_names[0]].rate)
        with av.open(str(path)) as container:
            if container.streams.audio:
                resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
                for frame in container.decode(audio=0):
                    chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
                chunks.extend(f.to_ndarray() for f in resampler.resample(None))
    if not frames and not chunks:
        raise ValueError(f"{path.name} holds no video or audio this app plays")
    payloads: dict[str, Any] = {}
    if video_names:
        if not frames:
            samples = sum(chunk.shape[-1] for chunk in chunks)
            count = max(1, round(samples / output_cls.__tracks__[audio_names[0]].rate * fps))
            frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * count
        payloads[video_names[0]] = np.stack(frames)
    if audio_names:
        rate = output_cls.__tracks__[audio_names[0]].rate
        audio = (
            np.concatenate(chunks, axis=-1).reshape(1, -1)
            if chunks
            else np.zeros((1, round(len(frames) * rate / fps)), dtype=np.int16)
        )
        payloads[audio_names[0]] = audio.astype(np.int16, copy=False)
    return output_cls(**payloads)
