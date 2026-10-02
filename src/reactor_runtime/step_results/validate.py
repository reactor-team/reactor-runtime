"""Checks on what a step hands the store: the media batch and the extra files.

A save runs off the model thread and writes to disk, so a batch or a filename
the store cannot take is refused here, at the call, with a message that names
the track or the file. Video is normalised to one ``(N, H, W, 3)`` array per
track and audio to one flat ``int16`` vector, which is the shape the encoder
reads.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from reactor_runtime.core.values import MediaBundle, TrackData, TrackKind
from reactor_runtime.step_results.result import MEDIA_FILENAME, RESULT_FILENAME

# A file a model adds to a step folder: one path segment, no leading dot, so it
# can neither escape the folder nor hide from a listing.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RESERVED_FILENAMES = frozenset({RESULT_FILENAME, MEDIA_FILENAME})


@dataclass(frozen=True, eq=False)
class PreparedTrack:
    """One track of a step, normalised for the encoder.

    Attributes:
        name: The track name, which labels its stream in the media file.
        kind: Video or audio.
        data: ``(N, H, W, 3)`` ``uint8`` for video; a flat ``int16`` vector for
            audio.
        rate: The audio sample rate in Hz; ``0`` for video.
        metadata: The per-frame metadata the track carried, as the model
            attached it, or ``None``.
    """

    name: str
    kind: TrackKind
    data: npt.NDArray[Any]
    rate: int = 0
    metadata: bytes | list[bytes] | None = None

    @property
    def frames(self) -> int:
        """How many frames a video track carries."""
        return int(self.data.shape[0])


def prepare_tracks(bundle: MediaBundle, fps: float) -> list[PreparedTrack]:
    """Validate a step's media and normalise each track for the encoder.

    Every video track is checked to be non-empty ``uint8`` RGB and all video
    tracks must carry the same number of frames, because the streams share one
    timeline. Audio is mono ``int16`` with an integer sample rate. Tracks keep
    the bundle's order, which is the ``Output`` class's declaration order.

    Args:
        bundle: The step's media, one payload per declared track.
        fps: The rate every video stream plays at.

    Returns:
        The normalised tracks, in declaration order.

    Raises:
        ValueError: If the frame rate, a video payload, or an audio payload is
            not something the encoder can take, or the video tracks disagree
            on their frame count.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"step result fps must be finite and positive, got {fps}")
    prepared = [_prepare(track) for track in bundle.get_tracks()]
    counts = {track.frames for track in prepared if track.kind is TrackKind.VIDEO}
    if len(counts) > 1:
        raise ValueError(
            "every video track of a step must carry the same number of frames, "
            f"got {sorted(counts)}"
        )
    return prepared


def _prepare(track: TrackData) -> PreparedTrack:
    """Normalise one track, refusing a payload the encoder cannot take."""
    name = track.info.name
    data = np.asarray(track.data)
    if track.info.kind is TrackKind.VIDEO:
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        if data.ndim != 4 or data.shape[-1] != 3 or not all(data.shape) or data.dtype != np.uint8:
            raise ValueError(
                f"track '{name}' must be non-empty uint8 RGB frames, (H, W, 3) or (N, H, W, 3), "
                f"got shape {tuple(data.shape)} of {data.dtype}"
            )
        return PreparedTrack(name, TrackKind.VIDEO, data, metadata=track.metadata)
    rate = track.info.rate
    if not math.isfinite(rate) or rate <= 0 or not float(rate).is_integer():
        raise ValueError(f"track '{name}' must declare an integer sample rate, got {rate}")
    if data.ndim == 2 and data.shape[0] == 1:
        data = data.reshape(-1)
    if data.ndim != 1 or data.dtype != np.int16:
        raise ValueError(
            f"track '{name}' must be mono int16 samples, (1, M) or (M,), "
            f"got shape {tuple(data.shape)} of {data.dtype}"
        )
    return PreparedTrack(name, TrackKind.AUDIO, data, rate=int(rate))


def check_filenames(files: Mapping[str, bytes | Path]) -> None:
    """Refuse an extra file whose name cannot sit in a step folder.

    Args:
        files: The extra files a model adds to the step, by name.

    Raises:
        ValueError: If a name is empty, carries a path separator, starts with
            a dot, or collides with the files the store writes itself.
    """
    for name in files:
        if not _FILENAME_RE.fullmatch(name):
            raise ValueError(
                f"step result file name {name!r} must be one plain path segment "
                "(letters, digits, '.', '_', '-'; not starting with '.')"
            )
        if name in _RESERVED_FILENAMES:
            raise ValueError(f"step result file name {name!r} is written by the runtime")
