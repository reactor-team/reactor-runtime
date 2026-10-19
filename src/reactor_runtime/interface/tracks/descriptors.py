"""Media track markers — :class:`Track`, :class:`Video`, :class:`Audio`.

A model declares its media topology by annotating fields with these markers; the
field name is the track name and the marker is its kind. The shared
:func:`_resolve_tracks` helper turns a track-holder's annotations into the
:class:`TrackInfo` records a schema needs.
"""

from __future__ import annotations

import math
from typing import ClassVar, cast, get_type_hints

from reactor_runtime.core.values import TrackDirection, TrackInfo, TrackKind


class Track:
    """Base marker for a media track kind."""

    kind: ClassVar[TrackKind]


class Video(Track):
    """Declares a video track."""

    kind = TrackKind.VIDEO


class Audio(Track):
    """Declares an audio track.

    Subclass to override the rate, e.g. ``class Narration(Audio): sample_rate = 16_000``.

    An input track can bound its buffer in seconds of audio instead of frames,
    so it holds the same length of audio whatever frame size the transport
    delivers::

        class AgentAudio(Audio):
            sample_rate = 24_000
            buffer_seconds = 120

    Attributes:
        sample_rate: Native sample rate in Hz.
        buffer_seconds: How many seconds of audio an input track buffers
            before dropping the oldest, or ``None`` to buffer 128 frames.
            Ignored on an output track.
    """

    kind = TrackKind.AUDIO
    sample_rate: ClassVar[int] = 48_000
    buffer_seconds: ClassVar[float | None] = None


def _resolve_tracks(cls: type, direction: TrackDirection) -> dict[str, TrackInfo]:
    """Resolve a track-holder's annotated fields into :class:`TrackInfo` records.

    Args:
        cls: The :class:`Output` or :class:`MediaInput` subclass to inspect.
        direction: The flow direction for every track the class declares.

    Returns:
        Track name to its metadata, in declaration order.

    Raises:
        ValueError: If an audio track declares a ``buffer_seconds`` that is not
            a finite positive number, or declares one with a ``sample_rate``
            that is not.
    """
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    tracks: dict[str, TrackInfo] = {}
    for name, annotation in hints.items():
        if isinstance(annotation, type) and issubclass(annotation, Track):
            rate = 0.0
            buffer_seconds: float | None = None
            if issubclass(annotation, Audio):
                buffer_seconds = _buffer_seconds(name, annotation.buffer_seconds)
                # Checked before the rate is coerced: float() would accept True or "24000".
                if buffer_seconds is not None and not _finite_positive(annotation.sample_rate):
                    raise ValueError(
                        f"track {name!r}: buffer_seconds needs a positive sample_rate, "
                        f"got {annotation.sample_rate!r}"
                    )
                rate = float(annotation.sample_rate)
            tracks[name] = TrackInfo(
                name=name,
                kind=annotation.kind,
                rate=rate,
                direction=direction,
                buffer_seconds=buffer_seconds,
            )
    return tracks


def _buffer_seconds(name: str, value: object) -> float | None:
    if value is None:
        return None
    if not _finite_positive(value):
        raise ValueError(
            f"track {name!r}: buffer_seconds must be a finite positive number, got {value!r}"
        )
    return float(cast("int | float", value))


def _finite_positive(value: object) -> bool:
    """Whether *value* is a real number, neither a bool nor infinite nor NaN, above zero."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )
