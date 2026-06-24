"""Media track markers — :class:`Track`, :class:`Video`, :class:`Audio`.

A model declares its media topology by annotating fields with these markers; the
field name is the track name and the marker is its kind. The shared
:func:`_resolve_tracks` helper turns a track-holder's annotations into the
:class:`TrackInfo` records a schema needs.
"""

from __future__ import annotations

from typing import ClassVar, get_type_hints

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

    Attributes:
        sample_rate: Native sample rate in Hz.
    """

    kind = TrackKind.AUDIO
    sample_rate: ClassVar[int] = 48_000


def _resolve_tracks(cls: type, direction: TrackDirection) -> dict[str, TrackInfo]:
    """Resolve a track-holder's annotated fields into :class:`TrackInfo` records.

    Args:
        cls: The :class:`Output` or :class:`Input` subclass to inspect.
        direction: The flow direction for every track the class declares.

    Returns:
        Track name to its metadata, in declaration order.
    """
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    tracks: dict[str, TrackInfo] = {}
    for name, annotation in hints.items():
        if isinstance(annotation, type) and issubclass(annotation, Track):
            rate = float(annotation.sample_rate) if issubclass(annotation, Audio) else 0.0
            tracks[name] = TrackInfo(
                name=name, kind=annotation.kind, rate=rate, direction=direction
            )
    return tracks
