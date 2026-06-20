"""Media track declarations — :class:`Output`, :class:`Input`, and markers.

A model declares its media topology by subclassing :class:`Output` (and
optionally :class:`Input`) with fields annotated as :class:`Video` or
:class:`Audio`. The field name is the track name; the marker is its kind::

    class GameOutput(Output):
        main_video: Video
        narration: Audio

    class GameInput(Input):
        camera: Video

Declaring the subclass resolves those annotations into :class:`TrackInfo`
records cached on the class — out for an :class:`Output`, in for an
:class:`Input`. This slice carries the topology a schema needs; the readable
frame buffers an input track exposes at run time belong to the engine.
"""

from __future__ import annotations

from typing import Any, ClassVar, get_type_hints

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


class Output:
    """Base for a model's outbound media tracks.

    Subclass with fields annotated :class:`Video` or :class:`Audio`; each field
    becomes an outbound track named after the field.
    """

    __tracks__: ClassVar[dict[str, TrackInfo]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__tracks__ = _resolve_tracks(cls, TrackDirection.OUT)


class Input:
    """Base for a model's inbound media tracks.

    Subclass with fields annotated :class:`Video` or :class:`Audio`; each field
    becomes an inbound track named after the field.
    """

    __tracks__: ClassVar[dict[str, TrackInfo]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__tracks__ = _resolve_tracks(cls, TrackDirection.IN)
