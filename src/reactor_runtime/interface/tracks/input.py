"""Inbound media topology — :class:`Input`.

Subclass :class:`Input` with fields annotated :class:`Video` or :class:`Audio`;
each field becomes an inbound track named after the field. The runtime binds a
live readable buffer to each track, reachable as ``self.<handle>.<track>``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from reactor_runtime.core.values import TrackDirection, TrackInfo
from reactor_runtime.interface.tracks.descriptors import _resolve_tracks

INPUT_REGISTRY: dict[str, type[Input]] = {}
"""Every :class:`Input` subclass that declared at least one track, by class name.

Auto-populated when a track-bearing subclass is created. :func:`all_input_tracks`
unions it for the schema; the readable buffers a model reads from stay bound to
the model's annotated :class:`Input` holder, not the union.
"""


class Input:
    """Base for a model's inbound media tracks.

    Subclass with fields annotated :class:`Video` or :class:`Audio`; each field
    becomes an inbound track named after the field. The runtime binds a live
    readable buffer to each track, reachable as ``self.<handle>.<track>``.

    Declaring a track-bearing subclass registers it in :data:`INPUT_REGISTRY`; a
    subclass that resolves no tracks (an abstract base or mixin) is left out.
    """

    __tracks__: ClassVar[dict[str, TrackInfo]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__tracks__ = _resolve_tracks(cls, TrackDirection.IN)
        if cls.__tracks__:
            INPUT_REGISTRY[cls.__name__] = cls

    def __init__(self, **buffers: Any) -> None:
        """Bind one readable buffer per declared track.

        Args:
            buffers: Track name to its live input buffer.
        """
        for name, buffer in buffers.items():
            setattr(self, name, buffer)


def all_input_tracks() -> dict[str, TrackInfo]:
    """Return the union of inbound tracks across every registered :class:`Input`.

    Two subclasses that declare a track of the same name collapse to one entry
    (the later registration wins) rather than conflicting — an inheritance chain
    re-declaring a track is not an error.
    """
    tracks: dict[str, TrackInfo] = {}
    for cls in INPUT_REGISTRY.values():
        tracks.update(cls.__tracks__)
    return tracks
