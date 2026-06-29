"""Outbound media topology — :class:`Output`.

Subclass :class:`Output` with fields annotated :class:`Video` or :class:`Audio`;
each field becomes an outbound track named after the field. Declaring the
subclass resolves those annotations into the :class:`TrackInfo` records cached on
the class; an instance carries one payload per track and is what a model emits.
"""

from __future__ import annotations

from typing import Any, ClassVar

from reactor_runtime.core.values import TrackDirection, TrackInfo
from reactor_runtime.interface.tracks.descriptors import _resolve_tracks

OUTPUT_REGISTRY: dict[str, type[Output]] = {}
"""Every :class:`Output` subclass that declared at least one track, by class name.

Auto-populated when a track-bearing subclass is created. The runtime runs one
model per process, so this is the model's outbound topology; :func:`all_output_tracks`
unions it into the track set the schema and output buffer read.
"""


class Output:
    """Base for a model's outbound media tracks.

    Subclass with fields annotated :class:`Video` or :class:`Audio`; each field
    becomes an outbound track named after the field. An instance carries one
    payload per track and is what a model passes to ``emit``::

        class GameOutput(Output):
            main_video: Video

        await self.emit(GameOutput(main_video=frame))

    Declaring a track-bearing subclass registers it in :data:`OUTPUT_REGISTRY`; a
    subclass that resolves no tracks (an abstract base or mixin) is left out.
    """

    __tracks__: ClassVar[dict[str, TrackInfo]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__tracks__ = _resolve_tracks(cls, TrackDirection.OUT)
        if cls.__tracks__:
            OUTPUT_REGISTRY[cls.__name__] = cls

    def __init__(self, **payloads: Any) -> None:
        """Bind one payload per declared track.

        Args:
            payloads: Track name to its frame data, one for every declared track.

        Raises:
            TypeError: If the payloads do not match the declared tracks exactly.
        """
        expected = set(type(self).__tracks__)
        given = set(payloads)
        if given != expected:
            raise TypeError(
                f"{type(self).__name__} expects payloads for {sorted(expected)}, "
                f"got {sorted(given)}"
            )
        for name, data in payloads.items():
            setattr(self, name, data)


def all_output_tracks() -> dict[str, TrackInfo]:
    """Return the union of outbound tracks across every registered :class:`Output`.

    Two subclasses that declare a track of the same name collapse to one entry
    (the later registration wins) rather than conflicting — an inheritance chain
    re-declaring a track is not an error.
    """
    tracks: dict[str, TrackInfo] = {}
    for cls in OUTPUT_REGISTRY.values():
        tracks.update(cls.__tracks__)
    return tracks
