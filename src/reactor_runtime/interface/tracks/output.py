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


class Output:
    """Base for a model's outbound media tracks.

    Subclass with fields annotated :class:`Video` or :class:`Audio`; each field
    becomes an outbound track named after the field. An instance carries one
    payload per track and is what a model passes to ``emit``::

        class GameOutput(Output):
            main_video: Video

        await self.emit(GameOutput(main_video=frame))
    """

    __tracks__: ClassVar[dict[str, TrackInfo]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__tracks__ = _resolve_tracks(cls, TrackDirection.OUT)

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
