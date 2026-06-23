"""Inbound media topology — :class:`Input`.

Subclass :class:`Input` with fields annotated :class:`Video` or :class:`Audio`;
each field becomes an inbound track named after the field. The runtime binds a
live readable buffer to each track, reachable as ``self.<handle>.<track>``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from reactor_runtime.core.values import TrackDirection, TrackInfo
from reactor_runtime.interface.tracks.descriptors import _resolve_tracks


class Input:
    """Base for a model's inbound media tracks.

    Subclass with fields annotated :class:`Video` or :class:`Audio`; each field
    becomes an inbound track named after the field. The runtime binds a live
    readable buffer to each track, reachable as ``self.<handle>.<track>``.
    """

    __tracks__: ClassVar[dict[str, TrackInfo]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__tracks__ = _resolve_tracks(cls, TrackDirection.IN)

    def __init__(self, **buffers: Any) -> None:
        """Bind one readable buffer per declared track.

        Args:
            buffers: Track name to its live input buffer.
        """
        for name, buffer in buffers.items():
            setattr(self, name, buffer)
