"""Outbound media topology — :class:`Output`.

Subclass :class:`Output` with fields annotated :class:`Video` or :class:`Audio`;
each field becomes an outbound track named after the field. Declaring the
subclass resolves those annotations into the :class:`TrackInfo` records cached on
the class; an instance carries one payload per track and is what a model emits.

A track's payload is the array itself, or a :class:`TrackPayload` wrapping it
when the model has metadata to send with the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy.typing as npt

from reactor_runtime.core.values import TrackDirection, TrackInfo
from reactor_runtime.interface.tracks.descriptors import _resolve_tracks

Metadata = dict[str, Any] | bytes
"""One frame's metadata: a JSON-serialisable mapping, or bytes the model encoded itself."""

OUTPUT_REGISTRY: dict[str, type[Output]] = {}
"""Every :class:`Output` subclass that declared at least one track, by class name.

Auto-populated when a track-bearing subclass is created. The runtime runs one
model per process, so this is the model's outbound topology; :func:`all_output_tracks`
unions it into the track set the schema and output buffer read.
"""


@dataclass(frozen=True, eq=False)
class TrackPayload:
    """A track's frame data plus the metadata to send alongside it.

    Pass one in place of a bare array to tag what a track emits::

        await self.emit(GameOutput(main_video=TrackPayload(frame, metadata={"seed": seed})))

    Equality is identity-based, as it is for any array-bearing value: comparing
    the payloads would raise rather than answer.

    Attributes:
        data: The frame data, exactly as a bare payload would carry it: one
            frame, or a batch of them.
        metadata: One value to send with the frame, or one per frame of a batch.
            A mapping is encoded as JSON; bytes are sent as they are.
    """

    data: npt.NDArray[Any]
    metadata: Metadata | list[Metadata] | None = field(default=None, kw_only=True)


class Output:
    """Base for a model's outbound media tracks.

    Subclass with fields annotated :class:`Video` or :class:`Audio`; each field
    becomes an outbound track named after the field. An instance carries one
    payload per track and is what a model passes to ``emit``::

        class GameOutput(Output):
            main_video: Video

        await self.emit(GameOutput(main_video=frame))

    A payload may also be a :class:`TrackPayload`, which carries metadata for the
    frame along with the array.

    Pass ``fps`` to play this output out at a rate of its own::

        MyOutput(main_video=frames, fps=24)

    Use it when the frames have a duration the model's own speed does not
    describe — a transform that has to play out at the cadence of the video it
    consumed, whatever the hardware managed. Left unset, the runtime paces from
    the time the step took.

    Declaring a track-bearing subclass registers it in :data:`OUTPUT_REGISTRY`; a
    subclass that resolves no tracks (an abstract base or mixin) is left out.
    """

    __tracks__: ClassVar[dict[str, TrackInfo]]
    __metadata__: dict[str, Metadata | list[Metadata]]

    fps: float | None
    """The rate this output's frames play out at, or ``None`` to let the runtime pace it."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__tracks__ = _resolve_tracks(cls, TrackDirection.OUT)
        if "fps" in cls.__tracks__:
            raise TypeError(
                f"{cls.__name__} declares a track named 'fps', which is reserved for the "
                f"playout rate an output may carry. Rename the track."
            )
        if cls.__tracks__:
            OUTPUT_REGISTRY[cls.__name__] = cls

    def __init__(self, *, fps: float | None = None, **payloads: Any) -> None:
        """Bind one payload per declared track.

        A track's attribute holds its array whichever form the payload took, so
        the metadata a :class:`TrackPayload` carried is kept aside under
        ``__metadata__`` rather than shadowing the frame.

        Args:
            fps: The rate these frames play out at. Omit it to have the runtime
                pace the output from how long producing it took.
            payloads: Track name to its frame data, as an array or a
                :class:`TrackPayload`, one for every declared track.

        Raises:
            TypeError: If the payloads do not match the declared tracks exactly.
            ValueError: If *fps* is not positive.
        """
        expected = set(type(self).__tracks__)
        given = set(payloads)
        if given != expected:
            raise TypeError(
                f"{type(self).__name__} expects payloads for {sorted(expected)}, "
                f"got {sorted(given)}"
            )
        if fps is not None and fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        self.__metadata__ = {}
        for name, payload in payloads.items():
            if isinstance(payload, TrackPayload):
                setattr(self, name, payload.data)
                if payload.metadata is not None:
                    self.__metadata__[name] = payload.metadata
            else:
                setattr(self, name, payload)


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
