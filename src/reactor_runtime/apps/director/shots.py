"""The shots a director app holds, and the queues and history that hold them.

A shot passes through three places: the generation queue while it waits to
build, the playout queue once it is built and waits to play, and the history
once it has played or will not play. Only bookkeeping lives here; the app
decides when a shot moves.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from reactor_runtime.interface.tracks import Output


@dataclass(frozen=True)
class Shot:
    """A built shot, as ``generate()`` receives it through a parameter typed ``Shot``.

    Attributes:
        id: The shot's id, the one the client sent.
        inputs: The values the shot was generated from, by ``generate()``
            parameter name. A ``Shot`` parameter holds the id of the shot it
            named, or ``None``.
        last_frame: The shot's last video frame, ``(H, W, 3)`` ``uint8`` RGB,
            or ``None`` when the shot has no video the app could read.
    """

    id: str
    inputs: Mapping[str, Any]
    last_frame: npt.NDArray[np.uint8] | None


class ShotStatus(StrEnum):
    """Where a shot is, as clients read it on every shot they receive."""

    QUEUED = "queued"
    """In the generation queue, waiting to build."""
    BUILDING = "building"
    """``generate()`` is running for it."""
    READY = "ready"
    """Built, in the playout queue, waiting to play."""
    PLAYING = "playing"
    """Its frames are on the output tracks."""
    BUILT = "built"
    """Built and in the history: played, discarded, or a file that does not play."""
    FAILED = "failed"
    """It did not build; ``error`` says why."""


@dataclass(eq=False)
class ShotEntry:
    """One shot, from ``enqueue`` to the history.

    Attributes:
        id: The shot's id.
        inputs: The values ``generate()`` is called with, by parameter name,
            with each ``Shot`` parameter holding the id it named or ``None``.
        sources: The ids the shot depends on, by ``Shot`` parameter name.
        continues: The id of the shot it continues, when the app has a
            ``continue_from`` parameter and the client set it.
        metadata: The client's string, echoed back on every message.
        status: Where the shot is.
        output: The built media, while the shot can still play.
        file: The built file, while the app still needs it.
        last_frame: The last video frame, once built.
        frames: How many frames the shot has, once built and known.
        error: Why the shot failed.
    """

    id: str
    inputs: dict[str, Any]
    sources: dict[str, str]
    continues: str | None
    metadata: str
    status: ShotStatus = ShotStatus.QUEUED
    output: Output | None = None
    file: Path | None = None
    last_frame: npt.NDArray[np.uint8] | None = None
    frames: int | None = None
    error: str | None = None

    def shot(self) -> Shot:
        """Return the read-only view ``generate()`` receives."""
        return Shot(
            id=self.id, inputs=MappingProxyType(dict(self.inputs)), last_frame=self.last_frame
        )

    def depends_on(self, shot_ids: set[str]) -> bool:
        """Return whether the shot depends on any of *shot_ids*."""
        return not shot_ids.isdisjoint(self.sources.values())


@dataclass
class ShotQueue:
    """A bounded, ordered queue of shots, addressed by position.

    Position 0 is the front. Nothing reorders the queue except :meth:`add`
    with a position and :meth:`move`.

    Attributes:
        capacity: The most shots the queue holds.
    """

    capacity: int
    _entries: list[ShotEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ShotEntry]:
        return iter(tuple(self._entries))

    def __contains__(self, entry: object) -> bool:
        return any(existing is entry for existing in self._entries)

    @property
    def full(self) -> bool:
        """Whether the queue holds :attr:`capacity` shots."""
        return len(self._entries) >= self.capacity

    def add(self, entry: ShotEntry, position: int | None = None) -> int:
        """Insert *entry* at *position*, the back when ``None``, and return where it landed.

        A position past the back is the back.

        Raises:
            ValueError: The queue is full.
        """
        if self.full:
            raise ValueError(f"the queue is full ({self.capacity} shots)")
        index = (
            len(self._entries) if position is None else min(max(position, 0), len(self._entries))
        )
        self._entries.insert(index, entry)
        return index

    def move(self, entry: ShotEntry, position: int) -> int:
        """Move *entry* to *position*, the back when past it, and return where it landed."""
        self.remove(entry)
        index = min(max(position, 0), len(self._entries))
        self._entries.insert(index, entry)
        return index

    def remove(self, entry: ShotEntry) -> None:
        """Take *entry* out, when the queue holds it."""
        self._entries = [existing for existing in self._entries if existing is not entry]

    def get(self, shot_id: str) -> ShotEntry | None:
        """Return the shot with *shot_id*, or ``None``."""
        return next((entry for entry in self._entries if entry.id == shot_id), None)

    def head(self) -> ShotEntry | None:
        """Return the front shot, or ``None`` when the queue is empty."""
        return self._entries[0] if self._entries else None

    def next_to_build(self, ready: Callable[[ShotEntry], bool]) -> ShotEntry | None:
        """Return the front-most queued shot that *ready* accepts.

        A shot *ready* refuses, for one waiting on a shot not built yet, is
        skipped rather than holding up the shots behind it.
        """
        return next(
            (
                entry
                for entry in self._entries
                if entry.status is ShotStatus.QUEUED and ready(entry)
            ),
            None,
        )

    def clear(self) -> list[ShotEntry]:
        """Take every shot out and return them, front first."""
        entries, self._entries = self._entries, []
        return entries


@dataclass
class History:
    """Built shots kept so later shots can continue them, oldest first.

    Attributes:
        capacity: The most shots kept, past which the oldest go first. A shot
            a queued shot depends on is never evicted, so the history can hold
            more for a while.
    """

    capacity: int
    _entries: dict[str, ShotEntry] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ShotEntry]:
        return iter(tuple(self._entries.values()))

    def put(self, entry: ShotEntry) -> None:
        """Keep *entry* as the newest shot."""
        self._entries.pop(entry.id, None)
        self._entries[entry.id] = entry

    def get(self, shot_id: str) -> ShotEntry | None:
        """Return the shot with *shot_id*, or ``None``."""
        return self._entries.get(shot_id)

    def trim(self, protected: set[str]) -> list[ShotEntry]:
        """Evict the oldest shots down to :attr:`capacity`, sparing *protected*, and return them."""
        evicted: list[ShotEntry] = []
        for shot_id in list(self._entries):
            if len(self._entries) <= self.capacity:
                break
            if shot_id not in protected:
                evicted.append(self._entries.pop(shot_id))
        return evicted

    def clear(self) -> list[ShotEntry]:
        """Take every shot out and return them, oldest first."""
        entries = list(self._entries.values())
        self._entries = {}
        return entries


def dependents(queue: ShotQueue, lost: set[str]) -> list[ShotEntry]:
    """Return every shot in *queue* that depends on a shot in *lost*, directly or through another.

    The order is the order they are found in: first the direct dependents, front
    first, then theirs.
    """
    found: list[ShotEntry] = []
    lost = set(lost)
    while True:
        layer = [entry for entry in queue if entry not in found and entry.depends_on(lost)]
        if not layer:
            return found
        found.extend(layer)
        lost.update(entry.id for entry in layer)
