"""Join camera frames and robot state of one capture by their capture stamp.

A robot stamps every frame and its state with one ``capture_time_us`` per
capture, and they reach the runtime separately: each camera on its own track,
the state on the data channel. The :class:`InputSynchronizer` holds the newest
of each in a ring per track and hands out a complete set, one item per track,
all with the same stamp::

    sync = InputSynchronizer(tracks=("scene", "wrist"), data_tracks=("state",))
    sync.push("scene", 1_000, scene_frame)
    sync.push("wrist", 1_000, wrist_frame)
    sync.push("state", 1_000, {"joints": [0.1, 0.2]})
    inputs = sync.take()  # InputSet(capture_time_us=1000, frames={...}, data={...})

A set is complete when every track holds an item with exactly the stamp of an
item on the reference track.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

SYNC_BUFFER_SIZE = 4
"""How many items each track keeps by default."""


@dataclass(frozen=True)
class InputSet:
    """The items of one capture, one per track, all with the same stamp.

    Attributes:
        capture_time_us: The stamp every item carries, in microseconds on the
            robot's clock.
        frames: The item of each camera track, keyed by track name. Read-only.
        data: The item of each data track, keyed by track name. Read-only;
            empty when the synchronizer has no data tracks.
    """

    capture_time_us: int
    frames: Mapping[str, Any]
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep read-only copies of the mappings the caller passed."""
        object.__setattr__(self, "frames", MappingProxyType(dict(self.frames)))
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


@dataclass(frozen=True)
class SyncStats:
    """What a synchronizer's input came to.

    Every count runs from the synchronizer's creation; :meth:`InputSynchronizer.reset`
    keeps them.

    Attributes:
        pushed: Items pushed, on every track.
        dropped: Items dropped by a full ring to make room for a newer one,
            including one older than everything a full ring holds, dropped on
            arrival.
        taken: Sets handed out by :meth:`InputSynchronizer.take`.
        skipped: Complete sets passed over because a newer complete set was
            taken first.
        trimmed: Items discarded by a take because their stamp was not newer
            than the set it handed out, the items of that set included.
        resets: Calls to :meth:`InputSynchronizer.reset`.
        buffered: Items each track holds now, keyed by track name. Read-only.
    """

    pushed: int = 0
    dropped: int = 0
    taken: int = 0
    skipped: int = 0
    trimmed: int = 0
    resets: int = 0
    buffered: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep a read-only copy of the per-track counts."""
        object.__setattr__(self, "buffered", MappingProxyType(dict(self.buffered)))


class _Ring:
    """The newest items of one track, ordered by stamp, at most ``size`` of them."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.stamps: list[int] = []
        self.items: list[Any] = []

    def __len__(self) -> int:
        return len(self.stamps)

    def push(self, capture_time_us: int, item: Any) -> int:
        """Insert *item* in stamp order and return how many items were dropped."""
        at = bisect.bisect_left(self.stamps, capture_time_us)
        if at < len(self.stamps) and self.stamps[at] == capture_time_us:
            self.items[at] = item
            return 0
        if len(self.stamps) < self.size:
            self.stamps.insert(at, capture_time_us)
            self.items.insert(at, item)
            return 0
        if at == 0:
            return 1
        del self.stamps[0], self.items[0]
        self.stamps.insert(at - 1, capture_time_us)
        self.items.insert(at - 1, item)
        return 1

    def get(self, capture_time_us: int) -> tuple[bool, Any]:
        """Return whether an item has exactly *capture_time_us*, and the item."""
        at = bisect.bisect_left(self.stamps, capture_time_us)
        if at < len(self.stamps) and self.stamps[at] == capture_time_us:
            return True, self.items[at]
        return False, None

    def trim(self, through_us: int) -> int:
        """Discard every item stamped *through_us* or older and return how many."""
        cut = bisect.bisect_right(self.stamps, through_us)
        del self.stamps[:cut], self.items[:cut]
        return cut

    def clear(self) -> None:
        self.stamps.clear()
        self.items.clear()


class InputSynchronizer:
    """Hand out the newest capture whose every track has arrived, joined by exact stamp.

    Each track keeps its newest ``buffer`` items in a ring ordered by stamp. An
    item whose stamp the ring already holds replaces the one held. When a ring
    is full, its oldest item is dropped; an item older than everything a full
    ring holds is dropped on arrival.

    :meth:`take` looks at the stamps on the reference track, newest first, and
    hands out the first one every other track also holds exactly. Only a set
    newer than the last one taken is handed out, so a capture reaches the
    policy at most once and never after a newer one. Taking a set discards,
    on every track, the items stamped at or before it.

    Items are kept as pushed, without copying or checking them.

    Not thread-safe: it lives on the model's event loop and is touched only
    there.

    Args:
        tracks: The camera tracks, whose items land in :attr:`InputSet.frames`.
        data_tracks: The data tracks, such as the robot state, whose items land
            in :attr:`InputSet.data`. None by default.
        reference: The track whose stamps define the captures, one of
            ``tracks`` or ``data_tracks``. Defaults to the first camera track,
            or the first data track when there is no camera track.
        buffer: How many items each track keeps. At least 1. Defaults to
            :data:`SYNC_BUFFER_SIZE`.

    Raises:
        ValueError: There is no track at all, a track name appears twice,
            ``reference`` is not one of the tracks, or ``buffer`` is less
            than 1.
    """

    def __init__(
        self,
        tracks: Sequence[str],
        data_tracks: Sequence[str] = (),
        reference: str | None = None,
        buffer: int = SYNC_BUFFER_SIZE,
    ) -> None:
        """Create a synchronizer with an empty ring per track."""
        if isinstance(tracks, str) or isinstance(data_tracks, str):
            raise ValueError("tracks and data_tracks are sequences of track names, not one string")
        self._tracks = tuple(tracks)
        self._data_tracks = tuple(data_tracks)
        names = self._tracks + self._data_tracks
        if not names:
            raise ValueError("an input synchronizer needs at least one track")
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"track names must be distinct, got {repeated} more than once")
        if reference is None:
            reference = names[0]
        if reference not in names:
            raise ValueError(
                f"the reference track {reference!r} is not one of the tracks {list(names)}"
            )
        if buffer < 1:
            raise ValueError(f"each track keeps at least 1 item, got buffer={buffer}")
        self._reference = reference
        self._buffer = buffer
        self._rings = {name: _Ring(buffer) for name in names}
        self._last_taken: int | None = None
        self._pushed = 0
        self._dropped = 0
        self._taken = 0
        self._skipped = 0
        self._trimmed = 0
        self._resets = 0

    @property
    def tracks(self) -> tuple[str, ...]:
        """The camera tracks, in the order they were given."""
        return self._tracks

    @property
    def data_tracks(self) -> tuple[str, ...]:
        """The data tracks, in the order they were given."""
        return self._data_tracks

    @property
    def reference(self) -> str:
        """The track whose stamps define the captures."""
        return self._reference

    @property
    def buffer(self) -> int:
        """The most items each track keeps."""
        return self._buffer

    def push(self, track: str, capture_time_us: int, item: Any) -> None:
        """Hold *item* on *track* under its capture stamp.

        Args:
            track: One of the synchronizer's tracks.
            capture_time_us: The item's capture stamp, in microseconds on the
                robot's clock.
            item: The frame or data, kept as is.

        Raises:
            KeyError: *track* is not one of the synchronizer's tracks.
        """
        ring = self._rings.get(track)
        if ring is None:
            raise KeyError(f"unknown track {track!r}, expected one of {list(self._rings)}")
        self._pushed += 1
        self._dropped += ring.push(capture_time_us, item)

    def take(self) -> InputSet | None:
        """Return the newest complete set newer than the last one taken, or ``None``.

        A set handed out discards, on every track, the items stamped at or
        before it, and complete sets older than it count as skipped. ``None``
        changes nothing.
        """
        reference = self._rings[self._reference]
        for at in range(len(reference) - 1, -1, -1):
            stamp = reference.stamps[at]
            if self._last_taken is not None and stamp <= self._last_taken:
                return None
            joined = self._join(stamp)
            if joined is None:
                continue
            self._skipped += sum(
                1
                for older in reference.stamps[:at]
                if (self._last_taken is None or older > self._last_taken)
                and self._join(older) is not None
            )
            for ring in self._rings.values():
                self._trimmed += ring.trim(stamp)
            self._last_taken = stamp
            self._taken += 1
            return InputSet(
                capture_time_us=stamp,
                frames={name: joined[name] for name in self._tracks},
                data={name: joined[name] for name in self._data_tracks},
            )
        return None

    def reset(self) -> None:
        """Drop every held item and forget the last set taken, keeping the counts.

        After a reset, any stamp can be handed out again, older ones included.
        """
        for ring in self._rings.values():
            ring.clear()
        self._last_taken = None
        self._resets += 1

    def stats(self) -> SyncStats:
        """Return the counts since the synchronizer was created."""
        return SyncStats(
            pushed=self._pushed,
            dropped=self._dropped,
            taken=self._taken,
            skipped=self._skipped,
            trimmed=self._trimmed,
            resets=self._resets,
            buffered={name: len(ring) for name, ring in self._rings.items()},
        )

    def _join(self, stamp: int) -> dict[str, Any] | None:
        """Return the item of every track stamped exactly *stamp*, or ``None`` if one lacks it."""
        joined: dict[str, Any] = {}
        for name, ring in self._rings.items():
            found, item = ring.get(stamp)
            if not found:
                return None
            joined[name] = item
        return joined
