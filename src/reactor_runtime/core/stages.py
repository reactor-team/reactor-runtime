"""Add up the time frames or messages spend in each stage of their trip, over a window.

Whoever times a stage, a transport or the runner, adds each measurement here,
and a report takes the totals. :class:`StageTotals` holds them, keyed by a
track (``None`` for a connection's messages) and a stage name, until a take
starts them again.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Collection, Hashable, Sequence


class StageTotals:
    """The time spent in each stage, and how many items it counted, since the last take.

    Items are timed from the threads that move them, and a report takes the
    totals from the event loop, so every access holds a lock.

    Args:
        order: Stage names in the order an item goes through them. A take
            returns them in this order, and a name not in it after them.
    """

    def __init__(self, order: Sequence[str] = ()) -> None:
        self._lock = threading.Lock()
        self._order = {name: index for index, name in enumerate(order)}
        self._totals: dict[tuple[Hashable, str], tuple[float, int]] = {}

    def add(self, key: Hashable, stage: str, ms: float, count: int = 1) -> None:
        """Add *count* items under *key* that spent *ms* milliseconds in *stage*, together.

        A count below one or a time that is negative or not finite is no
        measurement, and is left out.
        """
        if count < 1 or not math.isfinite(ms) or ms < 0:
            return
        with self._lock:
            total, counted = self._totals.get((key, stage), (0.0, 0))
            self._totals[(key, stage)] = (total + ms, counted + count)

    def take(
        self, keys: Collection[Hashable] | None = None
    ) -> dict[Hashable, list[tuple[str, float, int]]]:
        """Return each key's stages as ``(stage, total_ms, count)``, in order, and start them again.

        Every key when *keys* is ``None``. A key not named keeps its totals
        for a later take.
        """
        with self._lock:
            if keys is None:
                totals, self._totals = self._totals, {}
            else:
                wanted = set(keys)
                totals = {key: v for key, v in self._totals.items() if key[0] in wanted}
                for key in totals:
                    del self._totals[key]
        out: dict[Hashable, list[tuple[str, float, int]]] = {}
        for (key, stage), (total, count) in totals.items():
            out.setdefault(key, []).append((stage, total, count))
        last = len(self._order)
        for found in out.values():
            found.sort(key=lambda s: self._order.get(s[0], last))
        return out

    def clear(self) -> None:
        """Drop every total."""
        with self._lock:
            self._totals.clear()
