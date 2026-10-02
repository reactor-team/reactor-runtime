"""Decide which client stats batches the runner journals, and in what shape.

A client reports its own view of its receive-side quality in batches, and the
runner journals each batch as a ``metric`` fact. The client chooses how often
it sends and how much, while the journal keeps a bounded number of facts it
shares with lifecycle facts, so :class:`ClientStatsGate` drops a batch that
comes too soon after its connection's last one or that is larger than the SDK
ever sends.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

from reactor_runtime.core import ClientStatsBatch, ConnId

# The SDK sends a batch every few seconds, with a few tracks and a few dozen
# metrics each; these limits are far above that.
MIN_INTERVAL_SECONDS = 1.0
MAX_TRACKS = 32
MAX_METRICS = 64
MAX_NAME_LENGTH = 128


class ClientStatsGate:
    """Accept or drop each connection's client stats batches, and shape them.

    It keeps when each connection's last batch was accepted, on a monotonic
    clock, so a batch that follows it by less than :data:`MIN_INTERVAL_SECONDS`
    is dropped. The times go with the connection (:meth:`forget`) and with the
    session (:meth:`clear`).
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock
        self._accepted_at: dict[ConnId, float] = {}

    def accept(self, conn_id: ConnId, batch: ClientStatsBatch) -> str | None:
        """Accept *batch* from *conn_id*, or return why it is dropped.

        Returns ``None`` and records the time when the batch is accepted.
        """
        now = (self._clock or time.monotonic)()
        last = self._accepted_at.get(conn_id)
        if last is not None and now - last < MIN_INTERVAL_SECONDS:
            return "sent too soon"
        if not fits(batch):
            return "too large"
        self._accepted_at[conn_id] = now
        return None

    def forget(self, conn_id: ConnId) -> None:
        """Drop the last accepted time of a connection that closed."""
        self._accepted_at.pop(conn_id, None)

    def clear(self) -> None:
        """Drop every connection's last accepted time, at session end."""
        self._accepted_at.clear()


def fits(batch: ClientStatsBatch) -> bool:
    """Report whether *batch* is within the sizes a client may journal."""
    if len(batch.track_stats) > MAX_TRACKS:
        return False
    readings = [stat.metrics for stat in batch.track_stats]
    if batch.connection_stat is not None:
        readings.append(batch.connection_stat.metrics)
    if any(len(metrics) > MAX_METRICS for metrics in readings):
        return False
    names = [name for metrics in readings for name in metrics]
    names += [field for stat in batch.track_stats for field in (stat.track_name, stat.codec)]
    return all(len(name) <= MAX_NAME_LENGTH for name in names)


def to_detail(batch: ClientStatsBatch) -> dict[str, Any]:
    """Shape *batch* as the readings part of a ``client_stats`` fact's detail.

    A metric value that isn't a finite number is left out, since JSON has no
    way to carry it.
    """
    return {
        "track_stats": [
            {
                "timestamp": stat.timestamp,
                "track_name": stat.track_name,
                "kind": stat.kind,
                "direction": stat.direction,
                "codec": stat.codec,
                "paused": stat.paused,
                "metrics": _finite(stat.metrics),
            }
            for stat in batch.track_stats
        ],
        "connection_stat": (
            {
                "timestamp": batch.connection_stat.timestamp,
                "metrics": _finite(batch.connection_stat.metrics),
            }
            if batch.connection_stat is not None
            else None
        ),
    }


def _finite(metrics: Mapping[str, float]) -> dict[str, float]:
    """Keep the metric values JSON can carry: finite numbers, not NaN or infinity."""
    return {key: value for key, value in metrics.items() if math.isfinite(value)}
