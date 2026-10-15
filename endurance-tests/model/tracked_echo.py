"""examples/echo with a fixed burst and every output batch tracked.

The burst is fixed by ``ENDURANCE_BURST`` instead of left to the client, so
every emit is a clip-sized batch: a retained frame then keeps a whole batch
alive, which the probe sees and RSS shows. Each batch is held only by a weak
reference here, so tracking it changes nothing about when it is freed.

What is counted is *how* each batch is freed. A batch nothing holds any more
is freed the moment its last reference goes. A batch that is only freed while
the cyclic garbage collector runs was held by reference-cycle garbage until
then, which is how a closed connection's pacer kept model output alive before
reactor-runtime #276. That count does not depend on when a collection
happens, or on who triggers it.
"""

from __future__ import annotations

import gc
import os
import threading
import weakref

import numpy as np

from echo import Echo, EchoOutput
from reactor_runtime import StepOutcome, TrackPayload

BURST = int(os.environ.get("ENDURANCE_BURST", "30"))

# Re-entrant: a collection can start on a thread that already holds it (an
# allocation inside the critical section), and its callbacks take it again.
_lock = threading.RLock()
# Keyed by the reference's id: a weakref hashes as its referent, and an
# ndarray is unhashable.
_batches: dict[int, weakref.ref[np.ndarray]] = {}
_freed_by_gc = 0
# The thread running a collection, or None. A collection runs its weakref
# callbacks on its own thread, while a batch freed by refcount on another
# thread runs its callback on that thread, so matching the thread keeps an
# ordinary free that lands during a collection from being counted.
_collecting_on: int | None = None


def _on_gc(phase: str, _info: dict[str, int]) -> None:
    global _collecting_on
    _collecting_on = threading.get_ident() if phase == "start" else None


gc.callbacks.append(_on_gc)


def _on_freed(ref: weakref.ref[np.ndarray]) -> None:
    global _freed_by_gc
    with _lock:
        _batches.pop(id(ref), None)
        if _collecting_on == threading.get_ident():
            _freed_by_gc += 1


def alive_batches() -> int:
    """How many emitted batches are still alive, from any reference."""
    with _lock:
        return len(_batches)


def track(batch: np.ndarray) -> None:
    """Keep a weak reference to *batch*, counted as freed by a collection or not."""
    with _lock:
        ref = weakref.ref(batch, _on_freed)
        _batches[id(ref)] = ref


def tracked_batches() -> list[np.ndarray]:
    """Return the emitted batches still alive.

    The list holds them strongly, so their ids stay theirs while the caller
    has it; drop it before a collection whose frees should count.
    """
    with _lock:
        return [batch for ref in _batches.values() if (batch := ref()) is not None]


def freed_by_gc() -> int:
    """How many batches, since start, were only freed by a garbage collection."""
    with _lock:
        return _freed_by_gc


class TrackedEcho(Echo):
    """Echo at a fixed burst, recording a weak reference to each batch it emits."""

    async def process_output(self, outcome: StepOutcome) -> EchoOutput | None:
        """Emit as echo does, at the fixed burst, tracking each batch."""
        self.state.burst = BURST
        output = await super().process_output(outcome)
        if output is None:
            return None
        video = output.main_video
        # A batch with metadata comes wrapped in a TrackPayload.
        if isinstance(video, TrackPayload):
            batch = video.data
        elif isinstance(video, np.ndarray):
            batch = video
        else:
            return output
        track(batch)
        return output
