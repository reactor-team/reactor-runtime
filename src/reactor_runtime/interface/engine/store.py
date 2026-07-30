"""The per-session input queue behind the window rule — :class:`InputStore`.

Everything a client sends lands here and is stamped on arrival: a command
handler pushes the :class:`UserInput` it built, and a media frame goes into its
track's accumulator. Each step drains the store once, and what comes out is the
window — every input received since the previous drain, in one list, ordered by
arrival.

The store is written from the runtime's thread (media frames) and from the model
loop (command handlers and the drain), so every operation takes the same lock.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from reactor_runtime.core.values import InputFrame
from reactor_runtime.engine_contract.inputs import MediaInput, UserInput


def _now_ms() -> int:
    """Return the current instant on the monotonic millisecond clock inputs are stamped with."""
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class MediaSpec:
    """How one declared :class:`MediaInput` is materialized from its track.

    Attributes:
        track: The wire name of the track frames arrive on.
        input_cls: The declared class each materialized instance is built from.
        chunk_size: How many frames one instance carries.
    """

    track: str
    input_cls: type[MediaInput]
    chunk_size: int


@dataclass(frozen=True)
class _Arrival:
    """One inbound frame with when, and in what order, it reached the runtime.

    The millisecond clock is what the mapping reads, and a burst of input lands
    inside one of its ticks; the sequence number is the arrival order the window
    is actually sorted by, so a tie on the clock cannot reorder anything.
    """

    frame: InputFrame
    at_ms: int
    seq: int


class InputStore:
    """The ordered window of client input, per session.

    Args:
        media: The media specs to accumulate frames against, by track name. A
            store built without any accepts events only.
    """

    def __init__(self, media: dict[str, MediaSpec] | None = None) -> None:
        self._lock = threading.Lock()
        self._media = dict(media or {})
        self._events: list[tuple[int, UserInput]] = []
        self._deferred: dict[int, list[UserInput]] = {}
        self._frames: dict[str, deque[_Arrival]] = {track: deque() for track in self._media}
        self._seq = 0
        self._window_opened_seq = 0
        self._window_opened_ms = _now_ms()

    def push(self, event: UserInput, *, at_step: int | None = None) -> None:
        """Queue an input for the next window, or for a later step's.

        Args:
            event: The input to queue. It is stamped here, so a caller never
                sets ``timestamp_ms`` itself.
            at_step: Hold the input until the window opening at this step index,
                rather than delivering it in the next one. It is stamped at that
                window's opening instant, so it sorts ahead of the inputs that
                arrive live during the step.
        """
        with self._lock:
            if at_step is None:
                event.timestamp_ms = _now_ms()
                self._events.append((self._next_seq(), event))
            else:
                self._deferred.setdefault(at_step, []).append(event)

    def push_frame(self, track: str, frame: InputFrame) -> None:
        """Record an inbound media frame against its track.

        A frame for a track the engine did not declare is dropped.

        Args:
            track: The track the frame arrived on.
            frame: The decoded frame.
        """
        with self._lock:
            pending = self._frames.get(track)
            if pending is not None:
                pending.append(_Arrival(frame, _now_ms(), self._next_seq()))

    def drain(self, step_index: int) -> list[UserInput]:
        """Close the current window and return it, ordered by arrival.

        Media frames are materialized here: a track accumulates until it has a
        full chunk, so an incomplete batch waits for a later window rather than
        arriving short.

        Args:
            step_index: The step this window conditions, which selects the
                inputs deferred to it.

        Returns:
            Every input received since the previous drain, in arrival order.
        """
        with self._lock:
            window: list[tuple[int, UserInput]] = []
            for event in self._deferred.pop(step_index, []):
                event.timestamp_ms = self._window_opened_ms
                window.append((self._window_opened_seq, event))
            window.extend(self._events)
            self._events = []
            for spec in self._media.values():
                window.extend(self._materialize(spec))
            self._window_opened_ms = _now_ms()
            self._window_opened_seq = self._seq
        window.sort(key=lambda entry: entry[0])
        return [event for _, event in window]

    def clear_deferred(self) -> None:
        """Drop every input held for a future step.

        A new rollout restarts the step index, so inputs scheduled against the
        old one no longer address anything.
        """
        with self._lock:
            self._deferred.clear()

    def reset(self) -> None:
        """Drop everything queued, for a fresh session."""
        with self._lock:
            self._events = []
            self._deferred.clear()
            for pending in self._frames.values():
                pending.clear()
            self._window_opened_ms = _now_ms()
            self._window_opened_seq = self._seq

    def _next_seq(self) -> int:
        """Return the next arrival number. Caller holds the lock."""
        self._seq += 1
        return self._seq

    def _materialize(self, spec: MediaSpec) -> list[tuple[int, MediaInput]]:
        """Build one instance per complete chunk waiting on a track. Caller holds the lock."""
        pending = self._frames[spec.track]
        built: list[tuple[int, MediaInput]] = []
        while len(pending) >= spec.chunk_size:
            chunk = [pending.popleft() for _ in range(spec.chunk_size)]
            instance = spec.input_cls()
            instance.data = _payload(chunk)
            instance.pts = chunk[0].frame.pts
            # A batch is complete only once its last frame lands, so that is
            # where it belongs in the window's order.
            instance.timestamp_ms = chunk[-1].at_ms
            built.append((chunk[-1].seq, instance))
        return built


def _payload(chunk: list[_Arrival]) -> Any:
    """Return a chunk's payload: the frame itself, or the batch stacked in capture order."""
    if len(chunk) == 1:
        return chunk[0].frame.data
    return np.stack([arrival.frame.data for arrival in chunk])
