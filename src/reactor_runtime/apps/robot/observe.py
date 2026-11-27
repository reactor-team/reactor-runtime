"""The robot state a client sends stamped with its capture: the ``observe`` message.

A robot sends its state on the data channel as the ``observe`` command::

    {"type": "observe", "data": {"capture_time_us": 1712345678901, "state": {"joints": [...]}}}

``capture_time_us`` is the stamp of the capture the state belongs to, on the
robot's monotonic clock, the same stamp its camera frames carry. The
:class:`ObserveBuffer` keeps the newest of these, in stamp order, until the
camera frames of the same capture arrive.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

OBSERVE_BUFFER_SIZE = 8
"""How many stamped states a session keeps by default: a quarter of a second at 30 Hz."""

INVALID_OBSERVE = "invalid_observe"
"""The error code an ``observe`` message is refused with when it is malformed."""


class ObserveError(ValueError):
    """An ``observe`` message that is malformed, with the reason for the client."""


@dataclass(frozen=True)
class StampedState:
    """One robot state, with the stamp of the capture it belongs to.

    Attributes:
        capture_time_us: The capture stamp, in microseconds on the robot's
            monotonic clock. Zero or more.
        state: The robot state, as the JSON object the robot sent. A read-only
            copy.
        received_at: When the runtime received it, in seconds on its own
            monotonic clock.
    """

    capture_time_us: int
    state: Mapping[str, Any]
    received_at: float

    def __post_init__(self) -> None:
        """Keep a read-only copy of the state the caller passed."""
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))


@dataclass(frozen=True)
class ObserveStats:
    """What a session's ``observe`` messages came to.

    Attributes:
        accepted: Messages taken into the buffer.
        refused: Messages refused as malformed.
        evicted: Accepted states dropped to make room for newer ones, including
            one older than everything a full buffer holds, dropped on arrival.
        buffered: States in the buffer now.
    """

    accepted: int = 0
    refused: int = 0
    evicted: int = 0
    buffered: int = 0


def parse_observe(capture_time_us: Any, state: Any) -> tuple[int, dict[str, Any]]:
    """Check the fields of an ``observe`` message and return them in their Python form.

    An integral float stamp is taken as an integer, since the wire has no
    integer type.

    Args:
        capture_time_us: The ``capture_time_us`` field as it arrived, ``None``
            when it is missing.
        state: The ``state`` field as it arrived, ``None`` when it is missing.

    Returns:
        The stamp as an ``int`` and the state as a ``dict``.

    Raises:
        ObserveError: The stamp is missing, not an integer, or negative, or
            the state is missing or not a JSON object.
    """
    if capture_time_us is None:
        raise ObserveError(
            "observe needs capture_time_us, the capture stamp in microseconds on the robot's clock"
        )
    # is_integer() is false for inf and nan, so only a finite stamp is narrowed.
    if isinstance(capture_time_us, float) and capture_time_us.is_integer():
        capture_time_us = int(capture_time_us)
    if isinstance(capture_time_us, bool) or not isinstance(capture_time_us, int):
        raise ObserveError(
            "observe capture_time_us must be an integer number of microseconds, "
            f"got {capture_time_us!r}"
        )
    if capture_time_us < 0:
        raise ObserveError(f"observe capture_time_us must be zero or more, got {capture_time_us}")
    if state is None:
        raise ObserveError("observe needs state, the robot state as a JSON object")
    if not isinstance(state, Mapping):
        raise ObserveError(f"observe state must be a JSON object, got {_json_kind(state)}")
    for key in state:
        if not isinstance(key, str):
            raise ObserveError(f"observe state keys must be strings, got {key!r}")
    return capture_time_us, dict(state)


class ObserveBuffer:
    """The newest stamped robot states of one session, ordered by capture stamp.

    Holds at most ``size`` states. A state whose stamp is already held replaces
    the one held. When the buffer is full, the oldest stamp is dropped; a state
    older than everything a full buffer holds is dropped on arrival. Both count
    as evicted.

    Not thread-safe: it lives on the model's event loop and is touched only
    there.

    Args:
        size: How many states to keep. At least 1. Defaults to
            :data:`OBSERVE_BUFFER_SIZE`.

    Raises:
        ValueError: ``size`` is less than 1.
    """

    def __init__(self, size: int = OBSERVE_BUFFER_SIZE) -> None:
        """Create an empty buffer of *size* states."""
        if size < 1:
            raise ValueError(f"an observe buffer holds at least 1 state, got {size}")
        self._size = size
        self._stamps: list[int] = []
        self._states: list[StampedState] = []
        self._accepted = 0
        self._refused = 0
        self._evicted = 0

    def __len__(self) -> int:
        """Return how many states the buffer holds."""
        return len(self._states)

    @property
    def size(self) -> int:
        """The most states the buffer keeps."""
        return self._size

    def push(self, entry: StampedState) -> None:
        """Take *entry* in, in stamp order, dropping the oldest when full."""
        self._accepted += 1
        at = bisect.bisect_left(self._stamps, entry.capture_time_us)
        if at < len(self._stamps) and self._stamps[at] == entry.capture_time_us:
            self._states[at] = entry
            return
        if len(self._states) >= self._size:
            if at == 0:
                self._evicted += 1
                return
            del self._stamps[0], self._states[0]
            self._evicted += 1
            at -= 1
        self._stamps.insert(at, entry.capture_time_us)
        self._states.insert(at, entry)

    def refuse(self) -> None:
        """Count one malformed ``observe`` message."""
        self._refused += 1

    def entries(self) -> tuple[StampedState, ...]:
        """Return the states held, oldest stamp first."""
        return tuple(self._states)

    def stats(self) -> ObserveStats:
        """Return the counts since the buffer was created or last reset."""
        return ObserveStats(
            accepted=self._accepted,
            refused=self._refused,
            evicted=self._evicted,
            buffered=len(self._states),
        )

    def clear(self) -> None:
        """Drop every state, keeping the counts."""
        self._stamps.clear()
        self._states.clear()

    def reset(self) -> None:
        """Drop every state and zero the counts."""
        self.clear()
        self._accepted = 0
        self._refused = 0
        self._evicted = 0


def _json_kind(value: Any) -> str:
    """Name *value*'s JSON kind for a refusal message."""
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list | tuple):
        return "an array"
    return type(value).__name__
