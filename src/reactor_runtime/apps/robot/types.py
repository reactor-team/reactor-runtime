"""What a robot policy reads and returns: :class:`Observation` and :class:`Actions`."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Observation:
    """The camera frames and robot state of one capture, as ``generate()`` reads them.

    Every frame and the state share one capture stamp, taken on the robot's
    clock when it captured them. The mappings are read-only copies, so holding
    an observation past ``generate()`` is safe; the frame arrays themselves are
    not copied.

    Attributes:
        step: The number of the step this observation is for, counted from 0
            in the session.
        capture_time_us: The capture stamp, in microseconds on the robot's
            monotonic clock. It identifies the capture: the actions computed
            from this observation carry the same stamp back to the robot.
        frames: One RGB frame per camera, keyed by track name, each an
            ``(height, width, 3)`` ``uint8`` array.
        state: The robot state sent with the capture, as the JSON object the
            robot sent. Empty when the capture carried no state.
        task: The task the policy is asked to perform, or ``None`` when no task
            is set.
        skew_us: Per camera, the stamp of its frame minus ``capture_time_us``,
            in microseconds. Zero for a frame captured with the state; a camera
            missing from the mapping has no measured skew.
        repeated: The cameras whose frame is the last one seen again, because
            no new frame arrived in time for this capture. Empty by default.
        missing: The cameras that have no frame in this observation, and so no
            key in ``frames``. Empty by default.
    """

    step: int
    capture_time_us: int
    frames: Mapping[str, np.ndarray]
    state: Mapping[str, Any] = field(default_factory=dict)
    task: str | None = None
    skew_us: Mapping[str, int] = field(default_factory=dict)
    repeated: frozenset[str] = frozenset()
    missing: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Keep read-only copies of the mappings and sets the caller passed."""
        object.__setattr__(self, "frames", MappingProxyType(dict(self.frames)))
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))
        object.__setattr__(self, "skew_us", MappingProxyType(dict(self.skew_us)))
        object.__setattr__(self, "repeated", frozenset(self.repeated))
        object.__setattr__(self, "missing", frozenset(self.missing))


@dataclass(frozen=True, eq=False, init=False)
class Actions:
    """A chunk of actions a policy returns from one observation.

    The chunk is a table: one row per action, played ``dt_ms`` apart, starting
    at the observation's capture time, and one column per actuator value.

    Attributes:
        chunk: The actions, as a 2-D array or a sequence of rows of numbers.
            Stored as a read-only ``(rows, columns)`` ``float64`` copy, so the
            caller may reuse its buffer once the constructor returns. At least
            one row and one column.
        dt_ms: Milliseconds between two consecutive rows. A finite number
            greater than zero.
        names: The name of each column, in order, or ``None`` to leave the
            columns unnamed. One distinct name per column.
        extra: Anything else the policy returns with the chunk, as a JSON
            object for the robot. Empty by default.

    Raises:
        ValueError: The rows of ``chunk`` differ in length, ``chunk`` is not
            two-dimensional or is empty, ``dt_ms`` is not a finite number
            greater than zero, or ``names`` does not hold one distinct name
            per column.
        TypeError: ``chunk`` holds something other than numbers, ``dt_ms`` is
            not a number, or a name is not a string.
    """

    chunk: np.ndarray
    dt_ms: float
    names: tuple[str, ...] | None
    extra: Mapping[str, Any]

    def __init__(
        self,
        chunk: np.ndarray | Sequence[Sequence[float]],
        dt_ms: float,
        names: Sequence[str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate the chunk, its period and its column names, and keep copies of them."""
        object.__setattr__(self, "chunk", _as_chunk(chunk))
        object.__setattr__(self, "dt_ms", _as_period(dt_ms))
        object.__setattr__(self, "names", _as_names(names, self.chunk.shape[1]))
        object.__setattr__(self, "extra", MappingProxyType(dict(extra or {})))

    @property
    def rows(self) -> int:
        """The number of actions in the chunk."""
        return int(self.chunk.shape[0])

    @property
    def columns(self) -> int:
        """The number of values in each action."""
        return int(self.chunk.shape[1])


def _as_chunk(chunk: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return *chunk* as a read-only 2-D ``float64`` copy, refusing ragged or empty tables."""
    if not isinstance(chunk, np.ndarray):
        try:
            lengths = sorted({len(row) for row in chunk})
        except TypeError as exc:
            raise TypeError(
                f"an action chunk is a sequence of rows of numbers, got {chunk!r}"
            ) from exc
        if len(lengths) > 1:
            raise ValueError(
                f"every row of an action chunk must have the same length, got lengths {lengths}"
            )
    try:
        array = np.array(chunk, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"an action chunk holds numbers only: {exc}") from exc
    if array.ndim != 2:
        raise ValueError(
            f"an action chunk is two-dimensional (rows, columns), got shape {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"an action chunk has at least one row and one column, got shape {array.shape}"
        )
    array.flags.writeable = False
    return array


def _as_period(dt_ms: float) -> float:
    """Return *dt_ms* as a float, refusing a period that is not finite and positive."""
    if isinstance(dt_ms, bool) or not isinstance(dt_ms, int | float):
        raise TypeError(f"dt_ms must be a number, got {type(dt_ms).__name__}")
    period = float(dt_ms)
    if not math.isfinite(period) or period <= 0:
        raise ValueError(f"dt_ms must be a finite number greater than zero, got {dt_ms}")
    return period


def _as_names(names: Sequence[str] | None, columns: int) -> tuple[str, ...] | None:
    """Return *names* as a tuple, refusing one that does not name each column once."""
    if names is None:
        return None
    if isinstance(names, str):
        raise TypeError("names is a sequence of column names, got a single string")
    named = tuple(names)
    for name in named:
        if not isinstance(name, str):
            raise TypeError(f"a column name must be a string, got {type(name).__name__}")
    if len(named) != columns:
        raise ValueError(
            f"names must hold one name per column: the chunk has {columns} columns, "
            f"got {len(named)} names"
        )
    if len(set(named)) != len(named):
        raise ValueError(f"column names must be distinct, got {list(named)}")
    return named
