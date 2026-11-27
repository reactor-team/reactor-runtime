"""Observation and Actions: what a robot policy reads and returns."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from reactor_runtime.apps.robot import Actions, Observation


def _observation(**overrides: Any) -> Observation:
    values: dict[str, Any] = {
        "step": 3,
        "capture_time_us": 1_250_000,
        "frames": {"scene": np.zeros((4, 4, 3), dtype=np.uint8)},
        "state": {"joints": [0.1, 0.2]},
    }
    values.update(overrides)
    return Observation(**values)


# -- Observation ---------------------------------------------------------------


def test_an_observation_defaults_to_no_task_no_skew_and_every_camera_fresh() -> None:
    obs = _observation()

    assert obs.step == 3
    assert obs.capture_time_us == 1_250_000
    assert set(obs.frames) == {"scene"}
    assert obs.state == {"joints": [0.1, 0.2]}
    assert obs.task is None
    assert obs.skew_us == {}
    assert obs.repeated == frozenset()
    assert obs.missing == frozenset()


def test_an_observation_cannot_be_changed() -> None:
    obs = _observation()

    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.step = 4  # type: ignore[ty:invalid-assignment] (the assignment under test)
    with pytest.raises(TypeError):
        obs.frames["wrist"] = np.zeros((1, 1, 3), dtype=np.uint8)  # type: ignore[ty:invalid-assignment] (the assignment under test)
    with pytest.raises(TypeError):
        obs.state["gripper"] = 1.0  # type: ignore[ty:invalid-assignment] (the assignment under test)


def test_an_observation_keeps_copies_of_what_it_was_given() -> None:
    frames = {"scene": np.zeros((2, 2, 3), dtype=np.uint8)}
    state: dict[str, Any] = {"joints": [0.0]}
    skew = {"scene": 0}
    repeated = {"wrist"}
    obs = _observation(frames=frames, state=state, skew_us=skew, repeated=repeated)

    frames["wrist"] = np.zeros((2, 2, 3), dtype=np.uint8)
    state["gripper"] = 1.0
    skew["wrist"] = 5
    repeated.add("scene")

    assert set(obs.frames) == {"scene"}
    assert set(obs.state) == {"joints"}
    assert obs.skew_us == {"scene": 0}
    assert obs.repeated == frozenset({"wrist"})


def test_an_observation_carries_the_task_skew_and_the_late_cameras() -> None:
    obs = _observation(
        task="stack the cups",
        skew_us={"scene": 0, "wrist": -1_200},
        repeated=["wrist"],
        missing=["top"],
    )

    assert obs.task == "stack the cups"
    assert obs.skew_us == {"scene": 0, "wrist": -1_200}
    assert obs.repeated == frozenset({"wrist"})
    assert obs.missing == frozenset({"top"})


# -- Actions -------------------------------------------------------------------


def test_actions_keep_the_chunk_as_a_read_only_float_table() -> None:
    actions = Actions([[1, 2], [3, 4], [5, 6]], dt_ms=20)

    assert actions.chunk.dtype == np.float64
    assert actions.chunk.shape == (3, 2)
    assert (actions.rows, actions.columns) == (3, 2)
    assert actions.dt_ms == 20.0
    assert isinstance(actions.dt_ms, float)
    assert actions.names is None
    assert actions.extra == {}
    with pytest.raises(ValueError, match="read-only"):
        actions.chunk[0, 0] = 9.0


def test_actions_copy_the_callers_buffer() -> None:
    buffer = np.ones((2, 3), dtype=np.float32)
    actions = Actions(buffer, dt_ms=10.0)

    buffer[:] = 7.0

    assert np.array_equal(actions.chunk, np.ones((2, 3)))
    assert buffer.flags.writeable


def test_actions_cannot_be_changed() -> None:
    actions = Actions([[0.0]], dt_ms=10.0, extra={"gripper": "open"})

    with pytest.raises(dataclasses.FrozenInstanceError):
        actions.dt_ms = 5.0  # type: ignore[ty:invalid-assignment] (the assignment under test)
    with pytest.raises(TypeError):
        actions.extra["gripper"] = "closed"  # type: ignore[ty:invalid-assignment] (the assignment under test)


def test_actions_name_each_column() -> None:
    actions = Actions([[0.1, 0.2, 0.3]], dt_ms=33.3, names=["x", "y", "gripper"])

    assert actions.names == ("x", "y", "gripper")


@pytest.mark.parametrize("dt_ms", [0, 0.0, -1.0, math.inf, math.nan])
def test_actions_refuse_a_period_that_is_not_greater_than_zero(dt_ms: float) -> None:
    with pytest.raises(ValueError, match="dt_ms must be a finite number greater than zero"):
        Actions([[0.0]], dt_ms=dt_ms)


@pytest.mark.parametrize("dt_ms", [True, "20", None])
def test_actions_refuse_a_period_that_is_not_a_number(dt_ms: Any) -> None:
    with pytest.raises(TypeError, match="dt_ms must be a number"):
        Actions([[0.0]], dt_ms=dt_ms)


def test_actions_refuse_rows_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="same length, got lengths \\[2, 3\\]"):
        Actions([[0.0, 1.0], [0.0, 1.0, 2.0]], dt_ms=10.0)


@pytest.mark.parametrize(
    "chunk",
    [np.zeros(3), np.zeros((2, 2, 2)), np.zeros((0, 3)), np.zeros((3, 0)), [], [[]]],
)
def test_actions_refuse_a_chunk_that_is_not_a_non_empty_table(chunk: Any) -> None:
    with pytest.raises(ValueError, match="action chunk"):
        Actions(chunk, dt_ms=10.0)


@pytest.mark.parametrize("chunk", [[1.0, 2.0], [["a", "b"]], [[object()]]])
def test_actions_refuse_a_chunk_that_is_not_rows_of_numbers(chunk: Any) -> None:
    with pytest.raises(TypeError, match="action chunk"):
        Actions(chunk, dt_ms=10.0)


def test_actions_refuse_names_that_do_not_match_the_columns() -> None:
    with pytest.raises(ValueError, match="2 columns, got 3 names"):
        Actions([[0.0, 1.0]], dt_ms=10.0, names=["x", "y", "z"])
    with pytest.raises(ValueError, match="2 columns, got 1 names"):
        Actions([[0.0, 1.0]], dt_ms=10.0, names=["x"])


def test_actions_refuse_a_name_used_twice() -> None:
    with pytest.raises(ValueError, match="distinct"):
        Actions([[0.0, 1.0]], dt_ms=10.0, names=["x", "x"])


@pytest.mark.parametrize("names", ["xy", ["x", 1]])
def test_actions_refuse_names_that_are_not_strings(names: Any) -> None:
    with pytest.raises(TypeError):
        Actions([[0.0, 1.0]], dt_ms=10.0, names=names)
