"""A DistributedRunner drives one worker in its own process and never leaves a caller waiting.

These tests spawn real processes and cross real shared memory. No torch and no
GPU: the worker below is plain Python, so what is tested is the protocol around
the model, not the model.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime.distributed import (
    DistributedRunner,
    DistributedWorker,
    WorkerCrashed,
    WorkerTimeout,
)


@dataclass(frozen=True)
class CounterInput:
    step: int
    frames: np.ndarray


@dataclass(frozen=True)
class CounterResult:
    value: int
    calls: int
    frames: np.ndarray
    rank: int
    world_size: int
    device: str
    environment: dict[str, str]
    loaded_with: dict[str, str]


class Exhausted(Exception):  # noqa: N818 (a worker's own error, named for its state)
    """The worker's own error: the count reached its limit."""


class Counter(DistributedWorker):
    """Adds a base to the step and echoes the frames. Raises past a limit."""

    def load(
        self,
        *,
        base: int,
        limit: int = 100,
        config_path: Path | None = None,
        weights_root: Path | None = None,
    ) -> None:  # ty: ignore[invalid-method-override]  # every worker narrows the base's **kwargs
        self.base = base
        self.limit = limit
        self.calls = 0
        self.loaded_with = {
            "config_path": str(config_path),
            "weights_root": str(weights_root),
        }

    def generate(self, input: CounterInput, /) -> CounterResult:
        if input.step >= self.limit:
            raise Exhausted(input.step)
        self.calls += 1
        return CounterResult(
            value=self.base + input.step,
            calls=self.calls,
            frames=input.frames + 1,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            environment={
                k: os.environ[k] for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
            },
            loaded_with=self.loaded_with,
        )

    def reset(self) -> None:
        self.calls = 0


class SelfDestruct(Counter):
    """Dies mid-step, the way a segfault or an OOM kill would."""

    def generate(self, input: CounterInput, /) -> CounterResult:
        if input.step == 99:
            os.kill(os.getpid(), signal.SIGKILL)
        return super().generate(input)


class Sleeper(Counter):
    """Never answers a step in time."""

    def generate(self, input: CounterInput, /) -> CounterResult:
        time.sleep(5.0)
        return super().generate(input)


class RefusesToLoad(Counter):
    def load(self, **kwargs: Any) -> None:
        raise FileNotFoundError("weights.safetensors")


class NoReset:
    def load(self, **kwargs: Any) -> None: ...

    def generate(self, input: Any, /) -> Any: ...


def _input(step: int) -> CounterInput:
    return CounterInput(step=step, frames=np.full((2, 4, 4, 3), step, dtype=np.uint8))


@pytest.fixture
def runner() -> Iterator[DistributedRunner]:
    runner = DistributedRunner(
        Counter, load_kwargs={"base": 10}, call_timeout=20.0, start_timeout=60.0
    )
    runner.start()
    yield runner
    runner.shutdown()


def test_a_worker_answers_from_its_own_process(runner: DistributedRunner) -> None:
    result = runner.generate(_input(3))

    assert result.value == 13
    assert result.calls == 1
    np.testing.assert_array_equal(result.frames, np.full((2, 4, 4, 3), 4, dtype=np.uint8))
    assert (result.rank, result.world_size, result.device) == (0, 1, "cpu")
    assert result.environment["RANK"] == "0"
    assert result.environment["WORLD_SIZE"] == "1"
    assert result.environment["MASTER_ADDR"] == "127.0.0.1"
    assert int(result.environment["MASTER_PORT"]) > 0
    assert "RANK" not in os.environ  # set in the child only


def test_reset_reaches_the_worker(runner: DistributedRunner) -> None:
    runner.generate(_input(1))
    runner.generate(_input(2))
    runner.reset()
    assert runner.generate(_input(3)).calls == 1


def test_load_kwargs_arrive_as_keyword_arguments(tmp_path: Path) -> None:
    runner = DistributedRunner(
        Counter,
        load_kwargs={"base": 0, "config_path": tmp_path / "config.yml", "weights_root": tmp_path},
        start_timeout=60.0,
    )
    runner.start()
    try:
        loaded_with = runner.generate(_input(0)).loaded_with
    finally:
        runner.shutdown()
    assert loaded_with == {
        "config_path": str(tmp_path / "config.yml"),
        "weights_root": str(tmp_path),
    }


def test_the_workers_own_error_reaches_the_caller_unchanged(runner: DistributedRunner) -> None:
    with pytest.raises(Exhausted) as raised:
        runner.generate(_input(100))
    assert raised.value.args == (100,)

    # The group is intact: the caller recovers the way the design says, with reset().
    assert runner.healthy
    runner.reset()
    assert runner.generate(_input(1)).value == 11


def test_a_dead_rank_surfaces_within_seconds_not_after_the_timeout() -> None:
    runner = DistributedRunner(
        SelfDestruct, load_kwargs={"base": 0}, call_timeout=600.0, start_timeout=60.0
    )
    runner.start()
    try:
        started = time.monotonic()
        with pytest.raises(WorkerCrashed, match="rank 0 exited"):
            runner.generate(_input(99))
        assert time.monotonic() - started < 30.0
        assert not runner.healthy
        with pytest.raises(RuntimeError, match="unusable"):
            runner.generate(_input(1))
    finally:
        runner.shutdown()


def test_a_wedged_rank_hits_the_call_timeout_and_is_killed_on_shutdown() -> None:
    runner = DistributedRunner(
        Sleeper, load_kwargs={"base": 0}, call_timeout=0.5, start_timeout=60.0
    )
    runner.start()
    procs = list(runner._procs)
    try:
        with pytest.raises(WorkerTimeout, match="generate"):
            runner.generate(_input(1))
        assert not runner.healthy
    finally:
        runner.shutdown()
    assert all(not proc.is_alive() for proc in procs)


def test_a_load_failure_is_raised_from_start_and_leaves_no_process() -> None:
    runner = DistributedRunner(RefusesToLoad, load_kwargs={"base": 0}, start_timeout=60.0)
    with pytest.raises(FileNotFoundError, match=r"weights\.safetensors"):
        runner.start()
    assert all(not proc.is_alive() for proc in runner._procs)
    runner.shutdown()  # idempotent after the failed start


def test_calls_before_start_and_a_second_start_are_refused(runner: DistributedRunner) -> None:
    fresh = DistributedRunner(Counter, load_kwargs={"base": 0})
    with pytest.raises(RuntimeError, match="start"):
        fresh.generate(_input(0))
    with pytest.raises(RuntimeError, match="starts once"):
        runner.start()


def test_shutdown_is_idempotent_and_ends_the_process(runner: DistributedRunner) -> None:
    procs = list(runner._procs)
    runner.shutdown()
    runner.shutdown()
    assert all(not proc.is_alive() for proc in procs)
    with pytest.raises(RuntimeError, match="unusable"):
        runner.generate(_input(0))


def test_unpicklable_load_kwargs_are_refused_before_any_process_starts() -> None:
    with pytest.raises(TypeError, match=r"load_kwargs\['handle'\]"):
        DistributedRunner(Counter, load_kwargs={"handle": open(os.devnull)})  # noqa: SIM115


@pytest.mark.parametrize("count", [0, -1, 1.0, "2"])
def test_world_size_must_be_a_positive_integer(count: Any) -> None:
    with pytest.raises(ValueError, match="world_size"):
        DistributedRunner(Counter, world_size=count)


def test_a_class_without_the_three_methods_is_refused_by_name() -> None:
    with pytest.raises(TypeError, match="reset"):
        DistributedRunner(NoReset)


def test_a_plain_class_with_the_three_methods_is_accepted() -> None:
    class Plain:
        def load(self, **kwargs: Any) -> None: ...

        def generate(self, input: Any, /) -> Any: ...

        def reset(self) -> None: ...

    DistributedRunner(Plain)  # no base class needed; the shape is what counts
