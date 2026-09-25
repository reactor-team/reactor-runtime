"""Two ranks see the same requests in the same order, and every rank's answer counts.

Two CPU processes without a process group (``init_process_group=False``): the
collectives are the model's business, the protocol around them is the
runner's, and it is the protocol that is pinned down here.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from reactor_runtime.distributed import (
    DistributedRunner,
    DistributedWorker,
    RankDesync,
    WorkerCrashed,
)


@dataclass(frozen=True)
class StepInput:
    step: int
    frames: np.ndarray


@dataclass(frozen=True)
class StepResult:
    rank: int
    world_size: int
    frames: np.ndarray


class Exhausted(Exception):  # noqa: N818 (a worker's own error, named for its state)
    """Raised on every rank for the same input."""


class Replica(DistributedWorker):
    """Every rank does the same work and writes what it did to its own log file."""

    def load(self, *, log_dir: Path) -> None:  # ty: ignore[invalid-method-override]
        self.log = log_dir / f"rank{self.rank}.log"
        self.log.write_text("")

    def generate(self, input: StepInput, /) -> StepResult:
        self._note(f"generate {input.step}")
        if input.step >= 100:
            raise Exhausted(input.step)
        return StepResult(rank=self.rank, world_size=self.world_size, frames=input.frames * 2)

    def reset(self) -> None:
        self._note("reset")

    def _note(self, line: str) -> None:
        with self.log.open("a") as handle:
            handle.write(line + "\n")


class OneRankFails(Replica):
    def generate(self, input: StepInput, /) -> StepResult:
        if input.step == 42 and self.rank == 1:
            raise ValueError("rank 1 only")
        return super().generate(input)


class EachRankFailsDifferently(Replica):
    def generate(self, input: StepInput, /) -> StepResult:
        if input.step == 7:
            raise ValueError("rank 0") if self.rank == 0 else TypeError("rank 1")
        return super().generate(input)


class EachRankNamesItself(Replica):
    def generate(self, input: StepInput, /) -> StepResult:
        if input.step == 7:
            raise Exhausted(f"rank {self.rank}: out of budget")
        return super().generate(input)


class Reduces(Replica):
    """Sums one value per rank over the process group the runner formed."""

    def generate(self, input: StepInput, /) -> tuple[float, int]:  # ty: ignore[invalid-method-override]
        import torch  # ty: ignore[unresolved-import]  # the test skips without it
        import torch.distributed as dist  # ty: ignore[unresolved-import]

        value = torch.tensor([float(self.rank + 1)])
        dist.all_reduce(value)
        return float(value), int(os.environ["MASTER_PORT"])


class OneRankDies(Replica):
    def generate(self, input: StepInput, /) -> StepResult:
        if input.step == 99 and self.rank == 1:
            os.kill(os.getpid(), signal.SIGKILL)
        return super().generate(input)


def _input(step: int) -> StepInput:
    return StepInput(step=step, frames=np.full((2, 4, 4, 3), step, dtype=np.uint8))


def _log(tmp_path: Path, rank: int) -> list[str]:
    return (tmp_path / f"rank{rank}.log").read_text().splitlines()


def _runner(worker_cls: type, tmp_path: Path, **kwargs: float) -> DistributedRunner:
    return DistributedRunner(
        worker_cls,
        world_size=2,
        load_kwargs={"log_dir": tmp_path},
        init_process_group=False,
        start_timeout=60.0,
        call_timeout=kwargs.get("call_timeout", 20.0),
    )


@pytest.fixture
def runner(tmp_path: Path) -> Iterator[DistributedRunner]:
    runner = _runner(Replica, tmp_path)
    runner.start()
    yield runner
    runner.shutdown()


def test_every_rank_runs_every_request_in_the_same_order(
    runner: DistributedRunner, tmp_path: Path
) -> None:
    runner.generate(_input(1))
    runner.reset()
    result = runner.generate(_input(2))

    expected = ["generate 1", "reset", "generate 2"]
    assert _log(tmp_path, 0) == expected
    assert _log(tmp_path, 1) == expected
    assert (result.rank, result.world_size) == (0, 2)
    np.testing.assert_array_equal(result.frames, np.full((2, 4, 4, 3), 4, dtype=np.uint8))


def test_when_every_rank_raises_the_caller_gets_the_error_and_the_group_survives(
    runner: DistributedRunner,
) -> None:
    with pytest.raises(Exhausted):
        runner.generate(_input(100))
    assert runner.healthy
    runner.reset()
    assert runner.generate(_input(1)).rank == 0


def test_when_one_rank_raises_the_call_is_a_desync_and_the_group_refuses_more(
    tmp_path: Path,
) -> None:
    runner = _runner(OneRankFails, tmp_path)
    runner.start()
    procs = list(runner._procs)
    try:
        with pytest.raises(RankDesync, match="rank 1 raised ValueError") as raised:
            runner.generate(_input(42))
        assert isinstance(raised.value.__cause__, ValueError)
        assert not runner.healthy
        with pytest.raises(RuntimeError, match="unusable"):
            runner.generate(_input(1))
    finally:
        started = time.monotonic()
        runner.shutdown()
    assert time.monotonic() - started < 30.0
    assert all(not proc.is_alive() for proc in procs)


def test_the_same_error_naming_each_rank_leaves_the_group_usable(tmp_path: Path) -> None:
    runner = _runner(EachRankNamesItself, tmp_path)
    runner.start()
    try:
        with pytest.raises(Exhausted, match="rank 0: out of budget"):
            runner.generate(_input(7))
        assert runner.healthy
        runner.reset()
        assert runner.generate(_input(1)).rank == 0
    finally:
        runner.shutdown()


def test_every_rank_raising_a_different_error_is_a_desync(tmp_path: Path) -> None:
    runner = _runner(EachRankFailsDifferently, tmp_path)
    runner.start()
    try:
        with pytest.raises(RankDesync, match=r"rank 0 raised ValueError.*rank 1 TypeError"):
            runner.generate(_input(7))
        assert not runner.healthy
    finally:
        runner.shutdown()


def test_ranks_form_the_process_group_on_the_port_rank_0_bound(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    runner = DistributedRunner(
        Reduces,
        world_size=2,
        load_kwargs={"log_dir": tmp_path},
        start_timeout=120.0,
        call_timeout=60.0,
    )
    runner.start()
    try:
        total, port = runner.generate(_input(1))
    finally:
        runner.shutdown()
    assert total == 3.0  # 1 + 2: both ranks reduced over one group
    assert port > 0
    assert "MASTER_PORT" not in os.environ  # chosen in the ranks, never in the caller


def test_a_dead_rank_is_named(tmp_path: Path) -> None:
    runner = _runner(OneRankDies, tmp_path, call_timeout=600.0)
    runner.start()
    try:
        with pytest.raises(WorkerCrashed, match="rank 1 exited"):
            runner.generate(_input(99))
        assert not runner.healthy
    finally:
        runner.shutdown()
    assert all(not proc.is_alive() for proc in runner._procs)


def test_each_rank_is_its_own_process_with_its_own_rank(runner: DistributedRunner) -> None:
    pids = {proc.pid for proc in runner._procs}
    assert len(pids) == 2
    assert os.getpid() not in pids
