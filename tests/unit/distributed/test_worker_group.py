"""Protocol tests for :mod:`reactor_runtime.distributed`.

Runs the real spawn/queue/shared-memory machinery with 2 CPU worker
processes (``init_process_group=False`` — no torch needed). Covers the
lifecycle, per-rank slice writes, retryable session-init errors,
fail-fast chunk errors, and silent-crash detection.

The process group itself is out of scope here: forming one needs torch,
and forming an NCCL one needs GPUs. So what these tests leave untested
is the collective machinery, not the protocol around it — every failure
path below is pinned down without a single collective.
"""

from __future__ import annotations

import os
import queue as queue_mod
import signal
import time
from typing import Any

import numpy as np
import pytest

from reactor_runtime.distributed import (
    DistributedWorker,
    WorkerCrashed,
    WorkerError,
    WorkerGroup,
)
from reactor_runtime.distributed.protocol import Reply

FRAME_SHAPE = (4, 8, 6, 3)  # (frames, H, W, C); H=8 splits across 2 ranks


class BandWorker(DistributedWorker):
    """Each rank writes its horizontal band: pixel value = rank marker +
    chunk index, so the controller can assert exactly who wrote what."""

    # A concrete override necessarily narrows the base's ``**setup_kwargs``,
    # which a strict checker reads as an LSP violation. That is a property of
    # the abstraction — every model's worker declares its own kwargs — not of
    # this test, so it is suppressed here the same way a model would.
    def setup(self, *, base: int) -> None:  # ty: ignore[invalid-method-override]
        self.base = base
        self.band_h = FRAME_SHAPE[1] // self.world_size
        self.row0 = self.rank * self.band_h

    def start_session(self, params: dict[str, Any]) -> None:
        if params.get("explode"):
            raise ValueError("bad session params")
        # explode_rank: fail on ONE rank's first attempt only — the
        # init-consensus regression (a lone failure must fail the group,
        # and the retry must find every rank alive and aligned).
        self._init_attempts = getattr(self, "_init_attempts", 0) + 1
        if params.get("explode_rank") == self.rank and self._init_attempts == 1:
            raise ValueError(f"rank {self.rank} boom")
        self.session_open = True

    def generate_chunk(self, index: int, controls: dict[str, Any]) -> int:
        if controls.get("explode"):
            raise RuntimeError("chunk failure")
        if controls.get("explode_rank") == self.rank:
            raise RuntimeError("chunk failure")
        n = FRAME_SHAPE[0]
        value = self.base + 10 * self.rank + index
        band = np.full((n, self.band_h, *FRAME_SHAPE[2:]), value, dtype=np.uint8)
        self.frames.array[:n, self.row0 : self.row0 + self.band_h] = band
        return n  # end row: every rank wrote all n frames of its band

    def end_session(self) -> None:
        self.session_open = False


class LeaderReturnWorker(BandWorker):
    """Leader returns a full frame array; other ranks return None."""

    def generate_chunk(self, index: int, controls: dict[str, Any]) -> Any:
        if not self.is_leader:
            return None
        return np.full(FRAME_SHAPE, index + 1, dtype=np.uint8)


class FrameShardWorker(BandWorker):
    """Frames-axis sharding: rank r writes frames [r*k, (r+1)*k) and
    forwards write()'s end-row return — the pattern that used to
    truncate the chunk to rank 0's slice."""

    def generate_chunk(self, index: int, controls: dict[str, Any]) -> int:
        k = FRAME_SHAPE[0] // self.world_size
        shard = np.full((k, *FRAME_SHAPE[1:]), self.rank + 1, dtype=np.uint8)
        return self.frames.write(shard, start_row=self.rank * k)


@pytest.fixture(params=[1, 2])
def group(request):
    wg = WorkerGroup(
        BandWorker,
        frame_shape=FRAME_SHAPE,
        world_size=request.param,
        setup_kwargs={"base": 100},
        init_process_group=False,
    )
    wg.start(timeout=120)
    yield wg
    wg.shutdown()


def test_session_roundtrip_with_per_rank_slice_writes(group: WorkerGroup) -> None:
    group.start_session({}, seed=7)
    for index in range(2):
        frames = group.generate(index, {})
        assert frames.shape == FRAME_SHAPE
        band_h = FRAME_SHAPE[1] // group.world_size
        # rank 0 wrote the top band, rank 1 the bottom band.
        assert (frames[:, :band_h] == 100 + index).all()
        assert (frames[:, band_h:] == 110 + index).all()
    group.end_session()
    # A second session on the same workers must work (reset path).
    group.start_session({}, seed=8)
    assert group.generate(0, {}).shape == FRAME_SHAPE
    group.end_session()


def test_leader_returned_frames_are_written_by_framework() -> None:
    wg = WorkerGroup(
        LeaderReturnWorker,
        frame_shape=FRAME_SHAPE,
        world_size=2,
        setup_kwargs={"base": 0},
        init_process_group=False,
    )
    wg.start(timeout=120)
    try:
        wg.start_session({}, seed=1)
        assert (wg.generate(2, {}) == 3).all()
    finally:
        wg.shutdown()


def test_init_session_error_is_retryable(group: WorkerGroup) -> None:
    with pytest.raises(WorkerError, match="bad session params"):
        group.start_session({"explode": True}, seed=1)
    # Workers stayed alive: the retry with good params succeeds.
    group.start_session({}, seed=1)
    assert group.generate(0, {}).shape == FRAME_SHAPE


def test_invalid_seeds_are_rejected_before_touching_workers(group: WorkerGroup) -> None:
    for seed in (-1, 2**32, True):
        with pytest.raises(ValueError, match="seed must be an integer"):
            group.start_session({}, seed=seed)
    group.start_session({}, seed=7)
    assert group.generate(0, {}).shape == FRAME_SHAPE


def test_partial_chunk_failure_posts_exactly_one_error(group: WorkerGroup) -> None:
    if group.world_size == 1:
        pytest.skip("requires two processes")
    # One rank fails a chunk while its peer succeeds. The failing rank
    # posts its ERROR once (in the chunk handler) and the re-raise must
    # NOT post a second one from the outer handler — a duplicate would
    # linger on the result queue and be misread by a later collect
    # (e.g. an end_session during teardown reporting a stale chunk
    # error instead of the real WorkerCrashed).
    group.start_session({}, seed=1)
    with pytest.raises(WorkerError, match="chunk failure"):
        group.generate(0, {"explode_rank": 1})
    deadline = time.monotonic() + 2.0
    stragglers: list[Any] = []
    while time.monotonic() < deadline:
        try:
            stragglers.append(group._result_queue.get(timeout=0.2))
        except queue_mod.Empty:
            continue
    errors = [m for m in stragglers if m[0] is Reply.ERROR]
    assert not errors, f"duplicate ERROR left on queue: {errors}"


def test_single_rank_init_failure_fails_group_then_retry_aligns(
    group: WorkerGroup,
) -> None:
    if group.world_size == 1:
        pytest.skip("requires two ranks")
    # Rank 1 fails its first attempt while rank 0 succeeds: the group
    # must report failure (one rank's OK is not group success), and the
    # immediate retry must find every rank parked and aligned.
    with pytest.raises(WorkerError, match="rank 1 boom"):
        group.start_session({"explode_rank": 1}, seed=1)
    group.start_session({"explode_rank": 1}, seed=1)  # attempt 2 passes
    frames = group.generate(0, {})
    band_h = FRAME_SHAPE[1] // 2
    assert (frames[:, :band_h] == 100).all()
    assert (frames[:, band_h:] == 110).all()


class StubbornWorker(BandWorker):
    """Ignores the exit grace period — shutdown() must escalate past it."""

    def shutdown(self) -> None:
        time.sleep(60)


def test_shutdown_after_partial_failure_terminates_survivors() -> None:
    # After a fail-fast failure the group is broken: shutdown() must NOT
    # send the clean exit (a survivor would block forever in the exit
    # barrier waiting on its dead peer) — it must terminate survivors
    # and return promptly with every process dead.
    wg = WorkerGroup(
        BandWorker,
        frame_shape=FRAME_SHAPE,
        world_size=2,
        setup_kwargs={"base": 100},
        init_process_group=False,
    )
    wg.start(timeout=120)
    wg.start_session({}, seed=1)
    with pytest.raises(WorkerError, match="chunk failure"):
        wg.generate(0, {"explode_rank": 1})
    t0 = time.monotonic()
    wg.shutdown()
    assert time.monotonic() - t0 < 15.0
    assert all(not p.is_alive() for p in wg._procs)


def test_shutdown_escalates_past_stubborn_worker() -> None:
    wg = WorkerGroup(
        StubbornWorker,
        frame_shape=FRAME_SHAPE,
        world_size=2,
        setup_kwargs={"base": 100},
        init_process_group=False,
    )
    wg.start(timeout=120)
    t0 = time.monotonic()
    wg.shutdown(timeout=2.0)  # grace expires -> TERM -> (KILL)
    assert time.monotonic() - t0 < 20.0
    assert all(not p.is_alive() for p in wg._procs)


def test_frames_axis_sharding_totals_via_end_row_max() -> None:
    wg = WorkerGroup(
        FrameShardWorker,
        frame_shape=FRAME_SHAPE,
        world_size=2,
        setup_kwargs={"base": 0},
        init_process_group=False,
    )
    wg.start(timeout=120)
    try:
        wg.start_session({}, seed=1)
        frames = wg.generate(0, {})
        # Forwarding write()'s end row must yield the FULL chunk, not
        # rank 0's shard: 4 frames, first half from rank 0 (=1), second
        # half from rank 1 (=2).
        k = FRAME_SHAPE[0] // 2
        assert frames.shape == FRAME_SHAPE
        assert (frames[:k] == 1).all()
        assert (frames[k:] == 2).all()
    finally:
        wg.shutdown()


def test_chunk_error_surfaces_and_fails_fast(group: WorkerGroup) -> None:
    group.start_session({}, seed=1)
    with pytest.raises(WorkerError, match="chunk failure"):
        group.generate(0, {"explode": True})


def test_silent_rank_death_raises_worker_crashed(group: WorkerGroup) -> None:
    if group.world_size == 1:
        pytest.skip("requires a child process")
    group.start_session({}, seed=1)
    # Kill the leader: its ack can never arrive, so only liveness
    # polling can surface the failure (within seconds, not the timeout).
    os.kill(group._procs[0].pid, signal.SIGKILL)
    with pytest.raises(WorkerCrashed, match="rank 0"):
        group.generate(0, {})


def test_one_worker_uses_no_processes_or_shared_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("single-worker execution must not allocate process resources")

    monkeypatch.setattr("reactor_runtime.distributed.group.multiprocessing.get_context", forbidden)
    monkeypatch.setattr("reactor_runtime.distributed.group.SharedFrameBuffer", forbidden)
    wg = WorkerGroup(LeaderReturnWorker, frame_shape=FRAME_SHAPE, setup_kwargs={"base": 0})
    before = {key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE", "MASTER_PORT")}
    try:
        wg.start()
        assert wg.world_size == 1
        assert wg._local_worker is not None
        assert wg._local_worker.is_leader
        assert not wg._procs
        wg.start_session({}, seed=7)
        first = wg.generate(0, {})
        wg.generate(1, {})
        assert (first == 1).all()  # The next chunk cannot overwrite returned frames.
        assert before == {key: os.environ.get(key) for key in before}
    finally:
        wg.shutdown()
        wg.shutdown()
    assert wg._local_worker is None


@pytest.mark.parametrize("count", [0, -1, True, 2.5, None])
def test_worker_count_must_be_an_explicit_positive_integer(count) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        WorkerGroup(BandWorker, frame_shape=FRAME_SHAPE, world_size=count)


class SlowWorker(BandWorker):
    def generate_chunk(self, index: int, controls: dict[str, Any]) -> int:
        time.sleep(0.2)
        return super().generate_chunk(index, controls)


def test_a_timeout_cannot_leak_a_reply_into_another_command() -> None:
    wg = WorkerGroup(
        SlowWorker,
        frame_shape=FRAME_SHAPE,
        world_size=2,
        setup_kwargs={"base": 0},
        init_process_group=False,
    )
    wg.start(timeout=120)
    try:
        wg.start_session({}, seed=0)
        with pytest.raises(TimeoutError):
            wg.generate(0, {}, timeout=0.01)
        time.sleep(0.3)  # Both old replies are now eligible to arrive.
        with pytest.raises(WorkerError, match="unusable"):
            wg.generate(1, {})
        with pytest.raises(WorkerError, match="unusable"):
            wg.start_session({}, seed=1)
    finally:
        wg.shutdown()
    assert all(not proc.is_alive() for proc in wg._procs)


def test_group_lifecycle_rejects_out_of_order_calls() -> None:
    wg = WorkerGroup(BandWorker, frame_shape=FRAME_SHAPE, setup_kwargs={"base": 0})
    with pytest.raises(RuntimeError, match="start must complete"):
        wg.generate(0, {})
    wg.start()
    try:
        with pytest.raises(RuntimeError, match="more than once"):
            wg.start()
        with pytest.raises(RuntimeError, match="start_session"):
            wg.generate(0, {})
        wg.start_session({}, seed=0)
        with pytest.raises(RuntimeError, match="current session"):
            wg.start_session({}, seed=1)
    finally:
        wg.shutdown()
    with pytest.raises(WorkerError, match="unusable"):
        wg.end_session()


class StartupFailure(BandWorker):
    def warmup(self) -> None:
        raise RuntimeError("warmup failed")


@pytest.mark.parametrize("world_size", [1, 2])
def test_startup_failure_releases_every_process_and_buffer(world_size: int) -> None:
    wg = WorkerGroup(
        StartupFailure,
        frame_shape=FRAME_SHAPE,
        world_size=world_size,
        setup_kwargs={"base": 0},
        init_process_group=False,
    )
    with pytest.raises((WorkerError, WorkerCrashed)):
        wg.start(timeout=120)
    assert wg._shutdown_done
    assert all(not proc.is_alive() for proc in wg._procs)
    assert wg._frames is not None
    assert wg._frames.array.size == 0


class InvalidRowWorker(BandWorker):
    def generate_chunk(self, index: int, controls: dict[str, Any]) -> int:
        return controls["row"]


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("row", [-1, 5, True])
def test_invalid_end_rows_fail_the_group(world_size: int, row: int) -> None:
    wg = WorkerGroup(
        InvalidRowWorker,
        frame_shape=FRAME_SHAPE,
        world_size=world_size,
        setup_kwargs={"base": 0},
        init_process_group=False,
    )
    wg.start(timeout=120)
    try:
        wg.start_session({}, seed=0)
        with pytest.raises((WorkerError, WorkerCrashed)):
            wg.generate(0, {"row": row})
        with pytest.raises(WorkerError, match="unusable"):
            wg.generate(1, {"row": 1})
    finally:
        wg.shutdown()
