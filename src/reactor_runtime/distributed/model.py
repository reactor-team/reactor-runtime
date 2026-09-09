# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Experimental single-session video adapter on the ordinary model lifecycle."""

from __future__ import annotations

import asyncio
import copy
import math
import pickle
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from reactor_runtime.core.model import (
    ClientDisconnected,
    ReactorEvent,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.distributed.group import WorkerGroup
from reactor_runtime.distributed.worker import DistributedWorker
from reactor_runtime.interface.model.reactor_model import ReactorModel
from reactor_runtime.interface.tracks import Output


class DistributedVideoModel(ReactorModel):
    """Declare per-rank computation without managing worker sessions or threads.

    Experimental, single-node, single-session uint8 video only. Declare
    ``worker`` and ``frame_shape``, implement :meth:`to_output`, and optionally
    :meth:`controls`. Commands and lifecycle hooks use the ordinary decorators.
    The manifest supplies ``world_size``; one worker runs in-process.

    The adapter owns ``load`` and ``run``. Use :meth:`worker_setup` to read model
    configuration and pass weight paths to workers; use :meth:`session_params`
    for initialization data. Every hook returning a dictionary is snapshotted
    before dispatch, and must contain picklable CPU data. GPU state belongs to
    the worker, not the controller.

    ``paused`` holds the current worker session, including one completed
    in-flight chunk. :meth:`restart_generation` instead drops that chunk and
    reinitializes the workers at the next safe boundary. Neither operation
    interrupts a collective. Last-viewer disconnects also restart generation;
    additional viewers share the existing sequence.

    Attributes:
        worker: Importable per-rank worker class.
        frame_shape: Maximum ``(frames, height, width, channels)`` buffer shape.
        session_seed: Initial seed, in NumPy's uint32 range.
        adaptive_fps: Measure chunk throughput for playout; otherwise use ``fps``.
        startup_timeout: Multi-worker setup/warmup deadline in seconds.
        command_timeout: Multi-worker session/chunk deadline in seconds.
        shutdown_timeout: Grace period before terminating worker processes.
        init_process_group: Initialize NCCL/gloo; disable for CPU protocol tests.
    """

    worker: ClassVar[type[DistributedWorker] | None] = None
    frame_shape: ClassVar[tuple[int, ...] | None] = None
    session_seed: int = 0
    adaptive_fps: ClassVar[bool] = False
    startup_timeout: ClassVar[float] = 3600.0
    command_timeout: ClassVar[float] = 300.0
    shutdown_timeout: ClassVar[float] = 60.0
    init_process_group: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self._workers: WorkerGroup | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._epoch = 0
        self._active = False
        self._paused = False
        self._wakeup: asyncio.Event | None = None

    def load(self, config_path: Path | None) -> None:
        """Validate declarations, then load every rank on its owning thread."""
        if self._workers is not None:
            raise RuntimeError("load() must be called only once")
        if not isinstance(self.worker, type) or not issubclass(self.worker, DistributedWorker):
            raise TypeError("declare worker = YourDistributedWorker on the model")
        if self.frame_shape is None:
            raise TypeError("declare frame_shape = (max_frames, height, width, channels)")
        if type(self).to_output is DistributedVideoModel.to_output:
            raise TypeError("implement to_output(self, frames) to return your typed Output")
        if type(self.session_seed) is not int or not 0 <= self.session_seed < 2**32:
            raise ValueError("session_seed must be an integer in [0, 2**32)")
        for name in ("startup_timeout", "command_timeout", "shutdown_timeout"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number of seconds")
        setup = _snapshot(self.worker_setup(config_path), "worker_setup()")
        self._workers = WorkerGroup(
            self.worker,
            frame_shape=self.frame_shape,
            world_size=self.world_size,
            setup_kwargs=setup,
            init_process_group=self.init_process_group,
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-workers")
        try:
            self._executor.submit(self._workers.start, timeout=self.startup_timeout).result()
        except BaseException:
            self._executor.shutdown(wait=True)
            raise

    def worker_setup(self, config_path: Path | None) -> dict[str, Any]:
        """Read model configuration and return kwargs for every worker's setup."""
        return {}

    def session_params(self) -> dict[str, Any]:
        """Return worker initialization data for a new sequence or restart."""
        return {}

    def controls(self) -> dict[str, Any]:
        """Return the current conditioning, snapshotted at each chunk boundary."""
        return {}

    def to_output(self, frames: np.ndarray) -> Output:
        """Map a completed video chunk into the model's declared output track."""
        raise NotImplementedError

    @property
    def paused(self) -> bool:
        """Whether generation is paused, preserving worker state for resume."""
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("paused must be a bool")
        if value and not self._paused:
            self.output.flush()
        self._paused = value
        self._wake()

    def restart_generation(self, *, seed: int | None = None) -> None:
        """Discard pending output and begin a fresh sequence after in-flight work.

        Call from a command or lifecycle hook. Controls are preserved, and a
        paused model remains paused. An optional seed also applies to subsequent
        sequences; it must fit NumPy's uint32 range.
        """
        if seed is not None:
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("seed must be an integer in [0, 2**32)")
            self.session_seed = seed
        self._epoch += 1
        self.output.flush()
        self._wake()

    def _on_loop_ready(self) -> None:
        super()._on_loop_ready()
        self._wakeup = asyncio.Event()

    async def _dispatch_reactor_event(self, event: ReactorEvent) -> None:
        # Bookkeeping is independent of user-decorated hooks: overriding a hook
        # cannot accidentally bypass stale-result protection or worker teardown.
        if isinstance(event, SessionStarted):
            self._active = False
            self._paused = False
            self.restart_generation()
        elif isinstance(event, SessionEnded):
            self._active = False
            self.restart_generation()
        elif isinstance(event, ClientDisconnected) and event.total == 0:
            self.restart_generation()
        await super()._dispatch_reactor_event(event)
        if isinstance(event, SessionStarted):
            self._active = True
        self._wake()

    async def run(self) -> None:
        """Drive serial worker calls, holding or discarding output at safe boundaries."""
        workers = self._workers
        if workers is None:
            raise RuntimeError("load() must run first; override worker_setup(), not load()")
        try:
            while True:
                await self._wait_ready()
                epoch = self._epoch
                params = _snapshot(self.session_params(), "session_params()")
                await self._call(
                    workers.start_session,
                    params,
                    seed=self.session_seed,
                    timeout=self.command_timeout,
                )
                index = 0
                while self._current(epoch):
                    await self._wait_ready(epoch)
                    if not self._current(epoch):
                        break
                    controls = _snapshot(self.controls(), "controls()")
                    started = time.perf_counter()
                    frames = await self._call(
                        workers.generate,
                        index,
                        controls,
                        timeout=self.command_timeout,
                    )
                    compute_time = time.perf_counter() - started
                    await self._wait_ready(epoch)
                    if not self._current(epoch):
                        break
                    output = self.to_output(frames)
                    if not isinstance(output, Output):
                        raise TypeError("to_output() must return an Output instance")
                    await self.emit(
                        output,
                        compute_time=compute_time if self.adaptive_fps else None,
                    )
                    index += 1
                await self._call(workers.end_session, timeout=self.command_timeout)
        finally:
            try:
                await self._call(workers.shutdown, timeout=self.shutdown_timeout)
            finally:
                if self._executor is not None:
                    self._executor.shutdown(wait=False)

    def _current(self, epoch: int) -> bool:
        return epoch == self._epoch and self._active and self.connected.is_set()

    def _wake(self) -> None:
        if self._wakeup is not None:
            self._wakeup.set()

    async def _wait_ready(self, epoch: int | None = None) -> None:
        if self._wakeup is None:
            raise RuntimeError("run() must be started by the runtime's model loop")
        while True:
            if epoch is not None and not self._current(epoch):
                return
            if self._active and self.connected.is_set() and not self._paused:
                return
            self._wakeup.clear()
            await self._wakeup.wait()

    async def _call[T](self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        assert self._executor is not None
        future = asyncio.get_running_loop().run_in_executor(
            self._executor,
            partial(fn, *args, **kwargs),
        )
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Cancellation cannot stop a running GPU hook. Drain the command
            # before shutdown releases its buffers, even in the inline backend.
            try:
                await future
            finally:
                raise


def _snapshot(value: dict[str, Any], hook: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{hook} must return a dict, got {type(value).__name__}")
    try:
        result = copy.deepcopy(value)
        pickle.dumps(result)
    except Exception as exc:
        raise TypeError(
            f"{hook} must return picklable CPU data; pass paths and scalars, not live resources"
        ) from exc
    return result
