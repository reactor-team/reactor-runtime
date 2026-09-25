"""Drive one worker per GPU from one object the caller owns.

The caller constructs a :class:`DistributedRunner`, calls :meth:`start`, and
then calls :meth:`generate` and :meth:`reset` on it as if it were talking to
one object. Every call broadcasts one request to every rank and waits for one
answer from every rank. One call is outstanding at a time.

Every wait has a way out: it polls the processes while it waits, so a dead
rank surfaces as :class:`~reactor_runtime.distributed.WorkerCrashed` within
seconds, and it has a timeout, so a wedged rank surfaces as
:class:`~reactor_runtime.distributed.WorkerTimeout`.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import multiprocessing
import os
import pickle
import queue
import signal
import socket
import time
from typing import Any

from reactor_runtime.distributed.errors import RankDesync, WorkerCrashed, WorkerTimeout
from reactor_runtime.distributed.ipc import (
    SharedSlot,
    SlotReader,
    new_prefix,
    pack,
    unlink_blocks,
    unpack,
)
from reactor_runtime.distributed.protocol import (
    Answer,
    Generate,
    Join,
    Loaded,
    Rendezvous,
    Reset,
    Shutdown,
)
from reactor_runtime.distributed.rank import rank_main
from reactor_runtime.distributed.worker import check_shape
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_LIVENESS_POLL = 5.0
_JOIN_GRACE = 5.0
# Ranks leave on Shutdown together, so this is how long the slowest one may take
# to release its GPU memory before it is terminated.
_SHUTDOWN_GRACE = 30.0


class DistributedRunner:
    """Start N ranks of one worker class and drive them in lockstep.

    Every rank receives the same requests in the same order, so the
    collectives inside a sharded model line up. Rank 0's result is the one
    returned; ranks 1 to N-1 answer with success or an error and no payload.

    Args:
        worker_cls: A class with ``load()``, ``generate()``, and ``reset()``
            and a no-argument constructor. Constructed in each rank's process,
            never in the caller's.
        world_size: How many ranks to start, one per GPU. Explicit; never
            read from the machine or a manifest.
        load_kwargs: Keyword arguments for ``load()``. They cross a process
            boundary, so paths and scalars, not live objects.
        call_timeout: Seconds a ``generate()`` or ``reset()`` may take before
            it is treated as a hang.
        start_timeout: Seconds ``start()`` waits for every rank to load.
        init_process_group: Form the ``torch.distributed`` group in each rank
            when ``world_size > 1``. Leave on; off is for protocol tests that
            run ranks without torch.
    """

    def __init__(
        self,
        worker_cls: type,
        *,
        world_size: int = 1,
        load_kwargs: dict[str, Any] | None = None,
        call_timeout: float = 30.0,
        start_timeout: float = 3600.0,
        init_process_group: bool = True,
    ) -> None:
        check_shape(worker_cls)
        _crosses_spawn(worker_cls)
        if type(world_size) is not int or world_size < 1:
            raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
        self._load_kwargs = _picklable(load_kwargs or {})
        self._worker_cls = worker_cls
        self._world_size = world_size
        self._call_timeout = call_timeout
        self._start_timeout = start_timeout
        self._init_process_group = init_process_group
        self._ctx = multiprocessing.get_context("spawn")
        self._inboxes: list[Any] = []
        self._outbox: Any = None
        self._procs: list[Any] = []
        self._input_slot: SharedSlot | None = None
        self._result_prefix = new_prefix()
        self._reader = SlotReader()
        self._started = False
        self._healthy = True
        self._shutdown_done = False

    @property
    def world_size(self) -> int:
        """How many ranks this runner drives."""
        return self._world_size

    @property
    def healthy(self) -> bool:
        """Whether the group accepts calls. ``False`` after a crash, a timeout, or a desync."""
        return self._healthy and self._started and not self._shutdown_done

    def start(self) -> None:
        """Spawn every rank and block until each has finished ``load()``.

        Raises:
            Exception: What a rank's ``load()`` raised, unchanged.
            WorkerCrashed: A rank died before it reported ready.
            WorkerTimeout: A rank did not report ready within ``start_timeout``.
        """
        if self._started or self._shutdown_done:
            raise RuntimeError("a DistributedRunner starts once; construct a new one to restart")
        self._started = True
        atexit.register(self.shutdown)
        try:
            self._inboxes = [self._ctx.Queue() for _ in range(self._world_size)]
            self._outbox = self._ctx.Queue()
            self._input_slot = SharedSlot()
            forms_group = self._init_process_group and self._world_size > 1
            # When the runner forms the group, rank 0 binds its own port instead.
            port = None if forms_group else _free_port()
            level = logging.getLogger().getEffectiveLevel()
            for rank in range(self._world_size):
                proc = self._ctx.Process(
                    target=rank_main,
                    args=(
                        self._worker_cls,
                        rank,
                        self._world_size,
                        port,
                        self._load_kwargs,
                        self._inboxes[rank],
                        self._outbox,
                        self._result_prefix,
                        level,
                        self._init_process_group,
                    ),
                    # A daemonic process may not start children, which a DataLoader
                    # with workers needs. shutdown() is registered with atexit instead.
                    daemon=False,
                )
                proc.start()
                self._procs.append(proc)
            deadline = time.monotonic() + self._start_timeout
            if forms_group:
                self._rendezvous(self._start_timeout)
            remaining = max(0.0, deadline - time.monotonic())
            for message in self._collect("start", remaining):
                if isinstance(message, Answer) and message.error is not None:
                    raise message.error
                if not isinstance(message, Loaded):
                    raise RuntimeError(f"unexpected message during start: {message!r}")
        except BaseException:
            self._healthy = False
            self.shutdown()
            raise
        logger.info("runner ready", world_size=self._world_size)

    def generate(self, input: Any, /) -> Any:
        """Run one ``generate()`` on every rank and return rank 0's result.

        Blocks until every rank has answered.

        Raises:
            Exception: Rank 0's exception, unchanged, when every rank raised.
                The group stays healthy.
            RankDesync: Some ranks raised and others did not. The group refuses
                further calls.
            WorkerCrashed: A rank died during the call.
            WorkerTimeout: A rank did not answer within ``call_timeout``.
        """
        self._require_ready()
        assert self._input_slot is not None
        request = Generate(pack(input, self._input_slot))
        leader = self._decide(self._round_trip(request, "generate"), "generate")
        return unpack(leader.header, self._reader) if leader.header is not None else None

    def reset(self) -> None:
        """Call ``reset()`` on every rank and return when each has answered."""
        self._require_ready()
        self._decide(self._round_trip(Reset(), "reset"), "reset")

    def shutdown(self) -> None:
        """End every rank. Idempotent, and registered with ``atexit``.

        A healthy group is sent ``Shutdown`` and joined. An unhealthy group is
        not: a survivor that received it would wait in the exit barrier for a
        dead peer. Any rank alive after its grace period is terminated, then
        killed. Each rank leads its own process group and is signalled as a
        group, so the processes its worker started end with it, and anything
        still in a rank's group afterwards is killed. Either way this ends with
        dead processes and every shared block unlinked, including the result
        blocks of a rank that was killed before it could close them.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        atexit.unregister(self.shutdown)
        try:
            if self._healthy and self._procs:
                self._broadcast(Shutdown())
                deadline = time.monotonic() + _SHUTDOWN_GRACE
                for proc in self._procs:
                    proc.join(timeout=max(0.0, deadline - time.monotonic()))
            for proc in self._procs:
                if proc.is_alive():
                    logger.warning("terminating a rank that outlived shutdown", pid=proc.pid)
                    _signal_group(proc, signal.SIGTERM)
                    proc.terminate()
                    proc.join(timeout=_JOIN_GRACE)
                if proc.is_alive():
                    _signal_group(proc, signal.SIGKILL)
                    proc.kill()
                    proc.join(timeout=_JOIN_GRACE)
            for proc in self._procs:
                _signal_group(proc, signal.SIGKILL)
        except Exception as exc:
            logger.error("rank teardown failed", error=str(exc))
        finally:
            if self._input_slot is not None:
                self._input_slot.close()
                self._input_slot = None
            self._reader.close()
            if self._procs:
                unlink_blocks(self._result_prefix)
            for channel in [*self._inboxes, self._outbox]:
                if channel is not None:
                    # A dead rank never drains its queue; joining the feeder would hang.
                    channel.cancel_join_thread()
                    channel.close()

    def _rendezvous(self, timeout: float) -> None:
        """Wait for the port rank 0 bound for the process group, and send it to the others."""
        port = None
        for message in self._collect("start", timeout, count=1):
            if isinstance(message, Answer) and message.error is not None:
                raise message.error
            if isinstance(message, Rendezvous):
                port = message.port
        if port is None:
            raise RuntimeError("start: a rank answered before rank 0 reported its port")
        for inbox in self._inboxes[1:]:
            inbox.put(Join(port))

    def _round_trip(self, request: Generate | Reset, what: str) -> list[Any]:
        """Broadcast one request and collect every rank's answer to it.

        Anything that interrupts this, a ``KeyboardInterrupt`` above all, leaves
        requests in flight whose answers the next call would take for its own.
        The group is marked unusable instead.
        """
        try:
            self._broadcast(request)
            return self._collect(what, self._call_timeout)
        except BaseException:
            self._healthy = False
            raise

    def _broadcast(self, request: Generate | Reset | Shutdown) -> None:
        for inbox in self._inboxes:
            inbox.put(request)

    def _collect(self, what: str, timeout: float, count: int | None = None) -> list[Any]:
        """Wait for *count* messages, one from every rank by default, polling liveness."""
        deadline = time.monotonic() + timeout
        messages: list[Any] = []
        while len(messages) < (count or self._world_size):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._healthy = False
                raise WorkerTimeout(f"{what}: no answer from every rank within {timeout}s")
            try:
                messages.append(self._outbox.get(timeout=min(remaining, _LIVENESS_POLL)))
            except queue.Empty:
                self._check_alive(what, messages)
        return messages

    def _decide(self, answers: list[Any], what: str) -> Answer:
        """Apply the outcome policy to one call's answers and return rank 0's."""
        answers.sort(key=lambda answer: answer.rank)
        failed = [answer for answer in answers if answer.error is not None]
        if not failed:
            return answers[0]
        if len(failed) == len(answers) and len({type(a.error) for a in failed}) == 1:
            # Every rank raised the same type of error on the same input: the
            # model's own deterministic raise. The group is intact and the
            # caller may recover. The messages are not compared, because a
            # model may name its rank in them.
            raise failed[0].error
        self._healthy = False
        first = failed[0]
        others = len(answers) - len(failed)
        reason = (
            f"{others} rank(s) did not"
            if others
            else "other ranks raised something else: "
            + ", ".join(f"rank {a.rank} {type(a.error).__name__}" for a in failed[1:])
        )
        raise RankDesync(
            f"{what}: rank {first.rank} raised {type(first.error).__name__} while {reason}; "
            "the group refuses further calls"
        ) from first.error

    def _check_alive(self, what: str, messages: list[Any]) -> None:
        """Raise ``WorkerCrashed`` for a dead rank that has not answered this call.

        A rank that answered and then exited (a ``load()`` that raised, say)
        has its answer on the outbox, not a crash; that answer is drained here
        first so the caller receives it instead of ``WorkerCrashed``.
        """
        dead = [rank for rank, proc in enumerate(self._procs) if not proc.is_alive()]
        if not dead:
            return
        while True:
            try:
                messages.append(self._outbox.get_nowait())
            except queue.Empty:
                break
        answered = {getattr(message, "rank", None) for message in messages}
        for rank in dead:
            if rank not in answered:
                self._healthy = False
                raise WorkerCrashed(
                    f"{what}: rank {rank} exited with code {self._procs[rank].exitcode}; "
                    "its stderr has the faulthandler trace"
                )

    def _require_ready(self) -> None:
        if not self._started:
            raise RuntimeError("start() has not run")
        if self._shutdown_done or not self._healthy:
            raise RuntimeError("the runner is unusable; shut it down and construct a new one")


def _crosses_spawn(worker_cls: type) -> None:
    """Raise ``TypeError`` if *worker_cls* cannot be pickled by reference into a rank."""
    try:
        pickle.dumps(worker_cls)
    except Exception as exc:
        raise TypeError(
            f"{worker_cls.__qualname__} does not pickle, so it cannot reach the worker "
            "processes. Define it at the top level of an importable module, not inside "
            f"a function or a notebook cell: {exc}"
        ) from exc


def _picklable(load_kwargs: dict[str, Any]) -> dict[str, Any]:
    for key, value in load_kwargs.items():
        try:
            pickle.dumps(value)
        except Exception as exc:
            raise TypeError(
                f"load_kwargs[{key!r}] does not pickle, so it cannot reach the worker: {exc}"
            ) from exc
    return dict(load_kwargs)


def _signal_group(proc: Any, sig: int) -> None:
    """Send *sig* to the process group *proc* leads: the rank and what its worker started."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
