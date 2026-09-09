"""Controller-side handle for a fleet of :class:`DistributedWorker` ranks.

Held by the model in the runtime process. For more than one worker this
process does no GPU work itself. Any :class:`~reactor_runtime.ReactorModel` can
hold one, including a :class:`~reactor_runtime.ReactorPipeline`.

The protocol is uniform: every method broadcasts its verb to all ranks
(so NCCL collectives stay in lockstep) and collects one reply from every
rank — a command succeeds iff all ranks succeed. Collecting the full reply set
is also the framework's only synchronization: a rank's reply
happens-after its shared-memory writes, so the framework never issues
compute-time collectives of its own — the process group belongs to the
model. Every blocking wait polls worker liveness, so a dead rank
surfaces as :class:`WorkerCrashed` within seconds instead of a hung
collective.

:meth:`generate` is synchronous. Use a single dedicated executor thread to
keep an async model loop responsive; pass the measured compute time to
``emit`` for adaptive FPS. Never overlap group calls. Liveness is polled
only while waiting on workers — an
idle-time rank death is detected at the next command; a standing
watchdog is not implemented.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import queue as queue_mod
import socket
import time
from typing import Any

import numpy as np

from reactor_runtime.distributed.errors import WorkerCrashed, WorkerError
from reactor_runtime.distributed.frames import SharedFrameBuffer, _FrameBuffer, _validate_shape
from reactor_runtime.distributed.protocol import Reply, Verb
from reactor_runtime.distributed.worker import DistributedWorker, _seed, _torch, worker_main
from reactor_runtime.log import get_logger

logger = get_logger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class WorkerGroup:
    """Drive one local worker or one :class:`DistributedWorker` process per GPU.

    This experimental handle serves one session with one in-flight command.
    All calls must be serialized on the same thread. One worker runs inline:
    no child process, shared memory, environment changes, or process group.
    Its hooks and RNG seeding affect this process and cannot be preempted by a
    timeout. Multi-worker timeouts are terminal: shut down and replace the group.

    Args:
        worker_cls: the model's :class:`DistributedWorker` subclass.
            Must be importable in a fresh interpreter (module level).
        frame_shape: ``(max_frames_per_chunk, H, W, C)`` — sizes the
            shared uint8 frame buffer; size for the worst-case chunk.
        world_size: worker count, default one. Pass ``self.world_size`` from
            the model to use ``model.resources.gpu.count`` in the manifest.
            Never inferred from visible GPUs; a larger host must not change
            the model's layout.
        setup_kwargs: passed to every worker's ``setup()``. Must be
            picklable — prefer paths and scalars over live objects.
        init_process_group: create the NCCL/gloo process group during
            worker startup. Disable only for tests or single-process
            debugging.
    """

    def __init__(
        self,
        worker_cls: type[DistributedWorker],
        *,
        frame_shape: tuple[int, ...],
        world_size: int = 1,
        setup_kwargs: dict[str, Any] | None = None,
        init_process_group: bool = True,
    ) -> None:
        if type(world_size) is not int or world_size < 1:
            raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
        _validate_shape(frame_shape)
        self.world_size = world_size
        self._worker_cls = worker_cls
        self._frame_shape = tuple(frame_shape)
        self._setup_kwargs = setup_kwargs or {}
        self._init_process_group = init_process_group
        self._ctx = multiprocessing.get_context("spawn") if world_size > 1 else None
        self._cmd_queues = [self._ctx.Queue() for _ in range(world_size)] if self._ctx else []
        self._result_queue: Any = self._ctx.Queue() if self._ctx else None
        self._procs: list[Any] = []
        self._frames: _FrameBuffer | None = None
        self._local_worker: DistributedWorker | None = None
        self._started = False
        self._session_open = False
        self._shutdown_done = False
        # Flipped False on any fail-fast failure (a rank died or is
        # dying). Once broken, shutdown() must not attempt the clean
        # exit: survivors receiving EXIT would block forever in the
        # clean-path barrier waiting on the dead peer.
        self._healthy = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 3600.0) -> None:
        """Set up and warm up all ranks, releasing resources if startup fails."""
        if self._started or self._shutdown_done:
            raise RuntimeError("WorkerGroup cannot be started more than once")
        self._started = True
        atexit.register(self.shutdown)
        try:
            if self.world_size == 1:
                self._frames = _FrameBuffer(self._frame_shape)
                worker = self._worker_cls()
                self._local_worker = worker
                worker.rank, worker.world_size = 0, 1
                torch = _torch()
                worker.device = (
                    "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
                )
                if worker.device.startswith("cuda"):
                    torch.cuda.set_device(0)
                worker.frames = self._frames
                worker.setup(**self._setup_kwargs)
                worker.warmup()
            else:
                self._start_processes(timeout)
        except BaseException as exc:
            self._healthy = False
            self.shutdown()
            if self.world_size == 1 and isinstance(exc, Exception):
                raise WorkerError(f"worker startup failed: rank 0: {exc}") from exc
            raise
        logger.info("WorkerGroup ready", world_size=self.world_size)

    def _start_processes(self, timeout: float) -> None:
        """Spawn all ranks and block until setup + warmup complete.

        Raises :class:`WorkerError` on a reported startup failure,
        :class:`WorkerCrashed` if a rank dies silently, and
        ``TimeoutError`` if warmup exceeds ``timeout``.
        """
        assert self._ctx is not None
        frames = SharedFrameBuffer(self._frame_shape, create=True)
        self._frames = frames
        master_port = _find_free_port()
        log_level = logging.getLogger().getEffectiveLevel()
        for rank in range(self.world_size):
            proc = self._ctx.Process(
                target=worker_main,
                args=(
                    rank,
                    self.world_size,
                    master_port,
                    self._worker_cls,
                    self._setup_kwargs,
                    self._cmd_queues[rank],
                    self._result_queue,
                    log_level,
                    self._frame_shape,
                    frames.name,
                    self._init_process_group,
                ),
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)
        self._collect(Reply.READY, what="worker startup", timeout=timeout)

    def start_session(self, params: dict[str, Any], *, seed: int, timeout: float = 300.0) -> None:
        """Seed all ranks identically, then open a session on each.

        A worker-reported initialization error is retryable after every rank
        reports its outcome. A timeout, dead rank, or protocol failure instead
        makes the group unusable. ``seed`` must fit NumPy's uint32 seed range.
        """
        self._require_ready()
        if self._session_open:
            raise RuntimeError("end the current session before starting another")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self._local_worker is not None:
            try:
                _seed(int(seed))
                self._local_worker.start_session(params)
            except Exception as exc:
                raise WorkerError(f"init_session failed: rank 0: {exc}") from exc
            self._session_open = True
            return
        self._send((Verb.SEED, int(seed)))
        self._send((Verb.INIT_SESSION, params))
        # collect_all: a failed init is retryable, so wait for EVERY
        # rank's reply before raising — no rank can still be inside the
        # failed attempt when the retry is sent (it replied, then parked
        # on its command queue), and no straggler reply is left behind
        # to be misread by a later wait.
        self._collect(Reply.OK, what="init_session", timeout=timeout, collect_all_errors=True)
        self._session_open = True

    def generate(self, index: int, controls: dict[str, Any], timeout: float = 300.0) -> np.ndarray:
        """Run one lockstep ``generate_chunk`` on every rank and return its frames.

        The frames are copied out of the shared buffer, and the copy is what
        releases the buffer for the next chunk.

        The chunk's frame count is the max of the per-rank end rows, so
        it is correct whichever write pattern the worker uses: a
        leader-returned array, frames-axis sharding, or per-rank pixel
        bands. Collecting every rank's reply is also what guarantees all
        shared-memory writes have landed before the frames are read.
        """
        self._require_ready()
        if not self._session_open:
            raise RuntimeError("start_session must succeed before generate")
        assert self._frames is not None
        try:
            if self._local_worker is not None:
                result = self._local_worker.generate_chunk(index, controls)
                end_row = result if isinstance(result, int) else 0
                if result is not None and not isinstance(result, int):
                    end_row = self._frames.write(result)
                end_rows = [end_row]
            else:
                self._send((Verb.CHUNK, index, controls))
                end_rows = self._collect(Reply.FRAMES, what=f"chunk {index}", timeout=timeout)
            # Validate every rank, not only the max: a negative count is also
            # a protocol violation when another rank returned valid frames.
            for row in end_rows:
                if type(row) is not int or not 0 <= row <= self._frame_shape[0]:
                    raise ValueError(f"invalid frame end row: {row!r}")
            return self._frames.read(max(end_rows))
        except (WorkerError, WorkerCrashed, TimeoutError):
            self._healthy = False
            raise
        except Exception as exc:
            self._healthy = False
            raise WorkerError(f"chunk {index} failed: {exc}") from exc

    def end_session(self, timeout: float = 300.0) -> None:
        """Drop per-session state on every rank.

        Process-lifetime resources are kept, which is what makes the next
        session on the same workers start fast.
        """
        self._require_ready()
        try:
            if self._local_worker is not None:
                self._local_worker.end_session()
            else:
                self._send((Verb.DROP_SESSION,))
                self._collect(Reply.OK, what="drop_session", timeout=timeout)
        except Exception as exc:
            self._healthy = False
            if self._local_worker is not None:
                raise WorkerError(f"drop_session failed: rank 0: {exc}") from exc
            raise
        finally:
            self._session_open = False

    def shutdown(self, timeout: float = 60.0) -> None:
        """Stop the group deterministically.

        Idempotent, and also registered via ``atexit`` from :meth:`start`, so
        workers are torn down even if the model never calls this.

        Healthy group: send the exit verb so every rank runs the clean
        barrier+destroy path, then join. Broken group (any prior
        fail-fast failure): skip the clean path entirely — a survivor
        entering the exit barrier would wait forever on its dead peer —
        and go straight to termination. Either way, any rank still alive
        after its grace period is force-terminated, then killed: a
        broken group must end in dead processes, never in a hung
        teardown.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        atexit.unregister(self.shutdown)
        try:
            if self._local_worker is not None:
                self._local_worker.shutdown()
            if self._healthy:
                self._send((Verb.EXIT,))
                deadline = time.monotonic() + timeout
                for proc in self._procs:
                    proc.join(timeout=max(0.0, deadline - time.monotonic()))
            for proc in self._procs:  # escalation ladder: TERM, then KILL
                if proc.is_alive():
                    logger.warning(
                        "terminating worker that outlived its shutdown deadline",
                        pid=proc.pid,
                        healthy=self._healthy,
                    )
                    proc.terminate()
                    proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5.0)
        except Exception as exc:
            # Teardown is best effort by design: a failure here must not stop
            # the finally block below from releasing the shared buffer.
            logger.error("best-effort worker teardown failed", error=str(exc))
        finally:
            if self._frames is not None:
                self._frames.close()
            self._local_worker = None
            self._session_open = False
            for channel in [*self._cmd_queues, self._result_queue]:
                if channel is not None:
                    # Broken ranks may never consume queued commands. Joining
                    # a feeder that is still writing to them would hang exit.
                    channel.cancel_join_thread()
                    channel.close()

    # ------------------------------------------------------------------
    # Protocol internals
    # ------------------------------------------------------------------

    def _send(self, cmd: tuple[Any, ...]) -> None:
        # Broadcast to EVERY rank: a rank that misses a verb desyncs the
        # next collective and wedges the world.
        for cmd_queue in self._cmd_queues:
            cmd_queue.put(cmd)

    def _check_alive(self) -> None:
        for rank, proc in enumerate(self._procs):
            if not proc.is_alive():
                self._healthy = False
                raise WorkerCrashed(
                    f"rank {rank} no longer alive (exitcode={proc.exitcode}); "
                    f"see that rank's stderr for faulthandler output"
                )

    def _collect(
        self,
        expect: Reply,
        *,
        what: str,
        timeout: float,
        collect_all_errors: bool = False,
    ) -> list[Any]:
        """Collect one reply from every rank; return their payloads.

        Raises :class:`WorkerError` if any rank replied ``ERROR``. With
        ``collect_all_errors`` (recoverable verbs: init), the full reply
        set is collected before raising, so every rank is parked on its
        command queue when the caller retries. Without it (fail-fast
        verbs: chunk, drop, startup), the first ``ERROR`` raises
        immediately — a peer may be wedged in the collective the failure
        interrupted and would never reply. Cycles the queue get so a
        dead rank surfaces within seconds rather than after ``timeout``.
        """
        deadline = time.monotonic() + timeout
        payloads: list[Any] = []
        errors: list[str] = []
        while len(payloads) + len(errors) < self.world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._healthy = False
                raise TimeoutError(f"timed out waiting for {what}")
            self._check_alive()
            try:
                msg = self._result_queue.get(timeout=min(remaining, 5.0))
            except queue_mod.Empty:
                continue
            if msg[0] is expect:
                payloads.append(msg[1] if len(msg) > 1 else None)
            elif msg[0] is Reply.ERROR:
                if not collect_all_errors:
                    # Fail-fast ERROR: the reporting rank re-raises and
                    # dies by contract — the group is no longer whole.
                    self._healthy = False
                    raise WorkerError(f"{what} failed: {msg[1]}")
                errors.append(str(msg[1]))
            else:
                self._healthy = False
                raise WorkerError(f"unexpected worker result during {what}: {msg!r}")
        if errors:
            raise WorkerError(f"{what} failed: " + "; ".join(errors))
        return payloads

    def _require_ready(self) -> None:
        if self._shutdown_done or not self._healthy:
            raise WorkerError("WorkerGroup is unusable; shut it down and create a new group")
        if not self._started:
            raise RuntimeError("start must complete before sending commands")
