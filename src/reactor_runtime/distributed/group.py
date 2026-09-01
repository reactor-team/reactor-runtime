"""Controller-side handle for a fleet of :class:`DistributedWorker` ranks.

Held by the model (created in ``load()``) in the runtime process, which
does no GPU work itself. Any :class:`~reactor_runtime.ReactorModel` can
hold one, including a :class:`~reactor_runtime.ReactorPipeline`.

The protocol is uniform: every method broadcasts its verb to all ranks
(so NCCL collectives stay in lockstep) and collects one reply from every
rank — a command succeeds iff all ranks succeed. Collecting the full reply set
is also the framework's only synchronization: a rank's reply
happens-after its shared-memory writes, so the framework never issues
collectives of its own — the process group belongs entirely to the
model. Every blocking wait polls worker liveness, so a dead rank
surfaces as :class:`WorkerCrashed` within seconds instead of a hung
collective.

:meth:`generate` blocks deliberately: freezing the event loop for the
chunk's compute time is how the runtime derives the stream's dynamic
FPS today. Liveness is polled only while waiting on workers — an
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
from reactor_runtime.distributed.frames import SharedFrameBuffer
from reactor_runtime.distributed.protocol import Reply, Verb
from reactor_runtime.distributed.worker import DistributedWorker, worker_main
from reactor_runtime.log import get_logger

logger = get_logger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class WorkerGroup:
    """Spawn and drive one :class:`DistributedWorker` process per GPU.

    Args:
        worker_cls: the model's :class:`DistributedWorker` subclass.
            Must be importable in a fresh interpreter (module level).
        frame_shape: ``(max_frames_per_chunk, H, W, C)`` — sizes the
            shared uint8 frame buffer; size for the worst-case chunk.
        world_size: worker count. Defaults to the visible CUDA device
            count. Pass it explicitly whenever the model's own parallel
            layout decides the rank count — deriving it from
            ``device_count()`` is wrong the moment the two can differ,
            and the manifest's GPU count is not visible from here.
            When passing it, source the value from the model's config
            rather than a literal in ``load()``, so one setting decides
            both the deployment shape and the group size.
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
        world_size: int | None = None,
        setup_kwargs: dict[str, Any] | None = None,
        init_process_group: bool = True,
    ) -> None:
        if world_size is None:
            # Defaulting the rank count is the one place the controller needs
            # torch. Pass world_size explicitly to keep this side torch-free.
            import torch  # ty: ignore[unresolved-import]

            world_size = torch.cuda.device_count()
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        self.world_size = world_size
        self._worker_cls = worker_cls
        self._frame_shape = tuple(frame_shape)
        self._setup_kwargs = setup_kwargs or {}
        self._init_process_group = init_process_group
        self._ctx = multiprocessing.get_context("spawn")
        self._cmd_queues = [self._ctx.Queue() for _ in range(world_size)]
        self._result_queue = self._ctx.Queue()
        self._procs: list[Any] = []
        self._frames: SharedFrameBuffer | None = None
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
        """Spawn all ranks and block until setup + warmup complete.

        Raises :class:`WorkerError` on a reported startup failure,
        :class:`WorkerCrashed` if a rank dies silently, and
        ``TimeoutError`` if warmup exceeds ``timeout``.
        """
        self._frames = SharedFrameBuffer(self._frame_shape, create=True)
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
                    self._frames.name,
                    self._init_process_group,
                ),
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)
        # Send the exit verb on interpreter shutdown so workers run the
        # clean barrier+destroy teardown even if the pipeline forgets.
        atexit.register(self.shutdown)
        self._collect(Reply.READY, what="worker startup", timeout=timeout)
        logger.info("WorkerGroup ready", world_size=self.world_size)

    def start_session(self, params: dict[str, Any], *, seed: int, timeout: float = 300.0) -> None:
        """Seed all ranks identically, then open a session on each.

        A :class:`WorkerError` raised from here is retryable — every rank
        reports its own outcome and stays alive, so the controller may call
        this again with different params.
        """
        self._drain_stale_results()
        self._send((Verb.SEED, int(seed)))
        self._send((Verb.INIT_SESSION, params))
        # collect_all: a failed init is retryable, so wait for EVERY
        # rank's reply before raising — no rank can still be inside the
        # failed attempt when the retry is sent (it replied, then parked
        # on its command queue), and no straggler reply is left behind
        # to be misread by a later wait.
        self._collect(Reply.OK, what="init_session", timeout=timeout, collect_all_errors=True)

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
        self._send((Verb.CHUNK, index, controls))
        end_rows = self._collect(Reply.FRAMES, what=f"chunk {index}", timeout=timeout)
        assert self._frames is not None
        return self._frames.read(max(int(row or 0) for row in end_rows))

    def end_session(self, timeout: float = 300.0) -> None:
        """Drop per-session state on every rank.

        Process-lifetime resources are kept, which is what makes the next
        session on the same workers start fast.
        """
        self._send((Verb.DROP_SESSION,))
        self._collect(Reply.OK, what="drop_session", timeout=timeout)

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
        try:
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

    def _drain_stale_results(self) -> None:
        # A crashed or reset session can leave an unconsumed result that
        # would otherwise be misread as the ack for the next command.
        while True:
            try:
                stale = self._result_queue.get_nowait()
                logger.warning("drained stale worker result", result=repr(stale))
            except queue_mod.Empty:
                return

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
                logger.warning("unexpected worker result", during=what, result=repr(msg))
        if errors:
            raise WorkerError(f"{what} failed: " + "; ".join(errors))
        return payloads
