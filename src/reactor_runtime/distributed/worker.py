# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

"""Per-GPU worker base class and process entry point.

Every rank runs the same command loop in lockstep: the controller
broadcasts each :class:`~reactor_runtime.distributed.protocol.Verb` to
all ranks (a rank that missed one would desync the next collective) and
every rank replies to every verb — the controller collects the full
reply set, which is the framework's only synchronization mechanism.
The framework issues no collectives of its own; the process group
belongs entirely to the model.
"""

from __future__ import annotations

import faulthandler
import gc
import logging
import os
import random
import sys
from typing import Any

import numpy as np

from reactor_runtime.distributed.frames import SharedFrameBuffer
from reactor_runtime.distributed.protocol import Reply, Verb


class DistributedWorker:
    """Base class for one GPU's worth of a multi-GPU model.

    Subclass and override the hooks; the framework runs one instance
    per GPU in its own process, with ``self.rank`` / ``self.world_size``
    / ``self.device`` / ``self.frames`` populated and the process group
    initialized before ``setup`` is called. All hooks run on the
    worker's single command loop — no locking needed. Exceptions from
    ``start_session`` are recoverable: all ranks rendezvous on the
    outcome, so every rank stays alive and aligned for the controller's
    retry. Exceptions from ``generate_chunk`` tear all ranks down
    together, never leaving a half-alive NCCL world.
    """

    rank: int
    world_size: int
    #: ``"cuda:<rank>"`` when CUDA is available, else ``"cpu"``.
    device: str
    frames: SharedFrameBuffer

    @property
    def is_leader(self) -> bool:
        """True on rank 0 — the only rank whose acks the controller awaits."""
        return self.rank == 0

    def setup(self, **setup_kwargs: Any) -> None:
        """Load models and process-lifetime resources. Called once."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Prime kernels/graphs before the group reports ready. Runs
        after :meth:`setup`; warmup ordering is often model-sensitive,
        so everything order-dependent belongs here, explicitly."""

    def start_session(self, params: dict[str, Any]) -> None:
        """Initialize per-session state. The framework has already
        seeded RNGs identically on every rank."""
        raise NotImplementedError

    def generate_chunk(self, index: int, controls: dict[str, Any]) -> Any:
        """Produce one chunk of frames; called in lockstep on every rank.

        Return one of: an array of frames (the framework writes it to
        the shared buffer for you), an ``int`` end row — which is
        exactly what ``self.frames.write(...)`` returns, so slice
        writers just forward it — or ``None`` if this rank wrote
        nothing. The controller takes the max end row across all ranks
        as the chunk's frame count, so every write pattern (leader-only
        array, frames-axis sharding, per-rank pixel bands) yields the
        correct total without coordination.
        """
        raise NotImplementedError

    def end_session(self) -> None:
        """Drop per-session state; keep process-lifetime resources."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Release external resources on clean exit (optional)."""


def _torch() -> Any:
    """Lazy torch import — the controller side must stay torch-free."""
    try:
        import torch

        return torch
    except ImportError:
        return None


def worker_main(
    rank: int,
    world_size: int,
    master_port: int,
    worker_cls: type[DistributedWorker],
    setup_kwargs: dict[str, Any],
    cmd_queue: Any,
    result_queue: Any,
    log_level: int,
    frame_shape: tuple[int, ...],
    shm_name: str,
    init_process_group: bool,
) -> None:
    """Entry point every rank runs (spawned by :class:`WorkerGroup`)."""
    # Fresh interpreter: the parent's logging config does not propagate.
    # Non-leader ranks demote to WARNING to avoid N-fold duplicate lines.
    logging.basicConfig(
        level=log_level if rank == 0 else max(log_level, logging.WARNING),
        format=f"[rank {rank}] %(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Torchrun-parity env, set in THIS child process only: env:// rendezvous
    # in init_process_group reads MASTER_ADDR/PORT/RANK/WORLD_SIZE, and model
    # code inside setup() conventionally reads RANK/LOCAL_RANK. Spawned
    # processes get none of these for free the way torchrun-launched ones do.
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    # Crash context: the controller detects death; this says where.
    faulthandler.enable()

    torch = _torch()
    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

    dist: Any = None
    if init_process_group and world_size > 1 and torch is not None:
        import torch.distributed as torch_dist

        dist = torch_dist
        backend = "nccl" if device.startswith("cuda") else "gloo"
        dist.init_process_group(
            backend=backend, init_method="env://", rank=rank, world_size=world_size
        )

    frames = SharedFrameBuffer(frame_shape, name=shm_name)
    frames.pin()

    worker = worker_cls()
    worker.rank = rank
    worker.world_size = world_size
    worker.device = device
    worker.frames = frames

    clean_exit = False
    reported = False  # this rank already posted ERROR for the unwinding failure
    try:
        # Startup inside the try: a warmup OOM must become an
        # ("error", ...) post, not an opaque controller timeout.
        worker.setup(**setup_kwargs)
        worker.warmup()
        # Every rank reports ready; the controller collects all
        # world_size replies, so no verb can arrive before the slowest
        # rank finishes warmup.
        result_queue.put((Reply.READY,))

        while True:
            cmd = cmd_queue.get()
            verb = cmd[0]

            if verb is Verb.EXIT:
                worker.shutdown()
                break

            if verb is Verb.SEED:
                seed = int(cmd[1])
                random.seed(seed)
                np.random.seed(seed)
                if torch is not None:
                    torch.manual_seed(seed)

            elif verb is Verb.INIT_SESSION:
                # Recoverable: this rank reports its own outcome and the
                # controller collects every rank's reply before deciding
                # (or retrying) — so a retry can never race a peer still
                # inside a failed attempt, and one rank's success cannot
                # be misread as group success.
                try:
                    worker.start_session(cmd[1])
                    result_queue.put((Reply.OK,))
                except Exception as exc:  # noqa: BLE001 - reported to controller
                    logger.exception("init_session failed")
                    result_queue.put((Reply.ERROR, f"rank {rank}: init_session: {exc}"))

            elif verb is Verb.CHUNK:
                try:
                    index, controls = cmd[1], cmd[2]
                    result = worker.generate_chunk(index, controls)
                    end_row = result if isinstance(result, int) else 0
                    if result is not None and not isinstance(result, int):
                        end_row = frames.write(result)
                    # This reply happens-after this rank's buffer writes;
                    # the controller reads frames only once every rank
                    # has replied, so all slices have landed by then.
                    result_queue.put((Reply.FRAMES, int(end_row)))
                except Exception as exc:  # noqa: BLE001 - reported, then fail-fast
                    # Re-raise so all ranks exit together: a mid-chunk
                    # failure usually means a peer is wedged in the
                    # interrupted collective, and NCCL's async error
                    # handling will surface it there.
                    logger.exception("chunk %s failed", cmd[1])
                    result_queue.put(
                        (Reply.ERROR, f"rank {rank}: chunk {cmd[1]}: {exc}")
                    )
                    reported = True
                    raise

            elif verb is Verb.DROP_SESSION:
                worker.end_session()
                gc.collect()
                if torch is not None and device.startswith("cuda"):
                    torch.cuda.empty_cache()
                result_queue.put((Reply.OK,))

            else:
                logger.warning("unknown verb %r", verb)

        clean_exit = True
    except Exception as exc:  # noqa: BLE001 - reported to controller
        # Startup failures land here. Chunk failures already posted their
        # ERROR before re-raising — posting again would leave a straggler
        # on the queue for a later _collect to misread (e.g. an
        # end_session during teardown reporting a stale chunk error).
        if not reported:
            result_queue.put((Reply.ERROR, f"rank {rank}: {exc}"))
        raise
    finally:
        if dist is not None and dist.is_initialized():
            # Clean exit: all ranks are alive, so barrier before destroy.
            # Error path: a peer may be dead — a barrier would deadlock;
            # go straight to the (local) destroy.
            if clean_exit:
                dist.barrier()
            dist.destroy_process_group()
        frames.close()
