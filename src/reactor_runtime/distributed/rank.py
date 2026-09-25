"""What one rank's process does: set up, construct the worker, answer requests.

All of it belongs to the runner. The worker sees a configured process: the
environment a ``torchrun`` launch would set, the current CUDA device, the
process group when there is more than one rank, and the runtime logger. The
rank leads its own process group, so the processes the worker starts end
with it. It
then receives ``load(**load_kwargs)`` once and one call per request until
:class:`~reactor_runtime.distributed.protocol.Shutdown`, or until the runner's
process is gone.
"""

from __future__ import annotations

import faulthandler
import logging
import multiprocessing
import os
import pickle
import queue
import sys
from typing import Any

from reactor_runtime.distributed.ipc import SharedSlot, SlotReader, pack, unpack
from reactor_runtime.distributed.protocol import (
    Answer,
    Generate,
    Join,
    Loaded,
    Rendezvous,
    Reset,
    Shutdown,
)
from reactor_runtime.log import configure as configure_logging
from reactor_runtime.log import get_logger

_PARENT_POLL = 5.0
_HOST = "127.0.0.1"


def rank_main(
    worker_cls: type,
    rank: int,
    world_size: int,
    master_port: int | None,
    load_kwargs: dict[str, Any],
    inbox: Any,
    outbox: Any,
    result_prefix: str,
    log_level: int,
    init_process_group: bool,
) -> None:
    """Run one rank until it is told to shut down.

    Args:
        worker_cls: The class to construct. It needs ``load``, ``generate``,
            and ``reset``.
        rank: This process's rank.
        world_size: How many ranks the runner started.
        master_port: The port set as ``MASTER_PORT`` when the runner does
            not form the process group itself. ``None`` when it does: rank 0
            then binds a port and the others learn it through :class:`Join`.
        load_kwargs: Passed to ``worker.load()`` as keyword arguments.
        inbox: This rank's request queue.
        outbox: The queue every rank answers on.
        result_prefix: Names rank 0's result blocks, so the parent can
            reclaim them if this process is killed before it closes them.
        log_level: The parent's root log level. Ranks other than 0 log at
            WARNING or above, so N ranks do not write N copies of every line.
        init_process_group: Form the ``torch.distributed`` group when
            ``world_size > 1``. Off for protocol tests without torch.
    """
    # Its own process group, so the runner can signal this rank together with
    # every process the worker starts.
    os.setpgid(0, 0)
    configure_logging(
        level=log_level if rank == 0 else max(log_level, logging.WARNING), stream=sys.stderr
    )
    logger = get_logger(__name__)

    # Set in the child only. The parent is not a rank and must not claim to be.
    # Model code reads RANK and LOCAL_RANK; MASTER_PORT is set once it is known.
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = _HOST
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)
    faulthandler.enable()
    parent = multiprocessing.parent_process()

    reader = SlotReader()
    result_slot: SharedSlot | None = None
    group: Any = None
    loaded = False
    clean_exit = False
    try:
        device = _bind_device(rank, world_size)
        if init_process_group and world_size > 1:
            group = _form_group(rank, world_size, device, inbox, outbox, parent)
        worker = worker_cls()
        worker.rank = rank
        worker.world_size = world_size
        worker.device = device
        worker.load(**load_kwargs)
        # Only rank 0's result crosses back. The other ranks answer without a
        # payload, so a model with a piece on each rank gathers onto rank 0 first.
        if rank == 0:
            result_slot = SharedSlot(prefix=result_prefix)
        outbox.put(Loaded(rank))
        loaded = True
        logger.info("rank ready", rank=rank, device=device)

        while True:
            request = _receive(inbox, parent)
            if request is None:
                logger.warning("the runner's process is gone; exiting", rank=rank)
                return
            if isinstance(request, Shutdown):
                break
            if isinstance(request, Generate):
                outbox.put(_generate(worker, request, rank, reader, result_slot, logger))
            elif isinstance(request, Reset):
                outbox.put(_reset(worker, rank, logger))
            else:
                logger.warning("unknown request", rank=rank, request=repr(request))
        clean_exit = True
    except Exception as exc:
        # A failure before Loaded is the parent's to raise. A failure after it
        # was answered on the request that caused it.
        if not loaded:
            outbox.put(Answer(rank, error=portable(exc)))
        raise
    finally:
        if group is not None and group.is_initialized():
            # On a clean exit every rank is alive, so they meet at a barrier
            # before the group is destroyed. On an error a peer may be dead and
            # a barrier would wait forever, so the group is destroyed locally.
            if clean_exit:
                group.barrier()
            group.destroy_process_group()
        if result_slot is not None:
            result_slot.close()
        reader.close()


def portable(exc: Exception) -> Exception:
    """Return *exc* if it pickles, else a ``RuntimeError`` carrying its type and message.

    The parent re-raises what a worker raised. An exception whose arguments do
    not pickle would fail on the queue, so it is replaced by one that does.
    """
    try:
        pickle.loads(pickle.dumps(exc))
    except Exception:
        return RuntimeError(f"{type(exc).__name__}: {exc}")
    return exc


def _generate(
    worker: Any,
    request: Generate,
    rank: int,
    reader: SlotReader,
    result_slot: SharedSlot | None,
    logger: Any,
) -> Answer:
    try:
        result = worker.generate(unpack(request.header, reader))
        header = pack(result, result_slot) if result_slot is not None else None
    except Exception as exc:
        logger.exception("generate failed", rank=rank)
        return Answer(rank, error=portable(exc))
    return Answer(rank, header=header)


def _reset(worker: Any, rank: int, logger: Any) -> Answer:
    try:
        worker.reset()
    except Exception as exc:
        logger.exception("reset failed", rank=rank)
        return Answer(rank, error=portable(exc))
    return Answer(rank)


def _receive(inbox: Any, parent: Any) -> Any:
    """Return the next request, or ``None`` once the runner's process is gone.

    The runner's process may end without shutting the ranks down: killed, or
    exited past its atexit handlers. A rank waiting on its queue checks for
    that between polls.
    """
    while True:
        try:
            return inbox.get(timeout=_PARENT_POLL)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                return None


def _form_group(
    rank: int, world_size: int, device: str, inbox: Any, outbox: Any, parent: Any
) -> Any:
    """Form the process group on a port that rank 0 holds before anyone connects.

    Rank 0 opens the group's store on a port the operating system picks, and
    reports it. The runner sends it to every other rank, which connects there.
    """
    try:
        import torch.distributed as group  # ty: ignore[unresolved-import]  # model image only
    except ImportError as exc:
        raise RuntimeError(
            "world_size > 1 needs torch in the worker's environment to form the process group"
        ) from exc
    if rank == 0:
        store = group.TCPStore(_HOST, 0, world_size, is_master=True, wait_for_workers=False)
        port = int(store.port)
        outbox.put(Rendezvous(port))
    else:
        join = _receive(inbox, parent)
        if not isinstance(join, Join):
            raise RuntimeError(f"expected the rendezvous port before any request, got {join!r}")
        port = join.port
        store = group.TCPStore(_HOST, port, world_size, is_master=False)
    os.environ["MASTER_PORT"] = str(port)
    backend = "nccl" if device.startswith("cuda") else "gloo"
    group.init_process_group(backend, store=store, rank=rank, world_size=world_size)
    return group


def _bind_device(rank: int, world_size: int) -> str:
    """Make ``"cuda"`` mean this rank's GPU, or report ``"cpu"`` when there is none."""
    try:
        import torch  # ty: ignore[unresolved-import]  # installed in the model image only
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    visible = torch.cuda.device_count()
    if world_size > visible:
        raise ValueError(
            f"world_size={world_size} but {visible} CUDA device(s) are visible. Lower world_size "
            "to match the deployment's GPU allocation, or expose the requested GPUs"
        )
    torch.cuda.set_device(rank)
    return f"cuda:{rank}"
