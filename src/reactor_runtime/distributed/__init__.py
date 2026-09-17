"""Run a model in its own process, or in one process per GPU, behind one object.

A worker is a class with ``load()``, ``generate()``, and ``reset()``. A runner
starts N of them, one per GPU, and lets the caller drive all of them with one
call. The caller writes no process, queue, shared-memory, or NCCL code.

This package is standalone. Nothing else in the runtime depends on it, and it
reads no manifest and no environment variable of its own.
"""

from reactor_runtime.distributed.errors import (
    RankDesync,
    SharedSlotAllocationFailed,
    WorkerCrashed,
    WorkerTimeout,
)

__all__ = [
    "RankDesync",
    "SharedSlotAllocationFailed",
    "WorkerCrashed",
    "WorkerTimeout",
]
