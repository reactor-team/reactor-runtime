# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

"""Multi-GPU worker abstraction for real-time streaming models (REA-3667).

Vends the controller/worker pattern proven by the production multi-GPU
world models as two classes:

- :class:`DistributedWorker` — the per-GPU class a model author
  subclasses. Hooks: ``setup`` / ``warmup`` / ``start_session`` /
  ``generate_chunk`` / ``end_session``.
- :class:`WorkerGroup` — the controller handle a ``ReactorPipeline``
  holds. Owns process spawning, the command/ack protocol, the
  shared-memory frame transport, liveness detection, and teardown.

``torch`` is imported lazily and only where available: the controller
side runs torch-free, and worker processes use it (NCCL process group,
CUDA device binding, host-memory pinning) when the model image provides
it. See the ``multi-gpu-example`` model in reactor-models for the raw
pattern this package absorbs, and the REA-3667 design doc for the
rationale.
"""

from reactor_runtime.distributed.errors import WorkerCrashed, WorkerError
from reactor_runtime.distributed.frames import SharedFrameBuffer
from reactor_runtime.distributed.group import WorkerGroup
from reactor_runtime.distributed.worker import DistributedWorker

__all__ = [
    "DistributedWorker",
    "SharedFrameBuffer",
    "WorkerCrashed",
    "WorkerError",
    "WorkerGroup",
]
