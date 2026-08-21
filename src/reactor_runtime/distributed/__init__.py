"""Multi-GPU worker abstraction for real-time streaming models.

A model that needs several GPUs cannot simply run several copies of
itself: it is a server, with one event loop, one session, and one output
stream. So the process splits in two roles, vended here as two classes:

- :class:`DistributedWorker` — the per-GPU class a model author
  subclasses. Hooks: ``setup`` / ``warmup`` / ``start_session`` /
  ``generate_chunk`` / ``end_session``.
- :class:`WorkerGroup` — the controller handle a model holds, created in
  ``load()``. Owns process spawning, the command/ack protocol, the
  shared-memory frame transport, liveness detection, and teardown.

``torch`` is imported lazily and only where available: the controller
side runs torch-free, and worker processes use it (NCCL process group,
CUDA device binding, host-memory pinning) when the image provides it.

The process group belongs entirely to the model. The framework creates
it before ``setup`` runs and then issues no collectives of its own, so a
model is free to carve its own sub-groups out of it — tensor-,
sequence-, or context-parallel — by calling ``new_group`` or
``init_device_mesh`` inside ``setup``, where every rank reaches the call
in the same order.
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
