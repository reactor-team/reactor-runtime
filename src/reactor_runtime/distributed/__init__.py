"""Multi-GPU worker abstraction for real-time streaming models.

.. warning:: **Experimental.** This package supports single-node, single-session
   uint8 video, not arbitrary structured outputs or multi-session scheduling.
   Signatures may change between minor releases. Import from
   ``reactor_runtime.distributed``; these names stay off the package root.

Use :class:`DistributedVideoModel` for the video authoring path: declare a
worker and map its frames to an Output. The adapter owns threading, session
bookkeeping, pause/restart, and cleanup. It is a :class:`ReactorModel`, not an
engine host or multi-session scheduler. Use :class:`WorkerGroup` directly only
when custom orchestration requires the raw primitives.

A model that needs several GPUs cannot simply run several copies of
itself: it is a server, with one event loop, one session, and one output
stream. So the process splits in two roles, vended here as two classes:

- :class:`DistributedWorker` — the per-GPU class a model author
  subclasses. Hooks: ``setup`` / ``warmup`` / ``start_session`` /
  ``generate_chunk`` / ``end_session``.
- :class:`WorkerGroup` — the controller handle a model holds, created in
  ``load()``. Owns process spawning, the command/ack protocol, the
  frame transport, liveness detection, and teardown. A one-worker group
  runs inline without child processes, shared memory, or a process group.

``torch`` is imported lazily and only where available: the controller
side of a multi-worker group runs torch-free, and workers use it (NCCL process group,
CUDA device binding, host-memory pinning) when the image provides it.

The process group belongs entirely to the model. The framework creates
it before ``setup`` runs and issues no compute collectives of its own, so a
model is free to carve its own sub-groups out of it — tensor-,
sequence-, or context-parallel — by calling ``new_group`` or
``init_device_mesh`` inside ``setup``, where every rank reaches the call
in the same order. Clean shutdown includes a final barrier before destruction.
"""

from reactor_runtime.distributed.errors import WorkerCrashed, WorkerError
from reactor_runtime.distributed.frames import SharedFrameBuffer
from reactor_runtime.distributed.group import WorkerGroup
from reactor_runtime.distributed.model import DistributedVideoModel
from reactor_runtime.distributed.worker import DistributedWorker

__all__ = [
    "DistributedVideoModel",
    "DistributedWorker",
    "SharedFrameBuffer",
    "WorkerCrashed",
    "WorkerError",
    "WorkerGroup",
]
