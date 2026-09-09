"""Experimental single-node, single-session multi-GPU video primitives.

Signatures may change between minor releases. Import from
``reactor_runtime.distributed``; these names stay off the package root.

Subclass :class:`DistributedWorker` to define per-rank setup, warmup, session,
and generation hooks. :class:`WorkerGroup` owns process spawning, commands and
acknowledgements, bounded uint8 frame transport, liveness checks, and teardown.
A one-worker group runs inline without child processes or a process group.
Use the manifest-derived ``self.world_size`` when constructing the group
during model loading.

Torch is imported lazily in workers when the model image provides it. The
controller stays torch-free. The framework creates the process group before
``setup``, and the model owns its compute collectives. Clean shutdown includes
a final barrier before destruction. This package does not provide arbitrary
structured outputs or multi-session scheduling.
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
