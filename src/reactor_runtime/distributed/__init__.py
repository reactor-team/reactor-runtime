"""Experimental single-node GPU worker protocol and uint8 frame transport.

Signatures may change between minor releases. Import these primitives from
``reactor_runtime.distributed``; they stay off the package root.

Subclass :class:`DistributedWorker` to define per-rank setup, warmup, session,
and generation hooks. :class:`SharedFrameBuffer` provides bounded uint8 video
transport between a worker and its controller. These primitives do not provide
a model adapter, multi-session scheduling, or arbitrary structured outputs.

Importing this package does not require torch. Workers use torch for CUDA
device binding and process-group setup when the model image provides it.
"""

from reactor_runtime.distributed.errors import WorkerCrashed, WorkerError
from reactor_runtime.distributed.frames import SharedFrameBuffer
from reactor_runtime.distributed.worker import DistributedWorker

__all__ = [
    "DistributedWorker",
    "SharedFrameBuffer",
    "WorkerCrashed",
    "WorkerError",
]
