# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

"""Shared-memory frame transport between worker ranks and the controller.

One uint8 buffer of ``frame_shape`` is shared by every process. Workers
write pixels in place; only the frame count crosses the command queues
(pickling frames through a queue costs tens of milliseconds per chunk;
the shared-memory write is a few milliseconds). The protocol is strict
request/response — the controller copies frames out before dispatching
the next chunk — so a single buffer with no synchronization of its own
is safe, and is itself the backpressure.
"""

from __future__ import annotations

import contextlib
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from reactor_runtime.utils.log import get_logger

logger = get_logger(__name__)


class SharedFrameBuffer:
    """View over one POSIX shared-memory uint8 frame buffer.

    The controller creates it (``create=True``); each worker rank
    attaches by name. Workers with CUDA available should call
    :meth:`pin` once so per-chunk device-to-host copies into the buffer
    take the pinned DMA fast path.
    """

    def __init__(
        self, shape: tuple[int, ...], *, name: str | None = None, create: bool = False
    ) -> None:
        size = int(np.prod(shape))
        self._shm = shared_memory.SharedMemory(name=name, create=create, size=size)
        self._owner = create
        self._pinned = False
        self.shape = shape
        #: The buffer as a numpy array. Workers may write slices directly.
        self.array: np.ndarray = np.ndarray(shape, dtype=np.uint8, buffer=self._shm.buf)

    @property
    def name(self) -> str:
        """Attachment name for worker processes."""
        return self._shm.name

    def write(self, frames: Any, start_row: int = 0) -> int:
        """Copy ``frames`` into the buffer beginning at ``start_row``.

        Accepts a numpy array or a torch tensor (moved to CPU if
        needed). Returns the END row (``start_row + rows_written``) —
        forward it from ``generate_chunk`` as-is: the controller takes
        the max end row across all ranks as the chunk's frame count, so
        the total comes out right for whole-chunk writes and for
        frames-axis sharding alike. The controller reads the buffer only
        after every rank has replied, so all slices land first.
        """
        if not isinstance(frames, np.ndarray):
            frames = frames.detach().cpu().numpy()  # torch tensor
        end_row = start_row + int(frames.shape[0])
        self.array[start_row:end_row] = frames
        return end_row

    def read(self, n: int) -> np.ndarray:
        """Copy the first ``n`` frames out (releases the buffer for the
        next chunk — never hand out a view)."""
        return self.array[:n].copy()

    def pin(self) -> bool:
        """Best-effort ``cudaHostRegister`` of the buffer in this
        process's CUDA context, so GPU→buffer copies take the pinned DMA
        fast path. Registration is per-process. Failure (no torch, no
        CUDA, ``ulimit -l`` too low) logs and falls back to pageable —
        slower, not broken."""
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            ptr = self.array.ctypes.data
            err = int(torch.cuda.cudart().cudaHostRegister(ptr, self._shm.size, 0))
        except Exception as exc:  # noqa: BLE001 - degrade, never fail startup
            logger.warning(f"SharedFrameBuffer.pin failed ({exc}); pageable copies")
            return False
        if err != 0:
            logger.warning(
                f"cudaHostRegister failed (err={err}, {self._shm.size} bytes); "
                f"pageable copies. Likely ulimit -l too low."
            )
            return False
        self._pinned = True
        return True

    def close(self) -> None:
        """Detach (and unlink, in the creating process)."""
        if self._pinned:
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.cudart().cudaHostUnregister(self.array.ctypes.data)
            self._pinned = False
        # Drop the numpy view before closing the mapping.
        self.array = np.ndarray((0,), dtype=np.uint8)
        with contextlib.suppress(Exception):
            self._shm.close()
        if self._owner:
            with contextlib.suppress(Exception):
                self._shm.unlink()
