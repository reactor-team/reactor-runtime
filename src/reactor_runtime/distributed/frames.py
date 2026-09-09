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

from reactor_runtime.log import get_logger

logger = get_logger(__name__)


class _FrameBuffer:
    """An in-process uint8 frame buffer with the same write/read contract."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        _validate_shape(shape)
        self.shape = shape
        self.array: np.ndarray = np.empty(shape, dtype=np.uint8)

    def write(self, frames: Any, start_row: int = 0) -> int:
        """Copy complete frames and return their exclusive end row."""
        if not isinstance(frames, np.ndarray):
            frames = frames.detach().cpu().numpy()
        if frames.dtype != np.uint8:
            raise ValueError(f"frames must be uint8, got {frames.dtype}; convert before returning")
        if frames.ndim != len(self.shape) or frames.shape[1:] != self.shape[1:]:
            dimensions = ", ".join(map(str, self.shape[1:]))
            raise ValueError(
                f"frames have shape {frames.shape}; expected (N, {dimensions}) "
                f"for frame_shape={self.shape}"
            )
        if type(start_row) is not int or start_row < 0:
            raise ValueError(f"frame start_row must be a non-negative integer, got {start_row!r}")
        end_row = start_row + int(frames.shape[0])
        if end_row > self.shape[0]:
            raise ValueError(
                f"frame write [{start_row}:{end_row}] exceeds capacity {self.shape[0]}; "
                "increase frame_shape[0] or return a smaller chunk"
            )
        self.array[start_row:end_row] = frames
        return end_row

    def read(self, n: int) -> np.ndarray:
        """Copy frames, rejecting counts that would silently truncate output."""
        if type(n) is not int or not 0 <= n <= self.shape[0]:
            raise ValueError("frame count is outside the buffer")
        return self.array[:n].copy()

    def close(self) -> None:
        """Release the local array."""
        self.array = np.empty((0,), dtype=np.uint8)


def _validate_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("frame_shape must contain four positive integers: (frames, H, W, C)")


class SharedFrameBuffer(_FrameBuffer):
    """View over one POSIX shared-memory uint8 frame buffer.

    The controller creates it (``create=True``); each worker rank
    attaches by name. :meth:`pin` optionally registers the mapping for workers
    that copy tensors directly into it. :meth:`write` stages tensors on CPU;
    registration alone does not make that convenience path a direct DMA copy.
    """

    def __init__(
        self, shape: tuple[int, ...], *, name: str | None = None, create: bool = False
    ) -> None:
        _validate_shape(shape)
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
        return super().write(frames, start_row)

    def read(self, n: int) -> np.ndarray:
        """Copy the first ``n`` frames out.

        A copy, never a view: a view would let the next chunk overwrite frames
        the caller still holds, so copying is what releases the buffer.
        """
        return super().read(n)

    def pin(self) -> bool:
        """Register the buffer as pinned host memory, best effort.

        ``cudaHostRegister`` in this process's CUDA context, so GPU→buffer
        copies take the pinned DMA fast path. Registration is per-process, so
        every rank calls it for itself.

        Returns:
            Whether the buffer is now pinned. No torch, no CUDA, or a
            ``ulimit -l`` too low to lock the pages all log and return False,
            leaving copies pageable — slower, not broken.
        """
        try:
            # torch is an optional runtime dependency: the controller side runs
            # without it, so it is imported where it is used, not at module
            # level. ty cannot see an optional dependency, hence the ignore.
            import torch  # ty: ignore[unresolved-import]

            if not torch.cuda.is_available():
                return False
            ptr = self.array.ctypes.data
            err = int(torch.cuda.cudart().cudaHostRegister(ptr, self._shm.size, 0))
        except Exception as exc:
            # Any failure degrades to pageable copies rather than failing
            # startup: a slower copy is always better than a dead worker.
            logger.warning("host-memory pinning unavailable; copies stay pageable", error=str(exc))
            return False
        if err != 0:
            logger.warning(
                "cudaHostRegister failed; copies stay pageable. "
                "The usual cause is a low 'ulimit -l'.",
                error_code=err,
                bytes=self._shm.size,
            )
            return False
        self._pinned = True
        return True

    def close(self) -> None:
        """Detach (and unlink, in the creating process)."""
        if self._pinned:
            with contextlib.suppress(Exception):
                import torch  # ty: ignore[unresolved-import]

                torch.cuda.cudart().cudaHostUnregister(self.array.ctypes.data)
            self._pinned = False
        # Drop the numpy view before closing the mapping.
        self.array = np.ndarray((0,), dtype=np.uint8)
        with contextlib.suppress(Exception):
            self._shm.close()
        if self._owner:
            with contextlib.suppress(Exception):
                self._shm.unlink()
