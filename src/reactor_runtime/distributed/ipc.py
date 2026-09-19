"""Carry an author's object across a process boundary without copying it twice.

:func:`pack` pickles an object with protocol 5. Every flat buffer the object
holds, a contiguous numpy array above all, is handed to the writer's
:class:`SharedSlot` instead of being written into the pickle. The pickle
stays small and travels on a queue; the bytes sit in shared memory. The
header names the block, so the reader attaches to whatever it is told and
the block can grow without a protocol of its own.

A torch tensor and a non-contiguous array do not expose a flat buffer and
pickle inline. That still works, only slower, and is reported once per
process when the header passes about one megabyte.
"""

from __future__ import annotations

import contextlib
import pickle
import shutil
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

from reactor_runtime.distributed.errors import SharedSlotAllocationFailed
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_INITIAL_BYTES = 1 << 20
_INLINE_WARN_BYTES = 1 << 20
_SHM_DIR = Path("/dev/shm")

_inline_reported = False


class SharedSlot:
    """One shared-memory block a writer fills, one call at a time.

    A call starts with :meth:`begin`, appends buffers with :meth:`write`, and
    hands the returned spans to the reader inside the header. When a call's
    buffers do not fit, the block is replaced by a larger one and the bytes
    already written move with it. Steady state is one block at the writer's
    high-water mark.
    """

    def __init__(self, *, initial_bytes: int = _INITIAL_BYTES) -> None:
        self._shm = _allocate(initial_bytes)
        self._cursor = 0
        self._pinned = False

    @property
    def name(self) -> str:
        """The name a reader attaches to. Changes when the block grows."""
        return self._shm.name

    @property
    def capacity(self) -> int:
        """Bytes the current block holds."""
        return self._shm.size

    @property
    def _buf(self) -> memoryview:
        return _buffer_of(self._shm)

    def begin(self) -> None:
        """Start a new call; the next write lands at offset zero."""
        self._cursor = 0

    def write(self, buffer: Any) -> tuple[int, int]:
        """Append one flat buffer and return its ``(offset, length)`` span."""
        if isinstance(buffer, pickle.PickleBuffer):
            raw = buffer.raw()
        else:
            raw = memoryview(buffer).cast("B")
        length = raw.nbytes
        end = self._cursor + length
        if end > self.capacity:
            self._grow(end)
        self._buf[self._cursor : end] = raw
        span = (self._cursor, length)
        self._cursor = end
        return span

    def pin(self) -> bool:
        """Register the block as pinned host memory so GPU copies take the DMA path.

        Best effort. Without torch, without CUDA, or with a ``ulimit -l`` too
        low to lock the pages, the copy stays pageable and one warning says so.

        Returns:
            Whether the block is pinned.
        """
        try:
            import torch  # ty: ignore[unresolved-import]  # installed in the model image only

            if not torch.cuda.is_available():
                return False
            address = _address(self._buf)
            code = int(torch.cuda.cudart().cudaHostRegister(address, self._shm.size, 0))
        except Exception as exc:
            logger.warning("host-memory pinning unavailable; copies stay pageable", error=str(exc))
            return False
        if code != 0:
            logger.warning(
                "cudaHostRegister failed; copies stay pageable. The usual cause is a low ulimit -l",
                error_code=code,
                bytes=self._shm.size,
            )
            return False
        self._pinned = True
        return True

    def close(self) -> None:
        """Unpin, detach, and unlink the block."""
        self._unpin()
        with contextlib.suppress(Exception):
            self._shm.close()
        with contextlib.suppress(Exception):
            self._shm.unlink()

    def _grow(self, needed: int) -> None:
        replacement = _allocate(max(needed, 2 * self.capacity))
        _buffer_of(replacement)[: self._cursor] = self._buf[: self._cursor]
        was_pinned = self._pinned
        logger.info("shared block grown", from_bytes=self.capacity, to_bytes=replacement.size)
        self.close()
        self._shm = replacement
        if was_pinned:
            self.pin()

    def _unpin(self) -> None:
        if not self._pinned:
            return
        with contextlib.suppress(Exception):
            import torch  # ty: ignore[unresolved-import]  # same as in pin()

            torch.cuda.cudart().cudaHostUnregister(_address(self._buf))
        self._pinned = False


class SlotReader:
    """The reading side: attaches to the block a header names and keeps the mapping."""

    def __init__(self) -> None:
        self._shm: shared_memory.SharedMemory | None = None

    def attach(self, name: str) -> None:
        """Map the block called *name*, replacing the previous mapping if it differs."""
        if self._shm is not None and self._shm.name == name:
            return
        self.close()
        self._shm = shared_memory.SharedMemory(name=name)

    def copy(self, offset: int, length: int) -> bytearray:
        """Copy one span out. A copy, so the writer may reuse the block at once."""
        if self._shm is None:
            raise RuntimeError("attach() has not run")
        return bytearray(_buffer_of(self._shm)[offset : offset + length])

    def close(self) -> None:
        """Drop the mapping. The writer owns the block and unlinks it."""
        if self._shm is not None:
            with contextlib.suppress(Exception):
                self._shm.close()
            self._shm = None


def pack(obj: Any, slot: SharedSlot) -> bytes:
    """Pickle *obj* with its flat buffers written into *slot*.

    Returns:
        A header for the queue: the pickle bytes, the block's name, and the
        span of each out-of-band buffer, pickled together.
    """
    slot.begin()
    spans: list[tuple[int, int]] = []

    def out_of_band(buffer: pickle.PickleBuffer) -> None:
        spans.append(slot.write(buffer))

    blob = pickle.dumps(obj, protocol=5, buffer_callback=out_of_band)
    if len(blob) > _INLINE_WARN_BYTES:
        _report_inline(len(blob))
    return pickle.dumps((blob, slot.name, spans), protocol=5)


def unpack(header: bytes, reader: SlotReader) -> Any:
    """Rebuild the object a :func:`pack` header describes."""
    blob, name, spans = pickle.loads(header)
    reader.attach(name)
    return pickle.loads(blob, buffers=[reader.copy(offset, length) for offset, length in spans])


def _allocate(size: int) -> shared_memory.SharedMemory:
    if _SHM_DIR.exists():
        usage = shutil.disk_usage(_SHM_DIR)
        if size > usage.free:
            raise SharedSlotAllocationFailed(
                f"could not grow the shared buffer to {size / 2**20:.1f} MB. /dev/shm has "
                f"{usage.total / 2**20:.1f} MB total and {usage.free / 2**20:.1f} MB free. "
                "Run with a larger --shm-size, or size /dev/shm in the deployment."
            )
    try:
        return shared_memory.SharedMemory(create=True, size=size)
    except OSError as exc:
        raise SharedSlotAllocationFailed(
            f"could not allocate a shared buffer of {size / 2**20:.1f} MB: {exc}. "
            "Run with a larger --shm-size, or size /dev/shm in the deployment."
        ) from exc


def _buffer_of(shm: shared_memory.SharedMemory) -> memoryview:
    buf = shm.buf
    if buf is None:
        raise RuntimeError(f"shared block {shm.name} is closed")
    return buf


def _address(buf: memoryview) -> int:
    import ctypes

    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def _report_inline(size: int) -> None:
    global _inline_reported
    if _inline_reported:
        return
    _inline_reported = True
    logger.warning(
        "a call's header is large; some payload pickled inline instead of through shared memory. "
        "A torch tensor or a non-contiguous array does that: pass np.ascontiguousarray(x) or "
        "np.asarray(tensor.cpu())",
        header_bytes=size,
    )
