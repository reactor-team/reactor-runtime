"""Carry an object across a process boundary without copying its arrays into the pickle.

:func:`pack` pickles an object with protocol 5. Every flat buffer the object
holds, a contiguous numpy array above all, is written into the writer's
:class:`~reactor_runtime.ipc.SharedSlot` instead of into the pickle. The header
stays small and travels on a queue or a socket; the bytes sit in shared memory.
The header names the block, so the reader attaches to whatever it is told and
the block can grow without a protocol of its own.

A torch tensor and a non-contiguous array do not expose a flat buffer and
pickle inline. That still works, only slower, and is reported once per process
when the header passes about one megabyte.
"""

from __future__ import annotations

import pickle
from typing import Any

from reactor_runtime.ipc.slot import SharedSlot, SlotReader
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_INLINE_WARN_BYTES = 1 << 20

_inline_reported = False


def pack(obj: Any, slot: SharedSlot) -> bytes:
    """Pickle *obj* with its flat buffers written into *slot*.

    Returns:
        The header: the pickle bytes, the block's name, and the span of each
        out-of-band buffer, pickled together.

    Raises:
        SharedSlotAllocationFailed: The slot had to grow and could not.
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
    """Rebuild the object a :func:`pack` header describes, its buffers copied out of the block.

    The object owns its buffers: the writer may reuse the block as soon as this
    returns.
    """
    blob, name, spans = pickle.loads(header)
    if not spans:
        return pickle.loads(blob)
    reader.attach(name)
    return pickle.loads(blob, buffers=[reader.copy(offset, length) for offset, length in spans])


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
