"""Carry an object across a process boundary without copying its arrays into the pickle.

:func:`pack` pickles an object with protocol 5. Every flat buffer the object
holds, a contiguous numpy array above all, is written into the writer's
:class:`~reactor_runtime.ipc.SharedSlot` instead of into the pickle. The header
stays small and travels on a queue or a socket; the bytes sit in shared memory.
The header names the block, so the reader attaches to whatever it is told and
the block can be replaced without a protocol of its own.

A torch tensor and a non-contiguous array do not expose a flat buffer and
pickle inline. That still works, only slower, and is reported once per process
when the header passes about one megabyte.
"""

from __future__ import annotations

import pickle
from typing import Any

from reactor_runtime.ipc.slot import SharedSlot, SharedSlotLimitExceeded, SlotReader
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_INLINE_WARN_BYTES = 1 << 20

_inline_reported = False


def pack(obj: Any, slot: SharedSlot | None) -> bytes:
    """Pickle *obj* with its flat buffers written into *slot*, or inline with no slot.

    With no slot every buffer stays in the pickle, and the header has the
    same shape, so :func:`unpack` reads both without telling them apart.

    Returns:
        The header: the pickle bytes, the block's name, and the span of each
        out-of-band buffer, pickled together.

    Raises:
        SharedSlotLimitExceeded: The flat buffers hold more than the slot's
            ``max_bytes``. It names their total size; no header is returned,
            so the bytes written before the limit are never read.
        SharedSlotAllocationFailed: The slot had to grow and could not.
    """
    if slot is None:
        return pickle.dumps((pickle.dumps(obj, protocol=5), None, []), protocol=5)
    slot.begin()
    spans: list[tuple[int, int]] = []
    needed = 0
    refused: SharedSlotLimitExceeded | None = None

    def out_of_band(buffer: pickle.PickleBuffer) -> None:
        nonlocal needed, refused
        needed += buffer.raw().nbytes
        if refused is None:
            try:
                spans.append(slot.write(buffer))
            except SharedSlotLimitExceeded as exc:
                refused = exc

    blob = pickle.dumps(obj, protocol=5, buffer_callback=out_of_band)
    if refused is not None:
        raise SharedSlotLimitExceeded(needed, refused.limit)
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
