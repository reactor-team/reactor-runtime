"""A writer's shared block, and the reader that copies out of it.

A :class:`SharedSlot` is one block a writer fills, one call at a time; a
:class:`SlotReader` maps the block a header names and copies spans out of it.
Every block a slot creates is named ``<prefix><generation>``, so the holder of
the prefix can reclaim what a writer left behind with
:func:`~reactor_runtime.ipc.unlink_blocks`.
"""

from __future__ import annotations

import pickle
import secrets
from pathlib import Path
from typing import Any

from reactor_runtime.ipc.blocks import (
    Block,
    SharedSlotAllocationFailed,
    attach_block,
    create_block,
)
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_INITIAL_BYTES = 1 << 20


class SharedSlotLimitExceeded(ValueError):  # noqa: N818 (named for the event a caller catches)
    """A call holds more bytes than its slot's ``max_bytes``; no header describes it.

    Attributes:
        needed: The bytes the call's buffers hold.
        limit: The slot's ``max_bytes``.
    """

    def __init__(self, needed: int, limit: int) -> None:
        """Record the call's size and the slot's limit."""
        super().__init__(f"a call holds {needed} bytes, above the slot's limit of {limit}")
        self.needed = needed
        self.limit = limit

    def __reduce__(self) -> tuple[type[SharedSlotLimitExceeded], tuple[int, int]]:
        """Pickle by its two sizes, so it crosses a process boundary intact."""
        return (type(self), (self.needed, self.limit))


def new_prefix() -> str:
    """Return a fresh block-name prefix for one slot."""
    return f"rr{secrets.token_hex(4)}_"


class SharedSlot:
    """One shared block a writer fills, one call at a time.

    A call starts with :meth:`begin`, appends buffers with :meth:`write`, and
    hands the returned spans to the reader inside the header. When a call's
    buffers do not fit, the block is replaced by one twice its size, up to
    *max_bytes*, or by one of exactly the size the call needs when the doubled
    one cannot be allocated, and the bytes already written move with it.
    Steady state is one block at the writer's high-water mark.

    Every block reserves its whole capacity when it is created, not only the
    bytes written into it, and the old block is freed only once the new one
    is filled: a growth briefly holds both.

    With *max_bytes* equal to *initial_bytes* the block never grows: the slot
    holds exactly one block of that size, and a call larger than it is refused.

    Args:
        prefix: Names every block this slot creates, as ``<prefix><generation>``.
            A fresh one from :func:`new_prefix` when omitted.
        initial_bytes: The size of the first block.
        max_bytes: The most one call may hold; unbounded when omitted.
        directory: Where the blocks live; the reader names the same one.
            :func:`~reactor_runtime.ipc.default_directory` when omitted.

    Raises:
        SharedSlotAllocationFailed: The directory cannot hold the first block.
        ValueError: *initial_bytes* is above *max_bytes*.
    """

    def __init__(
        self,
        *,
        prefix: str | None = None,
        initial_bytes: int = _INITIAL_BYTES,
        max_bytes: int | None = None,
        directory: Path | None = None,
    ) -> None:
        if max_bytes is not None and initial_bytes > max_bytes:
            raise ValueError(f"initial_bytes {initial_bytes} is above max_bytes {max_bytes}")
        self._prefix = prefix or new_prefix()
        self._max_bytes = max_bytes
        self._directory = directory
        self._generation = 0
        self._block = create_block(f"{self._prefix}0", initial_bytes, directory)
        self._cursor = 0

    @property
    def prefix(self) -> str:
        """The prefix every block of this slot is named under."""
        return self._prefix

    @property
    def name(self) -> str:
        """The block a reader attaches to. Changes when the block grows."""
        return self._block.name

    @property
    def capacity(self) -> int:
        """Bytes the current block holds."""
        return self._block.size

    @property
    def max_bytes(self) -> int | None:
        """The most one call may hold, or ``None`` when the block grows without bound."""
        return self._max_bytes

    def begin(self) -> None:
        """Start a new call; the next write lands at offset zero."""
        self._cursor = 0

    def write(self, buffer: Any) -> tuple[int, int]:
        """Append one flat buffer and return its ``(offset, length)`` span.

        Raises:
            SharedSlotLimitExceeded: The call would hold more than
                :attr:`max_bytes`; this buffer is not written.
            SharedSlotAllocationFailed: The block had to grow and its
                directory cannot hold the larger one.
        """
        raw = buffer.raw() if isinstance(buffer, pickle.PickleBuffer) else memoryview(buffer)
        raw = raw.cast("B")
        length = raw.nbytes
        end = self._cursor + length
        if end > self.capacity:
            self._grow(end)
        self._block.buf[self._cursor : end] = raw
        span = (self._cursor, length)
        self._cursor = end
        return span

    def close(self) -> None:
        """Unmap and unlink the block. Idempotent."""
        self._block.close()
        self._block.unlink()

    def _grow(self, needed: int) -> None:
        size = max(needed, 2 * self.capacity)
        if self._max_bytes is not None:
            if needed > self._max_bytes:
                raise SharedSlotLimitExceeded(needed, self._max_bytes)
            size = min(size, self._max_bytes)
        self._generation += 1
        name = f"{self._prefix}{self._generation}"
        try:
            replacement = create_block(name, size, self._directory)
        except SharedSlotAllocationFailed:
            if size == needed:
                raise
            replacement = create_block(name, needed, self._directory)
        replacement.buf[: self._cursor] = self._block.buf[: self._cursor]
        logger.info("shared block grown", from_bytes=self.capacity, to_bytes=replacement.size)
        self.close()
        self._block = replacement


class SlotReader:
    """The reading side: maps the block a header names and copies spans out of it.

    Args:
        directory: Where the writer's blocks live;
            :func:`~reactor_runtime.ipc.default_directory` when omitted.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory
        self._block: Block | None = None

    def attach(self, name: str) -> None:
        """Map the block called *name*, replacing the previous mapping if it differs.

        Raises:
            OSError: The block is not there.
            ValueError: *name* is not a plain block name, or not a regular file.
        """
        if self._block is not None and self._block.name == name:
            return
        self.close()
        self._block = attach_block(name, self._directory)

    def copy(self, offset: int, length: int) -> bytearray:
        """Copy one span out. A copy, so the writer may reuse the block at once.

        Raises:
            RuntimeError: :meth:`attach` has not run.
            ValueError: The span ends past the block.
        """
        if self._block is None:
            raise RuntimeError("attach() has not run")
        if offset < 0 or length < 0 or offset + length > self._block.size:
            raise ValueError(
                f"span ({offset}, {length}) ends past the block's {self._block.size} bytes"
            )
        return bytearray(self._block.buf[offset : offset + length])

    def close(self) -> None:
        """Drop the mapping. The writer owns the block and unlinks it."""
        if self._block is not None:
            self._block.close()
            self._block = None
