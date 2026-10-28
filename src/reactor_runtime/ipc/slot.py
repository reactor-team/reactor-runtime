"""A writer's growing shared block, and the reader that copies out of it.

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

from reactor_runtime.ipc.blocks import Block, attach_block, create_block
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_INITIAL_BYTES = 1 << 20


def new_prefix() -> str:
    """Return a fresh block-name prefix for one slot."""
    return f"rr{secrets.token_hex(4)}_"


class SharedSlot:
    """One shared block a writer fills, one call at a time.

    A call starts with :meth:`begin`, appends buffers with :meth:`write`, and
    hands the returned spans to the reader inside the header. When a call's
    buffers do not fit, the block is replaced by one at least twice its size
    and the bytes already written move with it. Steady state is one block at
    the writer's high-water mark.

    Args:
        prefix: Names every block this slot creates, as ``<prefix><generation>``.
            A fresh one from :func:`new_prefix` when omitted.
        initial_bytes: The size of the first block.
        directory: Where the blocks live; the reader names the same one.
            :func:`~reactor_runtime.ipc.default_directory` when omitted.

    Raises:
        SharedSlotAllocationFailed: The directory cannot hold the first block.
    """

    def __init__(
        self,
        *,
        prefix: str | None = None,
        initial_bytes: int = _INITIAL_BYTES,
        directory: Path | None = None,
    ) -> None:
        self._prefix = prefix or new_prefix()
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

    def begin(self) -> None:
        """Start a new call; the next write lands at offset zero."""
        self._cursor = 0

    def write(self, buffer: Any) -> tuple[int, int]:
        """Append one flat buffer and return its ``(offset, length)`` span.

        Raises:
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
        self._generation += 1
        replacement = create_block(
            f"{self._prefix}{self._generation}",
            max(needed, 2 * self.capacity),
            self._directory,
        )
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
