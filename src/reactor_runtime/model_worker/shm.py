"""Shared-memory helpers the model worker and its clients add to the distributed ipc.

Two things the runner's :class:`~reactor_runtime.distributed.ipc.SlotReader`
does not do. :class:`ViewReader` hands out read-only views instead of copies,
for the worker side where a request's bytes stay put until its answer is sent,
and it checks every name and span before it maps anything, so a request names
only its own client's blocks. :func:`unlink_prefix` reclaims every block whose
name starts with a prefix, for the worker once a session that made blocks is
gone, and for a worker that starts after one that died.

On Python 3.12 attaching to a block registers it with the attaching process's
resource tracker, which unlinks it when that process exits, under the writer's
feet. Attaching here unregisters it again (3.13 and later take ``track=False``).
"""

from __future__ import annotations

import contextlib
import os
import sys
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any

from reactor_runtime.model_worker import codec

_SHM_DIR = Path("/dev/shm")


class RequestRefused(ValueError):  # noqa: N818 (named for the event a caller catches)
    """A request header that is malformed, names a foreign block, or is too large."""


# Prefixes of the blocks this process creates. Its tracker holds those for the
# creator, so attaching to one of them must leave the registration alone.
_created_here: set[str] = set()


def created_here(prefix: str) -> str:
    """Record that this process creates the blocks named under *prefix*, and return it."""
    _created_here.add(prefix)
    return prefix


def forget_created(prefix: str) -> None:
    """Drop a prefix :func:`created_here` recorded, once nothing more is made under it."""
    _created_here.discard(prefix)


def attach(name: str) -> shared_memory.SharedMemory:
    """Map the existing block *name* without handing it to this process's tracker."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    shm = shared_memory.SharedMemory(name=name)
    if not any(name.startswith(prefix) for prefix in _created_here):
        with contextlib.suppress(Exception):
            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return shm


def release(shm: shared_memory.SharedMemory) -> None:
    """Close *shm*, leaving its mapping to the views still exported from it.

    A view of the block keeps the mapping alive by itself; it is unmapped
    when the last view goes. Closing the wrapper must not try to unmap it
    first, which is what ``SharedMemory.close`` does and fails at.
    """
    try:
        shm.close()
    except BufferError:
        shm._mmap = None  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        fd = getattr(shm, "_fd", -1)
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
            shm._fd = -1  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    except OSError:
        pass


class ViewReader:
    """Rebuild a client's request from the block its header names, without copying.

    Args:
        prefix: The only block names this reader maps start with it.
        limit: The most bytes one request may carry, header and buffers together.
    """

    def __init__(self, prefix: str, limit: int) -> None:
        self._prefix = prefix
        self._limit = limit
        self._shm: shared_memory.SharedMemory | None = None

    def unpack(self, header: bytes) -> Any:
        """Return the payload *header* describes, its arrays as read-only views.

        The views stay valid while the client waits for its answer. Whoever
        keeps one past that copies it.

        Raises:
            RequestRefused: The header is malformed, names a block outside this
                reader's prefix, holds a span outside the block, or carries
                more than the limit.
        """
        if len(header) > self._limit:
            raise RequestRefused(f"the request's header passes the {self._limit} byte limit")
        try:
            name, spans, tree = codec.parse(header)
        except codec.MalformedPayload as exc:
            raise RequestRefused(f"the request does not decode: {exc}") from None
        total = len(header) + sum(length for _, length in spans)
        if total > self._limit:
            raise RequestRefused(f"the request carries {total} bytes, over the {self._limit} limit")
        views: list[memoryview] = []
        if spans:
            if not name.startswith(self._prefix):
                raise RequestRefused(f"the request names block {name!r}, not one of this client's")
            try:
                self._attach(name)
            except OSError as exc:
                raise RequestRefused(
                    f"the request's block {name!r} cannot be mapped: {exc}"
                ) from None
            assert self._shm is not None
            buf = self._shm.buf
            assert buf is not None
            size = self._shm.size
            for offset, length in spans:
                if offset + length > size:
                    raise RequestRefused(
                        f"a span ends at {offset + length}, past the block's {size}"
                    )
                views.append(buf[offset : offset + length].toreadonly())
        try:
            return codec.decode(tree, views.__getitem__)
        except codec.MalformedPayload as exc:
            raise RequestRefused(f"the request does not decode: {exc}") from None

    def close(self) -> None:
        """Drop the mapping. A view still held keeps it alive until it is released."""
        if self._shm is not None:
            release(self._shm)
            self._shm = None

    def _attach(self, name: str) -> None:
        if self._shm is not None and self._shm.name == name:
            return
        self.close()
        self._shm = attach(name)


def copy_unpack(header: bytes, mapping: dict[str, shared_memory.SharedMemory]) -> Any:
    """Rebuild an output *header* describes, copying its arrays out of the block.

    A copy, because the worker writes the next output into the same block while
    the session may still be emitting this one. *mapping* caches the one block
    mapped so far and is updated when the block grows.

    Raises:
        codec.MalformedPayload: The header is malformed.
    """
    name, spans, tree = codec.parse(header)
    copies: list[memoryview] = []
    if spans:
        shm = mapping.get("block")
        if shm is None or shm.name != name:
            if shm is not None:
                release(shm)
            shm = attach(name)
            mapping["block"] = shm
        buf = shm.buf
        assert buf is not None
        for offset, length in spans:
            if offset + length > shm.size:
                raise codec.MalformedPayload("a span ends past the output's block")
            copies.append(memoryview(bytearray(buf[offset : offset + length])))
    return codec.decode(tree, copies.__getitem__)


def close_mapping(mapping: dict[str, shared_memory.SharedMemory]) -> None:
    """Close the block :func:`copy_unpack` cached in *mapping*."""
    shm = mapping.pop("block", None)
    if shm is not None:
        release(shm)


def unlink_prefix(prefix: str) -> int:
    """Unlink every shared block whose name starts with *prefix*; return how many.

    Only where ``/dev/shm`` lists the blocks (Linux). Call it once the
    processes that write under *prefix* are dead.
    """
    if not prefix or not _SHM_DIR.is_dir():
        return 0
    count = 0
    for path in _SHM_DIR.glob(f"{prefix}*"):
        with contextlib.suppress(OSError):
            path.unlink()
            count += 1
    return count
