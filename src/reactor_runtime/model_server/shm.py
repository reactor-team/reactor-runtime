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

A worker can also keep a session's blocks in a directory of its own, the
session's memory-backed volume mounted in the worker, where the session sees
the same files as its ``/dev/shm`` (:class:`FileBlock`). Every function that
maps or unlinks a block then takes that *directory*; ``None`` is ``/dev/shm``
by name, as POSIX shared memory.
"""

from __future__ import annotations

import contextlib
import mmap
import os
import sys
import threading
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any

from reactor_runtime.model_server import codec

_SHM_DIR = Path("/dev/shm")


class RequestRefused(ValueError):  # noqa: N818 (named for the event a caller catches)
    """A request header that is malformed, names a foreign block, or is too large."""


# Prefixes of the blocks this process creates. Its tracker holds those for the
# creator, so attaching to one of them must leave the registration alone.
_created_here: set[str] = set()
_created_lock = threading.Lock()


def created_here(prefix: str) -> str:
    """Record that this process creates the blocks named under *prefix*, and return it."""
    with _created_lock:
        _created_here.add(prefix)
    return prefix


def forget_created(prefix: str) -> None:
    """Drop a prefix :func:`created_here` recorded, once nothing more is made under it."""
    with _created_lock:
        _created_here.discard(prefix)


class FileBlock:
    """A shared block kept as a file in a directory other than ``/dev/shm``.

    It answers what this package uses of :class:`multiprocessing.shared_memory.SharedMemory`
    (``name``, ``size``, ``buf``, :meth:`close`, :meth:`unlink`), so a block in
    a session's own volume is handled like one in ``/dev/shm``. No resource
    tracker sees it: its owner unlinks it.

    Raises:
        ValueError: *name* is not a plain file name.
        OSError: The file cannot be made or opened.
    """

    def __init__(self, directory: Path, name: str, *, create: bool = False, size: int = 0) -> None:
        if not name or "/" in name or "\0" in name or name in (".", ".."):
            raise ValueError(f"{name!r} is not a block name")
        self._path = directory / name
        self._name = name
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        self._fd = os.open(self._path, flags, 0o600)
        self._mmap: mmap.mmap | None = None
        self._buf: memoryview | None = None
        try:
            if create:
                if size <= 0:
                    raise ValueError("a new block needs a positive size")
                os.ftruncate(self._fd, size)
            self._size = os.fstat(self._fd).st_size
            self._mmap = mmap.mmap(self._fd, self._size)
        except BaseException:
            os.close(self._fd)
            if create:
                with contextlib.suppress(OSError):
                    self._path.unlink()
            raise
        self._buf = memoryview(self._mmap)

    @property
    def name(self) -> str:
        """The block's file name in its directory."""
        return self._name

    @property
    def size(self) -> int:
        """The block's size in bytes."""
        return self._size

    @property
    def buf(self) -> memoryview | None:
        """The block's bytes, writable; ``None`` once closed."""
        return self._buf

    def close(self) -> None:
        """Unmap the block, as ``SharedMemory.close`` does: a view still held raises BufferError."""
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def unlink(self) -> None:
        """Remove the block's file; mappings still open keep its pages until they go."""
        self._path.unlink()

    def abandon(self) -> None:
        """Let go of the block while views of it are still held; the last view unmaps it."""
        self._buf = None
        self._mmap = None
        if self._fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = -1


# A shared block: POSIX shared memory by name, or a file in a session's own directory.
type Block = shared_memory.SharedMemory | FileBlock


def create(name: str, size: int, directory: Path | None = None) -> Block:
    """Make block *name* of *size* bytes, in *directory* or ``/dev/shm``."""
    if directory is not None:
        return FileBlock(directory, name, create=True, size=size)
    return shared_memory.SharedMemory(name=name, create=True, size=size)


def attach(name: str, directory: Path | None = None) -> shared_memory.SharedMemory | FileBlock:
    """Map the existing block *name* without handing it to this process's tracker."""
    if directory is not None:
        return FileBlock(directory, name)
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    shm = shared_memory.SharedMemory(name=name)
    with _created_lock:
        mine = any(name.startswith(prefix) for prefix in _created_here)
    if not mine:
        with contextlib.suppress(Exception):
            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return shm


def release(shm: Block) -> None:
    """Close *shm*, leaving its mapping to the views still exported from it.

    A view of the block keeps the mapping alive by itself; it is unmapped
    when the last view goes. Closing the wrapper must not try to unmap it
    first, which is what ``SharedMemory.close`` does and fails at.
    """
    try:
        shm.close()
    except BufferError:
        if isinstance(shm, FileBlock):
            shm.abandon()
            return
        shm._mmap = None  # type: ignore[attr-defined]  # ty: ignore[invalid-assignment]
        fd = getattr(shm, "_fd", -1)
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
            shm._fd = -1  # type: ignore[attr-defined]  # ty: ignore[invalid-assignment]
    except OSError:
        pass


class ViewReader:
    """Rebuild a client's request from the block its header names, without copying.

    Args:
        prefix: The only block names this reader maps start with it.
        limit: The most bytes one request may carry, header and buffers together.
    """

    def __init__(self, prefix: str, limit: int, directory: Path | None = None) -> None:
        self._prefix = prefix
        self._limit = limit
        self._directory = directory
        self._shm: Block | None = None
        self._closed = False
        self._lock = threading.Lock()

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
            except (OSError, ValueError) as exc:
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
        """Drop the mapping, and map nothing more. A view still held keeps it alive."""
        with self._lock:
            self._closed = True
            if self._shm is not None:
                release(self._shm)
                self._shm = None

    def _attach(self, name: str) -> None:
        with self._lock:
            if self._closed:
                raise OSError("the session is closed")
            if self._shm is not None and self._shm.name == name:
                return
            if self._shm is not None:
                release(self._shm)
                self._shm = None
            self._shm = attach(name, self._directory)


def copy_unpack(header: bytes, mapping: dict[str, Block]) -> Any:
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


def close_mapping(mapping: dict[str, Block]) -> None:
    """Close the block :func:`copy_unpack` cached in *mapping*."""
    shm = mapping.pop("block", None)
    if shm is not None:
        release(shm)


def unlink_prefix(prefix: str, directory: Path | None = None) -> int:
    """Unlink every shared block whose name starts with *prefix*; return how many.

    In *directory*, or where ``/dev/shm`` lists the blocks (Linux). Call it
    once the processes that write under *prefix* are dead.
    """
    where = _SHM_DIR if directory is None else directory
    if not prefix or not where.is_dir():
        return 0
    count = 0
    for path in where.glob(f"{prefix}*"):
        with contextlib.suppress(OSError):
            path.unlink()
            count += 1
    return count
