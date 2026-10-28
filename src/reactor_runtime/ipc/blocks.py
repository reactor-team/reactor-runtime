"""Shared-memory blocks: named files in a memory-backed directory, mapped into each process.

A block is a regular file in a directory, ``/dev/shm`` by default, mapped with
``mmap``. Every process that sees the directory sees the same bytes: the
processes of one host, or containers that mount the same memory-backed volume.
Each block lives in the directory its creator chose, and both sides name that
directory; a block name never carries a path.

Blocks are opened as plain files, never through
:class:`multiprocessing.shared_memory.SharedMemory`, so no resource tracker
registers them: a process that maps a block it did not create never unlinks it
when it exits, on any Python version. The creator unlinks what it created, and
:func:`unlink_blocks` reclaims what a creator that died left behind.
"""

from __future__ import annotations

import contextlib
import errno
import mmap
import os
import shutil
import stat
import tempfile
from pathlib import Path

SHM_DIR = Path("/dev/shm")


class SharedSlotAllocationFailed(MemoryError):  # noqa: N818 (named for the event a caller catches)
    """A shared-memory block could not be created or grown.

    The message states the size that was asked for and, where the host exposes
    it, the size of the block's directory. Docker's default ``/dev/shm`` is
    64 MB; ``--shm-size`` on ``docker run`` and a ``medium: Memory``
    ``emptyDir`` on a pod raise it.
    """


def default_directory() -> Path:
    """Return where blocks live when the caller names no directory.

    ``/dev/shm`` where the host has it, and the temporary directory otherwise
    (macOS), where the same code runs on file-backed memory.
    """
    return SHM_DIR if SHM_DIR.is_dir() else Path(tempfile.gettempdir())


class Block:
    """One shared block, mapped into this process.

    Build one with :func:`create_block` or :func:`attach_block`. The mapping
    stays valid until :meth:`close`; the file stays until :meth:`unlink`, and
    a mapping still open keeps its pages after the file is gone.
    """

    def __init__(self, path: Path, mapping: mmap.mmap) -> None:
        """Hold an open mapping of *path*; use :func:`create_block` or :func:`attach_block`."""
        self._path = path
        self._mmap: mmap.mmap | None = mapping
        self._size = len(mapping)

    @property
    def name(self) -> str:
        """The block's file name in its directory, which a header carries."""
        return self._path.name

    @property
    def path(self) -> Path:
        """The block's file."""
        return self._path

    @property
    def size(self) -> int:
        """The block's size in bytes."""
        return self._size

    @property
    def buf(self) -> memoryview:
        """The block's bytes, writable.

        Raises:
            RuntimeError: The block is closed.
        """
        if self._mmap is None:
            raise RuntimeError(f"shared block {self.name} is closed")
        return memoryview(self._mmap)

    def close(self) -> None:
        """Unmap the block. Idempotent.

        Raises:
            BufferError: A view of the block is still held.
        """
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def unlink(self) -> None:
        """Remove the block's file; a mapping still open keeps its pages until it closes."""
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()


def create_block(name: str, size: int, directory: Path | None = None) -> Block:
    """Create block *name* of *size* bytes in *directory*, and reserve its pages.

    The pages are reserved with ``posix_fallocate`` where the platform has it,
    so a directory that cannot hold the block is an error here, not a
    ``SIGBUS`` on a later write. On a memory-backed directory the pages are
    charged to the creating process.

    Args:
        name: A plain file name; the block must not exist.
        size: Bytes, at least 1.
        directory: Where the block lives; :func:`default_directory` when omitted.

    Raises:
        ValueError: *name* is not a plain file name, or *size* is below 1.
        FileExistsError: A file of that name is already there.
        SharedSlotAllocationFailed: The directory cannot hold the block.
    """
    _check_name(name)
    if size < 1:
        raise ValueError(f"a shared block needs at least 1 byte, got {size}")
    path = (directory or default_directory()) / name
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        try:
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(fd, 0, size)
            else:
                os.ftruncate(fd, size)
        except OSError as exc:
            raise SharedSlotAllocationFailed(_no_room(path.parent, size, exc)) from exc
        mapping = mmap.mmap(fd, size)
    except BaseException:
        os.close(fd)
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    os.close(fd)
    return Block(path, mapping)


def attach_block(name: str, directory: Path | None = None) -> Block:
    """Map the existing block *name* in *directory*.

    Only a regular file is mapped: a symlink, a FIFO, or a directory under a
    block's name fails here instead of mapping something else.

    Args:
        name: The block's file name, as a header carries it.
        directory: Where the block lives; :func:`default_directory` when omitted.

    Raises:
        ValueError: *name* is not a plain file name, or names something other
            than a non-empty regular file.
        OSError: The file cannot be opened (``FileNotFoundError`` when it is
            not there, ``ELOOP`` for a symlink).
    """
    _check_name(name)
    path = (directory or default_directory()) / name
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"shared block {name!r} is not a regular file")
        if info.st_size < 1:
            raise ValueError(f"shared block {name!r} is empty")
        mapping = mmap.mmap(fd, info.st_size)
    finally:
        os.close(fd)
    return Block(path, mapping)


def unlink_blocks(prefix: str, directory: Path | None = None) -> int:
    """Unlink every block in *directory* whose name starts with *prefix*; return how many.

    The prefix is compared as plain text, so a ``*`` or ``[`` in it matches
    only itself.

    For blocks a creator left behind when it was terminated before it could
    unlink them. Call it only once the processes that create blocks under
    *prefix* are gone: a live creator's blocks would go with the rest.

    Raises:
        ValueError: *prefix* is empty or not a plain file name.
    """
    _check_name(prefix)
    where = directory or default_directory()
    count = 0
    for path in where.iterdir():
        if not path.name.startswith(prefix):
            continue
        with contextlib.suppress(OSError):
            path.unlink()
            count += 1
    return count


def _check_name(name: str) -> None:
    if not name or "/" in name or "\0" in name or name in (".", ".."):
        raise ValueError(f"{name!r} is not a plain block name")


def _no_room(directory: Path, size: int, exc: OSError) -> str:
    message = f"could not allocate a shared block of {size / 2**20:.1f} MB in {directory}"
    with contextlib.suppress(OSError):
        usage = shutil.disk_usage(directory)
        message += (
            f", which has {usage.total / 2**20:.1f} MB total and {usage.free / 2**20:.1f} MB free"
        )
    reason = errno.errorcode.get(exc.errno or 0, str(exc))
    return (
        f"{message} ({reason}). Run with a larger --shm-size, or size the memory-backed "
        "volume in the deployment."
    )
