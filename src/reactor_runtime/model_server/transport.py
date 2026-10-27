"""How a session's requests and the worker's outputs travel: the transport behind the client.

A :class:`Transport` supplies two things. The stream a connection's frames
travel on (:meth:`Transport.connect` and :meth:`Transport.listen`), and, per
connection, a carrier that turns a request or an output into the body of a
frame and back. The frames, the protocol, and the batching above them are the
same whatever the transport is, so a transport that reaches across machines
can be added beside :class:`UnixShmTransport` without touching the model
contract.

:class:`UnixShmTransport` is the one that exists: a Unix socket for the
frames, and shared memory for the arrays, so a frame carries only a small
header. It needs the worker and its sessions on one host, sharing the socket's
directory and ``/dev/shm``; in Kubernetes, containers of one pod with the same
two volumes mounted. :meth:`Transport.check` refuses a deployment where that
does not hold, before anything is served.
"""

from __future__ import annotations

import contextlib
import errno
import os
import secrets
import socket
from abc import ABC, abstractmethod
from collections.abc import Mapping
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

from reactor_runtime.distributed.errors import SharedSlotAllocationFailed
from reactor_runtime.model_server import codec
from reactor_runtime.model_server.shm import (
    ViewReader,
    attach,
    close_mapping,
    copy_unpack,
    created_here,
    forget_created,
    release,
    unlink_prefix,
)

_SHM_DIR = Path("/dev/shm")


class WorkerUnreachable(RuntimeError):  # noqa: N818 (named for the state a caller catches)
    """The deployment cannot carry this transport: fix it, retrying will not.

    The message says what is missing, such as the socket's directory (the
    shared volume is not mounted) or shared memory the worker and the session
    do not both see (they are not on one host, or ``/dev/shm`` is not shared).
    """


class NoListener(Exception):  # noqa: N818 (named for the state a caller catches)
    """Nothing accepts at the address yet: the worker may still be starting."""


class ClientCarrier(ABC):
    """The session side of one connection: requests out, outputs back."""

    @abstractmethod
    def size(self, payload: Any) -> int:
        """Return the bytes *payload* would carry, without writing anything."""

    @abstractmethod
    def encode(self, payload: Any) -> tuple[bytes, int]:
        """Return a request's frame body and the bytes it carries in all."""

    @abstractmethod
    def decode(self, body: bytes) -> Any:
        """Rebuild an output from its frame body, as an object this session owns."""

    @abstractmethod
    def renew(self) -> None:
        """Stop writing where an abandoned request may still be read."""

    @abstractmethod
    def settled(self) -> None:
        """An answer arrived after which nothing written before is read again."""

    @abstractmethod
    def close(self) -> None:
        """Release what this side of the connection holds."""


class WorkerCarrier(ABC):
    """The worker side of one connection: requests in, outputs out."""

    @abstractmethod
    def decode(self, body: bytes) -> Any:
        """Rebuild a request from its frame body.

        Raises:
            reactor_runtime.model_server.shm.RequestRefused: The body is
                malformed, names memory not this session's, or is too large.
        """

    @abstractmethod
    def encode(self, output: Any) -> bytes:
        """Return an output's frame body."""

    @abstractmethod
    def close(self) -> None:
        """Release what this side holds, and what the session left behind."""


class Transport(ABC):
    """Where frames travel and how a connection's payloads are carried."""

    @abstractmethod
    def check(self, address: str) -> None:
        """Refuse a deployment this transport cannot work in.

        Raises:
            WorkerUnreachable: Saying what is missing.
        """

    @abstractmethod
    def listen(self, address: str) -> socket.socket:
        """Return a listening socket at *address*, for the worker."""

    @abstractmethod
    def connect(self, address: str) -> socket.socket:
        """Return a socket connected to the worker at *address*.

        Raises:
            NoListener: Nothing accepts there yet.
            WorkerUnreachable: The address cannot be reached from here at all.
        """

    @abstractmethod
    def worker_started(self, prefix: str) -> dict[str, str]:
        """Prepare the worker's side; return what every welcome tells a session."""

    @abstractmethod
    def worker_stopped(self) -> None:
        """Release what :meth:`worker_started` made."""

    @abstractmethod
    def worker_carrier(
        self,
        prefix: str,
        slot: int,
        token: str,
        conn_id: int,
        limit: int,
        input_bytes: int,
        output_bytes: int,
    ) -> tuple[WorkerCarrier, dict[str, str]]:
        """Return the worker's carrier for a new connection and what its welcome adds.

        The map holds ``request_prefix``, which the welcome carries as its own
        field, and whatever else the session's side needs, which the welcome
        carries in its ``transport`` map.

        Whatever the carrier holds for the session (a transport over shared
        memory: its input and output blocks, of the sizes given) is made here,
        before the session is welcomed.
        """

    @abstractmethod
    def client_carrier(self, welcome: Mapping[str, str]) -> ClientCarrier:
        """Return the session's carrier for the connection *welcome* opened.

        *welcome* is the welcome's ``transport`` map with its ``request_prefix``.

        Raises:
            WorkerUnreachable: The worker's side cannot be reached from here.
        """


class UnixShmTransport(Transport):
    """Frames on a Unix socket, arrays in ``/dev/shm``: worker and sessions on one host."""

    def __init__(self) -> None:
        self._probe: shared_memory.SharedMemory | None = None

    def check(self, address: str) -> None:
        """Refuse a missing socket directory or an unusable ``/dev/shm``."""
        directory = Path(address).parent
        if not directory.is_dir():
            raise WorkerUnreachable(
                f"the model worker's socket directory {directory} does not exist: mount the "
                "volume the worker shares with the sessions there (in a pod, the same emptyDir "
                "in every container)"
            )
        if not _SHM_DIR.is_dir():
            return
        try:
            probe = shared_memory.SharedMemory(
                name=f"rrcheck{secrets.token_hex(4)}", create=True, size=16
            )
        except OSError as exc:
            raise WorkerUnreachable(
                f"/dev/shm cannot hold a shared block: {exc}. Mount a memory-backed volume "
                "at /dev/shm (in a pod, an emptyDir with medium: Memory)"
            ) from None
        probe.close()
        probe.unlink()

    def listen(self, address: str) -> socket.socket:
        """Bind *address*, replacing a stale socket file, readable by this user alone."""
        path = Path(address)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        path.chmod(0o600)
        listener.listen(64)
        return listener

    def connect(self, address: str) -> socket.socket:
        """Connect to *address*; a missing directory is unreachable, a missing socket not yet up."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(address)
        except OSError as exc:
            sock.close()
            if exc.errno == errno.ENOENT and not Path(address).parent.is_dir():
                self.check(address)
            if exc.errno in (errno.EACCES, errno.EPERM):
                raise WorkerUnreachable(
                    f"the model worker's socket {address} is not open to this user: run the "
                    "sessions as the worker's user"
                ) from None
            raise NoListener(str(exc)) from None
        return sock

    def worker_started(self, prefix: str) -> dict[str, str]:
        """Make the block every session reads at connect to prove it shares ``/dev/shm``."""
        nonce = secrets.token_bytes(8)
        name = created_here(f"{prefix}probe")
        self._probe = shared_memory.SharedMemory(name=name, create=True, size=len(nonce))
        buf = self._probe.buf
        assert buf is not None
        buf[: len(nonce)] = nonce
        return {"probe": name, "probe_nonce": nonce.hex()}

    def worker_stopped(self) -> None:
        """Unlink the probe block."""
        if self._probe is not None:
            release(self._probe)
            with contextlib.suppress(OSError):
                self._probe.unlink()
            self._probe = None

    def worker_carrier(
        self,
        prefix: str,
        slot: int,
        token: str,
        conn_id: int,
        limit: int,
        input_bytes: int,
        output_bytes: int,
    ) -> tuple[WorkerCarrier, dict[str, str]]:
        """Make and zero-fill this session's input and output blocks, before it is welcomed.

        Filling them here charges their memory to the worker, which made them,
        and a ``/dev/shm`` too small for one more session refuses it now
        rather than failing a step later.

        Raises:
            reactor_runtime.distributed.SharedSlotAllocationFailed: ``/dev/shm``
                cannot hold the two blocks.
        """
        request_prefix = f"{prefix}s{slot}_{token}_"
        result_prefix = created_here(f"{prefix}w{slot}c{conn_id}_")
        input_block = _allocate(created_here(f"{request_prefix}in"), input_bytes)
        try:
            output_block = _allocate(f"{result_prefix}out", output_bytes)
        except BaseException:
            _unlink(input_block)
            raise
        carrier = _WorkerShm(
            ViewReader(request_prefix, limit),
            request_prefix,
            result_prefix,
            input_block,
            _Blocks(output_block, f"{result_prefix}g", owned=True),
        )
        return carrier, {"request_prefix": request_prefix, "input_block": input_block.name}

    def client_carrier(self, welcome: Mapping[str, str]) -> ClientCarrier:
        """Check the worker's probe block is visible here, then carry under the welcome's prefix."""
        probe, nonce = welcome.get("probe"), welcome.get("probe_nonce")
        if isinstance(probe, str) and isinstance(nonce, str):
            expected = bytes.fromhex(nonce)
            try:
                shm = attach(probe)
            except PermissionError:
                raise WorkerUnreachable(
                    "the model worker's shared memory is not open to this user: run the "
                    "sessions as the worker's user"
                ) from None
            except OSError:
                raise WorkerUnreachable(
                    "the model worker's shared memory is not visible here: the worker and the "
                    "sessions must share /dev/shm (one host; in a pod, the same emptyDir with "
                    "medium: Memory mounted at /dev/shm in every container)"
                ) from None
            buf = shm.buf
            seen = bytes(buf[: len(expected)]) if buf is not None else b""
            release(shm)
            if seen != expected:
                raise WorkerUnreachable(
                    "the shared block the model worker made reads differently here: the worker "
                    "and the sessions do not share one /dev/shm"
                )
        return _ClientShm(str(welcome["request_prefix"]), str(welcome["input_block"]))


class _Blocks:
    """Where one direction's payloads are written: one block, grown to the largest.

    A payload that fits goes into the block; one that does not replaces it with
    a block its size, made by this side, which every later payload uses. The
    reader is done with the last payload before the next is written: an output
    is copied out before the session sends again, and a request is answered
    before the next one is sent. Only an abandoned request (its step timed out)
    may still be read, so :meth:`abandon` sends the next payload to a block of
    its own, and :meth:`settled` returns to the main block once an answer shows
    the reader is done.
    """

    def __init__(self, base: shared_memory.SharedMemory, grow_prefix: str, *, owned: bool) -> None:
        self._base = base
        self._owned = owned
        self._grow_prefix = created_here(grow_prefix)
        self._grown = 0
        self._abandoned = False
        self._extra: shared_memory.SharedMemory | None = None
        self._target = base
        self._cursor = 0

    def prepare(self, nbytes: int) -> None:
        """Choose where the next payload, of *nbytes* in its buffers, is written."""
        if self._extra is not None:
            _unlink(self._extra)
            self._extra = None
        if self._abandoned:
            self._extra = self._make(max(nbytes, 1))
            self._target = self._extra
            return
        if nbytes > self._base.size:
            replacement = self._make(nbytes)
            self._drop_base()
            self._base, self._owned = replacement, True
        self._target = self._base

    def settled(self) -> None:
        """The reader is done with everything written so far."""
        self._abandoned = False

    @property
    def name(self) -> str:
        return self._target.name

    def begin(self) -> None:
        self._cursor = 0

    def write(self, buffer: Any) -> tuple[int, int]:
        raw = memoryview(buffer).cast("B")
        end = self._cursor + raw.nbytes
        buf = self._target.buf
        assert buf is not None
        if end > self._target.size:
            raise RuntimeError("a payload outgrew the block prepared for it")
        buf[self._cursor : end] = raw
        span = (self._cursor, raw.nbytes)
        self._cursor = end
        return span

    def abandon(self) -> None:
        """Write nothing more where the last payload went, until :meth:`settled`."""
        self._abandoned = True

    def close(self) -> None:
        if self._extra is not None:
            _unlink(self._extra)
            self._extra = None
        self._drop_base()
        forget_created(self._grow_prefix)

    def _make(self, size: int) -> shared_memory.SharedMemory:
        self._grown += 1
        return _allocate(f"{self._grow_prefix}{self._grown}", size, fill=False)

    def _drop_base(self) -> None:
        if self._owned:
            _unlink(self._base)
        else:
            release(self._base)


class _ClientShm(ClientCarrier):
    def __init__(self, prefix: str, input_block: str) -> None:
        self._request = _Blocks(attach(input_block), f"{prefix}g", owned=False)
        self._mapping: dict[str, Any] = {}

    def size(self, payload: Any) -> int:
        return codec.size(payload)

    def encode(self, payload: Any) -> tuple[bytes, int]:
        self._request.prepare(codec.size(payload))
        return codec.encode(payload, self._request)

    def decode(self, body: bytes) -> Any:
        return copy_unpack(body, self._mapping)

    def renew(self) -> None:
        self._request.abandon()

    def settled(self) -> None:
        self._request.settled()

    def close(self) -> None:
        close_mapping(self._mapping)
        self._request.close()


class _WorkerShm(WorkerCarrier):
    def __init__(
        self,
        reader: ViewReader,
        request_prefix: str,
        result_prefix: str,
        input_block: shared_memory.SharedMemory,
        output: _Blocks,
    ) -> None:
        self._reader = reader
        self._request_prefix = request_prefix
        self._result_prefix = result_prefix
        self._input = input_block
        self._output = output

    def decode(self, body: bytes) -> Any:
        return self._reader.unpack(body)

    def encode(self, output: Any) -> bytes:
        self._output.prepare(codec.size(output))
        header, _ = codec.encode(output, self._output)
        return header

    def close(self) -> None:
        self._reader.close()
        self._output.close()
        _unlink(self._input)
        # Blocks the session made once a request outgrew its input block, which
        # a session process that was killed never unlinked.
        unlink_prefix(self._request_prefix)
        forget_created(self._result_prefix)
        forget_created(f"{self._request_prefix}in")


def _allocate(name: str, size: int, *, fill: bool = True) -> shared_memory.SharedMemory:
    """Create block *name* and reserve its pages, so a full ``/dev/shm`` is an error here.

    ``posix_fallocate`` reserves the pages (charging them to this process) and
    answers ENOSPC when they do not fit, where a write to a page tmpfs cannot
    back would kill the process with SIGBUS. *fill* also touches every page.

    Raises:
        SharedSlotAllocationFailed: ``/dev/shm`` cannot hold the block.
    """
    try:
        block = shared_memory.SharedMemory(name=name, create=True, size=size)
    except OSError as exc:
        raise SharedSlotAllocationFailed(f"cannot make a {size} byte shared block: {exc}") from None
    try:
        fd = getattr(block, "_fd", -1)
        if fd >= 0 and hasattr(os, "posix_fallocate"):
            os.posix_fallocate(fd, 0, size)
        if fill:
            buf = block.buf
            assert buf is not None
            np.frombuffer(buf, dtype=np.uint8)[:] = 0
    except OSError as exc:
        _unlink(block)
        raise SharedSlotAllocationFailed(
            f"/dev/shm cannot hold a {size} byte block: {exc}"
        ) from None
    except BaseException:
        _unlink(block)
        raise
    return block


def _unlink(block: shared_memory.SharedMemory) -> None:
    release(block)
    with contextlib.suppress(OSError):
        block.unlink()
