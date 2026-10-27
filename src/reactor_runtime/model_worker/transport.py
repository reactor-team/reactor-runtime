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
import secrets
import socket
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

from reactor_runtime.distributed.errors import SharedSlotAllocationFailed
from reactor_runtime.distributed.ipc import SharedSlot
from reactor_runtime.model_worker import codec
from reactor_runtime.model_worker.shm import (
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
    def encode(self, payload: Any) -> tuple[bytes, int]:
        """Return a request's frame body and the bytes it carries in all."""

    @abstractmethod
    def decode(self, body: bytes) -> Any:
        """Rebuild an output from its frame body, as an object this session owns."""

    @abstractmethod
    def renew(self) -> None:
        """Stop writing where an abandoned request may still be read."""

    @abstractmethod
    def close(self) -> None:
        """Release what this side of the connection holds."""


class WorkerCarrier(ABC):
    """The worker side of one connection: requests in, outputs out."""

    @abstractmethod
    def decode(self, body: bytes) -> Any:
        """Rebuild a request from its frame body.

        Raises:
            reactor_runtime.model_worker.shm.RequestRefused: The body is
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
    def worker_started(self, prefix: str) -> dict[str, Any]:
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
    ) -> tuple[WorkerCarrier, dict[str, Any]]:
        """Return the worker's carrier for a new connection and what its welcome adds.

        Whatever the carrier holds for the session (a transport over shared
        memory: its input and output blocks, of the sizes given) is made here,
        before the session is welcomed.
        """

    @abstractmethod
    def client_carrier(self, welcome: dict[str, Any]) -> ClientCarrier:
        """Return the session's carrier for the connection *welcome* opened.

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
            raise NoListener(str(exc)) from None
        return sock

    def worker_started(self, prefix: str) -> dict[str, Any]:
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
    ) -> tuple[WorkerCarrier, dict[str, Any]]:
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
            output = SharedSlot(prefix=result_prefix, initial_bytes=output_bytes)
        except BaseException:
            release(input_block)
            with contextlib.suppress(OSError):
                input_block.unlink()
            raise
        _zero_fill(output._shm)
        carrier = _WorkerShm(
            ViewReader(request_prefix, limit), request_prefix, result_prefix, input_block, output
        )
        return carrier, {"request_prefix": request_prefix, "input_block": input_block.name}

    def client_carrier(self, welcome: dict[str, Any]) -> ClientCarrier:
        """Check the worker's probe block is visible here, then carry under the welcome's prefix."""
        probe, nonce = welcome.get("probe"), welcome.get("probe_nonce")
        if isinstance(probe, str) and isinstance(nonce, str):
            expected = bytes.fromhex(nonce)
            try:
                shm = attach(probe)
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
        return _ClientShm(created_here(str(welcome["request_prefix"])), str(welcome["input_block"]))


class _InputSlot:
    """Writes requests into the input block the worker made, or into its own once outgrown."""

    def __init__(self, block: str, grow_prefix: str) -> None:
        self._block = attach(block)
        self._grow_prefix = grow_prefix
        self._grown = 0
        self._own: SharedSlot | None = None
        self._cursor = 0

    @property
    def name(self) -> str:
        return self._own.name if self._own is not None else self._block.name

    def begin(self) -> None:
        self._cursor = 0
        if self._own is not None:
            self._own.begin()

    def write(self, buffer: Any) -> tuple[int, int]:
        if self._own is not None:
            return self._own.write(buffer)
        raw = memoryview(buffer).cast("B")
        end = self._cursor + raw.nbytes
        buf = self._block.buf
        assert buf is not None
        if end > self._block.size:
            # Outgrown: carry on in a block of this session's own, bytes so far included.
            own = self._move(max(end, 2 * self._block.size))
            own.write(buf[: self._cursor])
            return own.write(raw)
        buf[self._cursor : end] = raw
        span = (self._cursor, raw.nbytes)
        self._cursor = end
        return span

    def renew(self) -> None:
        """Write the next request elsewhere: the worker may still read the abandoned one."""
        self._move(self._block.size)

    def close(self) -> None:
        release(self._block)
        if self._own is not None:
            self._own.close()

    def _move(self, size: int) -> SharedSlot:
        if self._own is not None:
            self._own.close()
        self._grown += 1
        self._own = SharedSlot(prefix=f"{self._grow_prefix}{self._grown}_", initial_bytes=size)
        self._own.begin()
        return self._own


class _ClientShm(ClientCarrier):
    def __init__(self, prefix: str, input_block: str) -> None:
        self._prefix = prefix
        self._request = _InputSlot(input_block, prefix)
        self._mapping: dict[str, Any] = {}

    def encode(self, payload: Any) -> tuple[bytes, int]:
        return codec.encode(payload, self._request)

    def decode(self, body: bytes) -> Any:
        return copy_unpack(body, self._mapping)

    def renew(self) -> None:
        self._request.renew()

    def close(self) -> None:
        close_mapping(self._mapping)
        self._request.close()
        forget_created(self._prefix)


class _WorkerShm(WorkerCarrier):
    def __init__(
        self,
        reader: ViewReader,
        request_prefix: str,
        result_prefix: str,
        input_block: shared_memory.SharedMemory,
        output: SharedSlot,
    ) -> None:
        self._reader = reader
        self._request_prefix = request_prefix
        self._result_prefix = result_prefix
        self._input = input_block
        self._result: SharedSlot | None = output

    def decode(self, body: bytes) -> Any:
        return self._reader.unpack(body)

    def encode(self, output: Any) -> bytes:
        if self._result is None:
            self._result = SharedSlot(prefix=self._result_prefix)
        header, _ = codec.encode(output, self._result)
        return header

    def close(self) -> None:
        self._reader.close()
        if self._result is not None:
            self._result.close()
            self._result = None
        release(self._input)
        with contextlib.suppress(OSError):
            self._input.unlink()
        # Blocks the session made once a request outgrew its input block, which
        # a session process that was killed never unlinked.
        unlink_prefix(self._request_prefix)
        forget_created(self._result_prefix)
        forget_created(f"{self._request_prefix}in")


def _allocate(name: str, size: int) -> shared_memory.SharedMemory:
    block = shared_memory.SharedMemory(name=name, create=True, size=size)
    try:
        _zero_fill(block)
    except BaseException:
        release(block)
        with contextlib.suppress(OSError):
            block.unlink()
        raise
    return block


def _zero_fill(block: shared_memory.SharedMemory) -> None:
    """Touch every page of *block*, so tmpfs charges it to this process now.

    A page of a full ``/dev/shm`` raises SIGBUS when first touched; checking the
    free space first turns that into an error instead.
    """
    buf = block.buf
    assert buf is not None
    if _SHM_DIR.is_dir():
        import shutil

        free = shutil.disk_usage(_SHM_DIR).free
        if free < block.size:
            raise SharedSlotAllocationFailed(
                f"/dev/shm has {free} bytes free, not the {block.size} one session's block needs"
            )
    np.frombuffer(buf, dtype=np.uint8)[:] = 0
