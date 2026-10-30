"""The link over a Unix socket, with every array in memory both ends share.

A session runtime and the model server on one host share a memory-backed
directory per session: the session sees it as its ``/dev/shm``, the model
server under a directory of its own for that slot. A session's
:class:`~reactor_runtime.model_server.protocol.Step` and the model server's
:class:`~reactor_runtime.model_server.protocol.Result` write their arrays into
a block their sender creates in that directory
(:class:`~reactor_runtime.ipc.SharedSlot`), and only the small pickled header
crosses the socket; the other end copies the arrays out
(:func:`~reactor_runtime.ipc.unpack`) as soon as it reads the header. The
arrays a model's ``step()`` receives are its own.

Every other message, a :class:`~reactor_runtime.model_server.protocol.Signal`
included, is pickled whole into its frame, under the frame limit: a signal
carries small values, and large data belongs in a step. That keeps one message
in a block at a time in each direction, with no acknowledgement. A session
sends its next step only once the previous one is answered or cancelled, and
a session has one request at a time on the model server, so a result is
written only after the previous one was sent; the one overlap, a late result
for a cancelled step followed by the next step's, is a result the client
discards.

One directory per session keeps a fault in one session in that session: a
session that leaks or writes past its blocks fills only its own volume, never
another session's or the model server's.

Each end creates the blocks it writes, and their pages count toward the memory
of the process that creates them: the session creates its request block when it
connects (:meth:`UnixShmTransport.connect`), the model server creates the slot's
output block when it sets the link up. A block is created at the size
``reactor.yaml`` declares, ``runtime.worker.shm_input_bytes`` for a request and
``runtime.worker.shm_output_bytes`` for a result, and never grows. A message
whose arrays do not fit its block is refused before anything is sent, with
:class:`~reactor_runtime.model_server.PayloadTooLarge` naming both sizes and
the key to raise, and so is one whose header, the part pickle keeps inline
(a torch tensor, a non-contiguous array), passes the frame limit; the request
fails alone and the link goes on.

A slot holds a second request block in one case. When the session sends a
:class:`~reactor_runtime.model_server.protocol.Cancel`, it moves to a fresh
block of the same size and keeps the old one, still mapped and on disk, as its
retired block: the model server reads a block only when it reads the header
that names it, and may not have reached the abandoned step's header yet. The
retired block is released at the next rotation, before its replacement is
allocated, or when the link closes. A model server that reads a step only
after two cancels (its reader stalled while both were sent) finds its block
gone: it skips that step, which its session already cancelled. The model
server never cancels, so its output block is never replaced.

The model server's end releases a link's blocks as soon as the link closes,
without waiting for a step still running for its session: that step's arrays
are its own, and its result, sent on the closed link, is dropped. It does the
same for the previous link of a slot before a new link allocates there, since
a slot belongs to one session. So a session runtime that restarts during a
long step finds its slot's directory empty. The setup probe below is gone
before the session allocates its first request block. A slot's directory
therefore holds at most two request blocks and one output block, in whole
64 KiB pages: :func:`~reactor_runtime.core.shm_slot_bytes`, the size a
deployment gives each slot's volume.

Blocks are named by who writes them: ``<prefix>s<slot>_`` for a session's
requests and ``<prefix>w<slot>_`` for the model server's outputs, where the
prefix is the deployment's (``rrms_`` by default). A process that starts
again clears what its previous run left behind under its own names: a session
runtime, the blocks of its slot; the model server, every output block, in every
slot's directory (:meth:`UnixShmTransport.reclaim`). A process killed before it
could release its blocks leaks nothing past its next start.

Before the protocol's first message the two ends set the link up themselves.
The session names its slot and the prefix of its request blocks; the model
server answers with a probe block it wrote in the slot's directory, and the
session reads it from its own: a deployment where the two do not see the same
memory fails at once, with :class:`~reactor_runtime.model_server.WorkerUnreachable`
saying so, instead of on the first step. The session removes the probe once it
has read it; the model server removes it too, on the link's first message or
when the link closes, for a session that never got that far.
"""

from __future__ import annotations

import contextlib
import errno
import pickle
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reactor_runtime.core import DEFAULT_SHM_INPUT_BYTES, DEFAULT_SHM_OUTPUT_BYTES
from reactor_runtime.ipc import (
    SharedSlot,
    SharedSlotAllocationFailed,
    SharedSlotLimitExceeded,
    SlotReader,
    attach_block,
    create_block,
    default_directory,
    new_prefix,
    pack,
    unlink_blocks,
    unpack,
)
from reactor_runtime.log import get_logger
from reactor_runtime.model_server.errors import PayloadTooLarge, WorkerUnreachable
from reactor_runtime.model_server.link import DEFAULT_FRAME_LIMIT, NoListener, StreamLink
from reactor_runtime.model_server.protocol import (
    Cancel,
    Closed,
    ProtocolError,
    Result,
    Step,
    check_message,
)

logger = get_logger(__name__)

SLOT_PLACEHOLDER = "{slot}"

DEFAULT_BLOCK_PREFIX = "rrms_"
"""The prefix of every block a split model's processes create, when the deployment names none."""


class _BlockGoneError(Exception):
    """The block a header names is not in the directory any more."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _LinkReader(SlotReader):
    """A slot reader that tells a missing block apart from any other failure."""

    def attach(self, name: str) -> None:
        try:
            super().attach(name)
        except FileNotFoundError:
            raise _BlockGoneError(name) from None


@dataclass(frozen=True)
class ShmHello:
    """The session's first frame: its slot, and the prefix of the blocks it writes."""

    slot: int
    request_prefix: str


@dataclass(frozen=True)
class ShmSetup:
    """The model server's answer: the probe it wrote, and the size of a request block.

    Attributes:
        probe: The name of a block in the slot's directory.
        nonce: The bytes written into it, which the session must read back.
        input_bytes: The size of the session's request block, which never grows.
    """

    probe: str
    nonce: bytes
    input_bytes: int


@dataclass(frozen=True)
class ShmRefused:
    """The model server cannot set the link up for this slot, and why."""

    reason: str


class ShmLink:
    """One session's link, on either end, carrying each message's arrays in shared memory.

    Build one with :meth:`UnixShmTransport.connect` (a session's end) or
    :meth:`UnixShmListener.accept` (the model server's end). The model server's
    end sets itself up on its first :meth:`receive`.
    """

    def __init__(
        self,
        stream: StreamLink,
        *,
        writer: SharedSlot | None,
        directory: Path | None,
        server: UnixShmTransport | None = None,
    ) -> None:
        """Hold the stream and the blocks; use the transport to build one."""
        self._stream = stream
        self._writer = writer
        self._base_prefix = writer.prefix if writer is not None else None
        self._retired: SharedSlot | None = None
        self._rotations = 0
        self._directory = directory
        self._reader = _LinkReader(directory) if directory is not None else None
        self._server = server
        self._peer_prefix: str | None = None
        self._probe: str | None = None
        self._slot: int | None = None
        self._released = False
        self._lock = threading.Lock()
        # Held while a message's arrays are copied out, so a release from
        # another thread never unmaps a block mid-copy.
        self._read_lock = threading.Lock()

    def send(self, message: Any) -> None:
        """Send *message*, a step's or a result's arrays through this end's block.

        Every other message travels whole in its frame. A
        :class:`~reactor_runtime.model_server.protocol.Cancel` also moves this
        end to a new block (see the module docstring): the replacement is
        allocated before the cancel is sent, and the block in use becomes the
        retired one only once it was.

        Raises:
            PayloadTooLarge: The message's arrays do not fit this end's block,
                or its header passes the frame limit. Nothing was sent, and the
                link goes on.
            SharedSlotAllocationFailed: A cancel's replacement block cannot be
                allocated; the cancel was not sent, and the link keeps its
                block.
            Closed: The link is closed.
            OSError: The connection failed.
        """
        check_message(message)
        with self._lock:
            if self._writer is None:
                raise Closed(
                    "the link is closed" if self._released else "the link is not set up yet"
                )
            uses_block = isinstance(message, Step | Result)
            try:
                header = pack(message, self._writer if uses_block else None)
            except SharedSlotLimitExceeded as exc:
                raise PayloadTooLarge(self._arrays_too_large(exc)) from None
            limit = self._stream.frame_limit
            if len(header) > limit:
                raise PayloadTooLarge(self._header_too_large(message, len(header), limit))
            if isinstance(message, Cancel):
                self._send_cancel(header)
            else:
                self._stream.send_bytes(header)

    def _send_cancel(self, header: bytes) -> None:
        writer = self._writer
        assert writer is not None
        assert self._base_prefix is not None
        # Release the block a previous cancel retired before allocating the
        # replacement, so a slot never holds more than two request blocks.
        if self._retired is not None:
            self._retired.close()
            self._retired = None
        self._rotations += 1
        replacement = _fixed_slot(
            f"{self._base_prefix}r{self._rotations}_", writer.capacity, self._directory
        )
        try:
            self._stream.send_bytes(header)
        except BaseException:
            replacement.close()
            raise
        self._retired, self._writer = writer, replacement

    def receive(self) -> Any:
        """Block for the next message, its arrays copied out of the peer's block.

        On the model server's end, a step whose block the session already
        released is skipped: the session releases a request block only after
        it cancelled the steps written to it (see the module docstring). Once
        the peer closes the link, the model server's end releases its blocks
        at once (:meth:`release_blocks`).

        Raises:
            Closed: The peer or this end closed the link.
            ProtocolError: The peer sent something that is not a message, or a
                header that names a block that is not there (on the session's
                end).
            OSError: The connection failed.
        """
        if self._server is not None and self._reader is None:
            if self._released:
                raise Closed("the link is closed")
            self._set_up()
        while True:
            try:
                header = self._stream.receive_bytes()
            except (Closed, OSError):
                if self._server is not None:
                    self.release_blocks()
                raise
            if self._probe is not None:
                self._drop_probe()
            try:
                with self._read_lock:
                    if self._reader is None:
                        raise Closed("the link is closed")
                    message = unpack(header, self._reader)
            except _BlockGoneError as gone:
                if self._server is not None:
                    logger.debug("skipped a step whose request block is gone", block=gone.name)
                    continue
                raise ProtocolError(
                    f"a message names the block {gone.name}, which is gone"
                ) from None
            except (OSError, ValueError) as exc:
                raise ProtocolError(f"a message's arrays cannot be read: {exc}") from None
            except pickle.UnpicklingError as exc:
                raise ProtocolError(f"a frame does not hold a message: {exc}") from None
            return check_message(message)

    def close(self) -> None:
        """Close the socket and release this end's blocks. Idempotent.

        The model server's end also reclaims the request blocks a session that
        died left behind.
        """
        self._stream.close()
        self.release_blocks()

    def release_blocks(self) -> None:
        """Release every block this end holds, the socket left as it is. Idempotent.

        The model server's end calls it as soon as its session's link closes,
        and for the previous link of a slot before a new one allocates there:
        a step still running for that session keeps its arrays, which it
        copied out, and its result, sent on this link, raises ``Closed``.
        """
        with self._lock:
            writer, self._writer = self._writer, None
            retired, self._retired = self._retired, None
            self._released = True
        for block in (writer, retired):
            if block is not None:
                block.close()
        with self._read_lock:
            reader, self._reader = self._reader, None
        if reader is not None:
            reader.close()
        if self._probe is not None:
            self._drop_probe()
        if self._peer_prefix is not None and self._directory is not None:
            unlink_blocks(self._peer_prefix, self._directory)
        if self._server is not None and self._slot is not None:
            self._server._let_go(self._slot, self)

    def _arrays_too_large(self, exc: SharedSlotLimitExceeded) -> str:
        if self._server is not None:
            what, key = "result", "shm_output_bytes"
        else:
            what, key = "request", "shm_input_bytes"
        return (
            f"a {what}'s arrays hold {exc.needed} bytes, above the {exc.limit} bytes of its "
            f"shared-memory block: raise runtime.worker.{key} in reactor.yaml"
        )

    def _header_too_large(self, message: Any, size: int, limit: int) -> str:
        if isinstance(message, Step | Result):
            what = "request" if isinstance(message, Step) else "result"
            return (
                f"a {what} pickled {size} bytes inline, above the {limit}-byte frame limit: "
                "only contiguous numpy arrays travel through shared memory; pass "
                "np.ascontiguousarray(x), or np.asarray(t.cpu()) for a tensor"
            )
        return (
            f"a {type(message).__name__} pickled {size} bytes, above the {limit}-byte frame "
            "limit: a signal carries small values, and large data belongs in a step"
        )

    def _set_up(self) -> None:
        assert self._server is not None
        try:
            hello = pickle.loads(self._stream.receive_bytes())
        except (Closed, OSError, ProtocolError):
            raise
        except Exception as exc:
            raise ProtocolError(
                f"a session opened the link with an unreadable frame: {exc!r}"
            ) from None
        if not isinstance(hello, ShmHello) or type(hello.slot) is not int or hello.slot < 0:
            raise ProtocolError("a session opened the link without a valid ShmHello")
        expected = self._server.request_prefix(hello.slot)
        prefix = hello.request_prefix
        if (
            not isinstance(prefix, str)
            or "/" in prefix
            or "\0" in prefix
            or not prefix.startswith(expected)
            or len(prefix) <= len(expected)
        ):
            self._stream.send_bytes(
                pickle.dumps(
                    ShmRefused(
                        f"the session announced an invalid request prefix {prefix!r}: it must "
                        f"be a plain name that starts with {expected!r} and is longer than it"
                    )
                )
            )
            raise ProtocolError(f"slot {hello.slot} announced an invalid request prefix")
        directory = self._server.slot_directory(hello.slot)
        if not directory.is_dir():
            self._stream.send_bytes(
                pickle.dumps(
                    ShmRefused(
                        f"slot {hello.slot} has no shared-memory directory at {directory}: mount "
                        "the session's memory-backed volume there in the model server, and at "
                        "the session's shared-memory directory in the session"
                    )
                )
            )
            raise ProtocolError(f"slot {hello.slot} has no shared-memory directory")
        self._directory = directory
        self._reader = _LinkReader(directory)
        self._peer_prefix = prefix
        self._slot = hello.slot
        nonce = secrets.token_bytes(8)
        # Claiming the slot and allocating in it are one step per slot: a link
        # for the same slot that sets up meanwhile waits, then releases this
        # one's blocks whole, so a slot never holds two links' output blocks.
        with self._server._slot_lock(hello.slot):
            # A slot belongs to one session: a new link for it means the
            # previous one's session is gone, so its blocks go first.
            previous = self._server._hold(hello.slot, self)
            if previous is not None:
                previous.release_blocks()
            try:
                self._writer = _fixed_slot(
                    f"{self._server.output_prefix(hello.slot)}{new_prefix()}",
                    self._server.output_bytes,
                    directory,
                )
                probe = create_block(f"{self._writer.prefix}probe", len(nonce), directory)
            except SharedSlotAllocationFailed as exc:
                if self._writer is not None:
                    self._writer.close()
                    self._writer = None
                failure = exc
            else:
                failure = None
        if failure is not None:
            self._stream.send_bytes(
                pickle.dumps(ShmRefused(f"no shared memory for slot {hello.slot}: {failure}"))
            )
            raise ProtocolError(f"slot {hello.slot} has no room for its blocks") from None
        probe.buf[: len(nonce)] = nonce
        probe.close()
        self._probe = probe.name
        self._stream.send_bytes(pickle.dumps(ShmSetup(probe.name, nonce, self._server.input_bytes)))

    def _drop_probe(self) -> None:
        name, self._probe = self._probe, None
        if name is not None and self._directory is not None:
            with contextlib.suppress(OSError):
                (self._directory / name).unlink()


class UnixShmListener:
    """The model server's end of a :class:`UnixShmTransport`: one link per accepted session."""

    def __init__(self, sock: socket.socket, path: Path, transport: UnixShmTransport) -> None:
        """Hold a bound, listening socket; use :meth:`UnixShmTransport.listen`."""
        self._sock = sock
        self._path = path
        self._transport = transport
        self._identity = _identity(path)

    def accept(self) -> ShmLink:
        """Block for the next session and return its link; it sets itself up on its first receive.

        Raises:
            OSError: The listener is closed.
        """
        sock, _ = self._sock.accept()
        stream = StreamLink(sock, frame_limit=self._transport.frame_limit)
        return ShmLink(stream, writer=None, directory=None, server=self._transport)

    def close(self) -> None:
        """Stop accepting, and remove the socket file if it is still this listener's.

        Links already accepted stay open. A socket file another listener has
        bound at the same path since is left in place.
        """
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
        with contextlib.suppress(OSError):
            if _identity(self._path) == self._identity:
                self._path.unlink()


class UnixShmTransport:
    """Links over a Unix socket, with every array in a memory-backed directory per session.

    Args:
        slot_directory: Where the model server finds each slot's directory, a
            path with ``{slot}`` in it, such as ``/run/reactor-shm/s{slot}``.
            Needed to :meth:`listen`; a session runtime that only connects
            leaves it out.
        session_directory: Where a session runtime finds its own;
            :func:`~reactor_runtime.ipc.default_directory` (``/dev/shm``) when
            omitted. It is the same memory as the model server's directory for
            the session's slot, which the link checks when it opens.
        input_bytes: The size of each request block,
            ``runtime.worker.shm_input_bytes``. A request whose arrays need
            more is refused.
        output_bytes: The size of each output block,
            ``runtime.worker.shm_output_bytes``. A result whose arrays need
            more is refused.
        frame_limit: The longest frame a link sends or accepts on the socket;
            the same on both ends of a deployment.
        block_prefix: The prefix of every block both ends create; the same
            on both ends of a deployment.

    Raises:
        ValueError: *slot_directory* has no ``{slot}`` in it.
    """

    def __init__(
        self,
        *,
        slot_directory: str | None = None,
        session_directory: Path | None = None,
        input_bytes: int = DEFAULT_SHM_INPUT_BYTES,
        output_bytes: int = DEFAULT_SHM_OUTPUT_BYTES,
        frame_limit: int = DEFAULT_FRAME_LIMIT,
        block_prefix: str = DEFAULT_BLOCK_PREFIX,
    ) -> None:
        if slot_directory is not None and SLOT_PLACEHOLDER not in slot_directory:
            raise ValueError(
                f"the per-slot shared-memory path {slot_directory!r} has no {SLOT_PLACEHOLDER}"
            )
        self._slot_directory = slot_directory
        self._session_directory = session_directory
        self.input_bytes = input_bytes
        self.output_bytes = output_bytes
        self.frame_limit = frame_limit
        self.block_prefix = block_prefix
        self._held: dict[int, ShmLink] = {}
        self._held_lock = threading.Lock()
        self._slot_locks: dict[int, threading.Lock] = {}

    def _slot_lock(self, slot: int) -> threading.Lock:
        """The lock that makes a link's claim of *slot* and its allocation there one step."""
        with self._held_lock:
            return self._slot_locks.setdefault(slot, threading.Lock())

    def _hold(self, slot: int, link: ShmLink) -> ShmLink | None:
        """Make *link* the model server's link for *slot*; return the one it replaces."""
        with self._held_lock:
            previous = self._held.get(slot)
            self._held[slot] = link
        return previous if previous is not link else None

    def _let_go(self, slot: int, link: ShmLink) -> None:
        with self._held_lock:
            if self._held.get(slot) is link:
                del self._held[slot]

    def request_prefix(self, slot: int) -> str:
        """The name every request block of slot *slot* starts with."""
        return f"{self.block_prefix}s{slot}_"

    def output_prefix(self, slot: int) -> str:
        """The name every output block of slot *slot* starts with."""
        return f"{self.block_prefix}w{slot}_"

    def reclaim(self, slots: int) -> int:
        """Unlink the output blocks a previous model server left in slots ``0`` to *slots* - 1.

        For a model server that starts: no link is open yet, so every output
        block in a slot's directory is a previous run's. Returns how many were
        unlinked.

        Raises:
            ValueError: The transport was built without a ``slot_directory``.
        """
        count = 0
        for slot in range(slots):
            directory = self.slot_directory(slot)
            if directory.is_dir():
                count += unlink_blocks(self.output_prefix(slot), directory)
        return count

    def slot_directory(self, slot: int) -> Path:
        """Return where the model server keeps slot *slot*'s blocks.

        Raises:
            ValueError: The transport was built without a ``slot_directory``.
        """
        if self._slot_directory is None:
            raise ValueError("this transport has no slot_directory: it can connect, not listen")
        return Path(self._slot_directory.replace(SLOT_PLACEHOLDER, str(slot)))

    def listen(self, address: str) -> UnixShmListener:
        """Bind *address*, open to this user alone, replacing a stale socket file only.

        A socket file something still accepts on is another model server's,
        and is never taken over.

        Raises:
            WorkerUnreachable: The socket's directory does not exist, or a model
                server already listens at *address*.
            ValueError: The transport was built without a ``slot_directory``.
        """
        if self._slot_directory is None:
            raise ValueError("this transport has no slot_directory: it can connect, not listen")
        path = Path(address)
        if not path.parent.is_dir():
            raise WorkerUnreachable(_missing_directory(path.parent))
        if path.exists() or path.is_symlink():
            if _accepts(address):
                raise WorkerUnreachable(f"a model server already listens at {address}")
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        path.chmod(0o600)
        sock.listen(64)
        return UnixShmListener(sock, path, self)

    def connect(self, address: str, slot: int, *, timeout: float | None = None) -> ShmLink:
        """Open the link for the session in *slot*, and check the two ends share memory.

        Blocks no longer than *timeout* seconds in total, the setup exchange
        with the model server included; ``None`` is no bound.

        Raises:
            NoListener: Nothing accepts at *address* yet, or the model server
                did not finish setting the link up within *timeout*.
            WorkerUnreachable: The socket's directory does not exist, the
                socket is not open to this user, the model server has no
                directory for the slot, the two ends do not see the same
                memory, or the session's directory has no room for its
                request block.
        """
        path = Path(address)
        deadline = None if timeout is None else time.monotonic() + timeout
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(_remaining(deadline))
            sock.connect(address)
        except OSError as exc:
            sock.close()
            if exc.errno == errno.ENOENT and not path.parent.is_dir():
                raise WorkerUnreachable(_missing_directory(path.parent)) from None
            if exc.errno in (errno.EACCES, errno.EPERM):
                raise WorkerUnreachable(
                    f"the model server's socket {address} is not open to this user: run the "
                    "session runtimes as the model server's user"
                ) from None
            raise NoListener(f"nothing accepts at {address}: {exc}") from None
        stream = StreamLink(sock, frame_limit=self.frame_limit)
        directory = self._session_directory or default_directory()
        # This process serves the slot alone, so a request block of the slot that
        # is already there is a previous run's.
        unlink_blocks(self.request_prefix(slot), directory)
        prefix = f"{self.request_prefix(slot)}{new_prefix()}"
        try:
            try:
                sock.settimeout(_remaining(deadline))
            except TimeoutError:
                raise NoListener(_SETUP_LATE) from None
            setup = self._handshake(stream, slot, prefix, directory, deadline)
            try:
                writer = _fixed_slot(prefix, setup.input_bytes, directory)
            except SharedSlotAllocationFailed:
                raise WorkerUnreachable(
                    f"slot {slot}'s shared-memory directory {directory} has no room for a "
                    f"{setup.input_bytes}-byte request block: size the session's volume to at "
                    "least 2 * shm_input_bytes + shm_output_bytes, in whole 64 KiB pages"
                ) from None
        except BaseException:
            stream.close()
            raise
        sock.settimeout(None)
        return ShmLink(stream, writer=writer, directory=directory)

    def _handshake(
        self,
        stream: StreamLink,
        slot: int,
        prefix: str,
        directory: Path,
        deadline: float | None,
    ) -> ShmSetup:
        try:
            stream.send_bytes(pickle.dumps(ShmHello(slot, prefix)))
            answer = pickle.loads(stream.receive_bytes(deadline=deadline))
        except TimeoutError:
            raise NoListener(_SETUP_LATE) from None
        except (Closed, OSError) as exc:
            raise NoListener(f"the model server closed the link while it opened: {exc}") from None
        except Exception as exc:
            raise ProtocolError(
                f"the model server answered the link's hello unreadably: {exc!r}"
            ) from None
        if isinstance(answer, ShmRefused):
            raise WorkerUnreachable(answer.reason)
        if not isinstance(answer, ShmSetup):
            raise ProtocolError("the model server answered the link's hello with something else")
        try:
            probe = attach_block(answer.probe, directory)
        except (OSError, ValueError):
            raise WorkerUnreachable(_not_shared(directory)) from None
        try:
            seen = bytes(probe.buf[: len(answer.nonce)])
        finally:
            probe.close()
        if seen != answer.nonce:
            raise WorkerUnreachable(_not_shared(directory))
        with contextlib.suppress(OSError):
            (directory / answer.probe).unlink()
        return answer


_SETUP_LATE = "the model server did not finish setting up the link in time"


def _remaining(deadline: float | None) -> float | None:
    """The seconds left before *deadline*, for a socket timeout; ``None`` is no bound.

    Raises:
        TimeoutError: The deadline passed. A timeout of 0 would make the socket
            non-blocking, and its EAGAIN would hide that the time ran out.
    """
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("the connect's timeout passed")
    return remaining


def _identity(path: Path) -> tuple[int, int, int]:
    """The file at *path*, told apart from one created at the same path later.

    An inode number freed by an unlink is often given to the next file at once,
    so the change time joins it.
    """
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_ctime_ns)


def _accepts(address: str) -> bool:
    """Whether something accepts connections on the Unix socket at *address*.

    Only a refused connection, or a file that is gone, proves the socket is
    stale; any other failure (a full backlog, a socket of another user, a
    timeout) is a live socket.
    """
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        probe.connect(address)
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    except OSError:
        return True
    finally:
        probe.close()
    return True


def _fixed_slot(prefix: str, size: int, directory: Path | None) -> SharedSlot:
    """A slot whose one block is *size* bytes and never grows."""
    return SharedSlot(prefix=prefix, initial_bytes=size, max_bytes=size, directory=directory)


def _missing_directory(directory: Path) -> str:
    return (
        f"the model server's socket directory {directory} does not exist: mount the volume "
        "the model server and the session runtimes share there"
    )


def _not_shared(directory: Path) -> str:
    return (
        f"the model server's shared memory is not visible in {directory}: the session's "
        "shared-memory directory must be the same memory as the model server's directory for "
        "its slot"
    )
