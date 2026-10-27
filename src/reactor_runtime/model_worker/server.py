"""The model worker's process: one :class:`ModelWorker`, many session processes.

:class:`WorkerServer` listens on its transport's socket (a Unix socket, with
shared memory for the arrays, by default). Each session process that connects
gets a reader thread, which decodes its frames, rebuilds its requests (mapping
their arrays out of shared memory without copying them), and queues them. One
step thread, the thread that called :meth:`WorkerServer.serve`, forms batches
from the queue by the policy :class:`ModelWorker` documents, runs ``step()``,
and sends each answer back the same way.

A fault stays with the session it belongs to. A request that does not decode,
names another session's memory, or is too large is answered as rejected; a
frame that breaks the protocol closes that session's connection alone; an
output that fails, or holds a value that does not travel, fails that request alone.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import re
import socket
import struct
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reactor_runtime.distributed.errors import WorkerCrashed
from reactor_runtime.interface.events.errors import ModelUnavailable
from reactor_runtime.log import get_logger
from reactor_runtime.model_worker import wire
from reactor_runtime.model_worker.contract import ModelWorker, WorkerRequest
from reactor_runtime.model_worker.shm import RequestRefused
from reactor_runtime.model_worker.transport import Transport, UnixShmTransport, WorkerCarrier
from reactor_runtime.model_worker.wire import Kind

logger = get_logger(__name__)

# How long a step thread waits for work before it checks whether it was asked to stop.
_STOP_POLL = 0.2
# How long a new connection may take to say hello.
_HELLO_TIMEOUT = 10.0
# Room for the header around a request's arrays, past the request limit.
_FRAME_SLACK = 64 << 10
_PEERCRED = struct.Struct("3i")
_TOKEN = re.compile(r"[0-9a-f]{4,16}")


@dataclass(eq=False)
class _Conn:
    slot: int
    peer_pid: int
    sock: socket.socket
    carrier: WorkerCarrier
    welcome: dict[str, Any]
    hello_id: int
    generation: int = 0
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    opened: bool = False
    closed: bool = False
    leave_queued: bool = False


@dataclass(eq=False)
class _Request:
    conn: _Conn
    request_id: int
    payload: Any
    arrived: float
    deadline: float | None
    order: int


class WorkerServer:
    """Serve one loaded :class:`ModelWorker` to the session processes that connect.

    Args:
        worker: The worker, already loaded.
        socket_path: Where sessions connect. A stale file there is replaced.
        shm_prefix: Every shared block a session writes, and every one this
            server writes, is named under this prefix.
        transport: How frames and payloads travel; a Unix socket and shared
            memory by default.
    """

    def __init__(
        self,
        worker: ModelWorker,
        socket_path: str | Path,
        *,
        shm_prefix: str,
        transport: Transport | None = None,
    ) -> None:
        max_batch = type(worker).max_batch
        if type(max_batch) is not int or max_batch < 1:
            raise ValueError(f"max_batch must be a positive integer, got {max_batch!r}")
        window = float(type(worker).batch_window)
        if not window >= 0:
            raise ValueError(f"batch_window must be at least 0, got {window!r}")
        limit = type(worker).max_request_bytes
        if type(limit) is not int or limit < 1:
            raise ValueError(f"max_request_bytes must be a positive integer, got {limit!r}")
        for name in ("shm_input_bytes", "shm_output_bytes"):
            size = getattr(type(worker), name)
            if type(size) is not int or size < 1:
                raise ValueError(f"{name} must be a positive integer, got {size!r}")
        self._worker = worker
        self._path = Path(socket_path)
        self._prefix = shm_prefix
        signals = type(worker).signals
        if not isinstance(signals, frozenset | set | tuple | list) or not all(
            isinstance(name, str) for name in signals
        ):
            raise ValueError(f"signals must be a set of names, got {signals!r}")
        self._signals = frozenset(signals)
        self._transport = transport or UnixShmTransport()
        self._shared: dict[str, Any] = {}
        self._max_batch = max_batch
        self._window = window
        self._limit = limit
        self._cond = threading.Condition()
        self._waiting: list[_Request] = []
        self._joining: list[_Conn] = []
        self._leaving: list[_Conn] = []
        self._resets: list[tuple[_Conn, int]] = []
        self._by_slot: dict[int, _Conn] = {}
        # Slots whose connection is being set up, outside the lock.
        self._opening: set[int] = set()
        self._open: set[_Conn] = set()
        self._arrivals = itertools.count()
        self._conn_ids = itertools.count()
        self._stopping = False
        self._listener: socket.socket | None = None
        self._step_started: float | None = None

    @property
    def socket_path(self) -> Path:
        """The socket sessions connect to."""
        return self._path

    def bind(self) -> None:
        """Bind and listen, so a session can connect from now on."""
        self._transport.check(str(self._path))
        self._shared = self._transport.worker_started(self._prefix)
        self._listener = self._transport.listen(str(self._path))

    def serve(self) -> None:
        """Accept sessions and run steps until :meth:`stop`; :meth:`bind` first."""
        if self._listener is None:
            self.bind()
        accept = threading.Thread(target=self._accept_loop, name="worker-accept", daemon=True)
        accept.start()
        logger.info(
            "model worker serving",
            socket=str(self._path),
            max_batch=self._max_batch,
            batch_window=self._window,
        )
        try:
            self._step_loop()
        finally:
            self._shutdown()

    @property
    def step_running_for(self) -> float:
        """Seconds the step running now has taken, ``0`` between steps."""
        started = self._step_started
        return 0.0 if started is None else time.monotonic() - started

    def stop(self) -> None:
        """Ask :meth:`serve` to return once the step running, if any, finishes.

        Safe from a signal handler: it takes no lock.
        """
        self._stopping = True

    # -- connections -------------------------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stopping:
            try:
                sock, _ = self._listener.accept()
            except OSError:
                return
            thread = threading.Thread(
                target=self._client_loop, args=(sock,), name="worker-client", daemon=True
            )
            thread.start()

    def _client_loop(self, sock: socket.socket) -> None:
        conn: _Conn | None = None
        try:
            conn = self._hello(sock)
            if conn is None:
                return
            frame_limit = self._limit + _FRAME_SLACK
            while True:
                kind, request_id, payload = wire.recv(sock, frame_limit)
                if kind is not Kind.STEP and len(payload) > wire.CONTROL_LIMIT:
                    raise wire.ProtocolError(f"a {kind.name} frame of {len(payload)} bytes")
                if kind is Kind.STEP:
                    self._take_request(conn, request_id, payload)
                elif kind is Kind.SIGNAL:
                    self._signal(conn, payload)
                elif kind is Kind.RESET:
                    with self._cond:
                        self._resets.append((conn, request_id))
                        self._cond.notify_all()
                elif kind is Kind.CANCEL:
                    with self._cond:
                        self._waiting = [
                            r
                            for r in self._waiting
                            if not (r.conn is conn and r.request_id == request_id)
                        ]
                else:
                    raise wire.ProtocolError(f"a session sent a {kind.name} frame")
        except (wire.Closed, ConnectionError):
            pass
        except wire.ProtocolError as exc:
            logger.warning(
                "closing a session that broke the protocol",
                slot=conn.slot if conn else None,
                error=str(exc),
            )
        except OSError as exc:
            logger.warning(
                "a session's socket failed", slot=conn.slot if conn else None, error=str(exc)
            )
        except Exception:
            logger.exception("a session's reader failed", slot=conn.slot if conn else None)
        finally:
            with contextlib.suppress(OSError):
                sock.close()
            if conn is not None:
                with self._cond:
                    self._leave(conn)
                    self._cond.notify_all()

    def _hello(self, sock: socket.socket) -> _Conn | None:
        """Read a session's hello and queue its connection to be opened."""
        sock.settimeout(_HELLO_TIMEOUT)
        kind, request_id, payload = wire.recv(sock, wire.CONTROL_LIMIT)
        if kind is not Kind.HELLO:
            raise wire.ProtocolError(f"expected a HELLO, got {kind.name}")
        hello = wire.from_json(payload)
        slot = hello.get("slot")
        token = hello.get("token")
        if slot is not None and (type(slot) is not int or slot < 0):
            self._refuse(sock, request_id, f"slot must be a non-negative integer, got {slot!r}")
            return None
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            self._refuse(sock, request_id, "the hello carries no valid token")
            return None
        sock.settimeout(None)
        peer_pid = _peer_pid(sock)
        with self._cond:
            if self._stopping:
                self._refuse(sock, request_id, "the model worker is stopping")
                return None
            if slot is None:
                slot = next(
                    n
                    for n in itertools.count()
                    if n not in self._by_slot and n not in self._opening
                )
            if slot in self._opening:
                self._refuse(sock, request_id, f"slot {slot} is being opened by another connection")
                return None
            held = self._by_slot.get(slot)
            if held is not None and not held.closed:
                if _alive(held.peer_pid):
                    self._refuse(
                        sock, request_id, f"slot {slot} is held by process {held.peer_pid}"
                    )
                    return None
                # The process that held the slot is gone and its reader has
                # not seen the end of its socket yet. Its close is queued
                # first, so the slot is closed before it is opened again.
                self._leave(held)
                with contextlib.suppress(OSError):
                    held.sock.shutdown(socket.SHUT_RDWR)
            self._opening.add(slot)
            conn_id = next(self._conn_ids)
        # Made and zero-filled outside the lock: the step thread goes on meanwhile.
        try:
            carrier, welcome = self._transport.worker_carrier(
                self._prefix,
                slot,
                token,
                conn_id,
                self._limit,
                type(self._worker).shm_input_bytes,
                type(self._worker).shm_output_bytes,
            )
        except Exception as exc:
            with self._cond:
                self._opening.discard(slot)
            self._refuse(sock, request_id, f"no shared memory for one more session: {exc}")
            return None
        conn = _Conn(
            slot=slot,
            peer_pid=peer_pid,
            sock=sock,
            carrier=carrier,
            welcome=welcome,
            hello_id=request_id,
        )
        with self._cond:
            self._opening.discard(slot)
            if self._stopping:
                carrier.close()
                self._refuse(sock, request_id, "the model worker is stopping")
                return None
            self._by_slot[slot] = conn
            self._joining.append(conn)
            self._cond.notify_all()
        return conn

    def _leave(self, conn: _Conn) -> None:
        """Mark *conn* closed and queue its close. Call with the condition held."""
        conn.closed = True
        self._waiting = [r for r in self._waiting if r.conn is not conn]
        if self._by_slot.get(conn.slot) is conn:
            del self._by_slot[conn.slot]
        if not conn.leave_queued:
            conn.leave_queued = True
            self._leaving.append(conn)

    def _take_request(self, conn: _Conn, request_id: int, payload: bytes) -> None:
        if len(payload) < wire.TIMEOUT.size:
            self._reply(conn, Kind.REJECTED, request_id, b"the request has no timeout field")
            return
        (timeout,) = wire.TIMEOUT.unpack_from(payload)
        try:
            value = conn.carrier.decode(payload[wire.TIMEOUT.size :])
        except RequestRefused as exc:
            logger.warning("rejected a session's request", slot=conn.slot, reason=str(exc))
            self._reply(conn, Kind.REJECTED, request_id, str(exc).encode())
            return
        now = time.monotonic()
        request = _Request(
            conn=conn,
            request_id=request_id,
            payload=value,
            arrived=now,
            deadline=now + timeout if timeout > 0 else None,
            order=next(self._arrivals),
        )
        superseded: list[_Request] = []
        with self._cond:
            if conn.closed:
                return
            superseded = [r for r in self._waiting if r.conn is conn]
            self._waiting = [r for r in self._waiting if r.conn is not conn]
            self._waiting.append(request)
            self._cond.notify_all()
        for old in superseded:
            self._reply(conn, Kind.EXPIRED, old.request_id)

    def _signal(self, conn: _Conn, payload: bytes) -> None:
        try:
            signal = wire.from_json(payload)
        except wire.ProtocolError as exc:
            logger.warning("dropped a signal that does not decode", slot=conn.slot, error=str(exc))
            return
        name, value = signal.get("name"), signal.get("payload")
        if not isinstance(name, str) or name not in self._signals:
            logger.warning(
                "dropped a signal the worker does not declare", slot=conn.slot, signal=name
            )
            return
        if not conn.opened:
            return
        try:
            self._worker.signal(conn.slot, name, value)
        except Exception:
            logger.exception("signal() raised", slot=conn.slot, signal=name)

    def _refuse(self, sock: socket.socket, request_id: int, reason: str) -> None:
        logger.warning("refused a session", reason=reason)
        with contextlib.suppress(OSError):
            wire.send(sock, Kind.REFUSED, request_id, reason.encode())

    def _reply(self, conn: _Conn, kind: Kind, request_id: int, payload: bytes = b"") -> None:
        if conn.closed:
            return
        try:
            with conn.send_lock:
                wire.send(conn.sock, kind, request_id, payload)
        except OSError:
            pass  # The reader sees the socket end and closes the connection.

    # -- the step thread ---------------------------------------------------------------------

    def _step_loop(self) -> None:
        while not self._stopping:
            with self._cond:
                if not (self._waiting or self._joining or self._leaving or self._resets):
                    self._cond.wait(_STOP_POLL)
                    continue
                leaving, self._leaving = self._leaving, []
                joining, self._joining = self._joining, []
                resets, self._resets = self._resets, []
            for conn in leaving:
                self._close(conn)
            for conn in joining:
                self._open_conn(conn)
            for conn, request_id in resets:
                self._reset(conn, request_id)
            if leaving or joining or resets:
                continue
            batch = self._take_batch()
            if batch:
                self._run(batch)

    def _take_batch(self) -> list[_Request]:
        """Take the next batch off the queue, waiting out the batch window first."""
        with self._cond:
            ready = self._ready()
            if ready and self._window > 0:
                until = ready[0].arrived + self._window
                while (
                    not self._stopping
                    and len(ready) < self._max_batch
                    and len(ready) < len(self._open)
                    and not (self._joining or self._leaving or self._resets)
                ):
                    remaining = until - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)
                    ready = self._ready()
            now = time.monotonic()
            expired = [r for r in ready if r.deadline is not None and r.deadline <= now]
            batch = [r for r in ready if r not in expired][: self._max_batch]
            taken = set(map(id, batch + expired))
            self._waiting = [r for r in self._waiting if id(r) not in taken]
        for request in expired:
            self._reply(request.conn, Kind.EXPIRED, request.request_id)
        return batch

    def _ready(self) -> list[_Request]:
        return sorted(
            (r for r in self._waiting if r.conn.opened and not r.conn.closed),
            key=lambda r: r.order,
        )

    def _open_conn(self, conn: _Conn) -> None:
        if conn.closed:
            return
        conn.generation += 1
        try:
            self._worker.session_opened(conn.slot, conn.generation)
        except Exception as exc:
            logger.exception("session_opened() refused a session", slot=conn.slot)
            self._reply(conn, Kind.REFUSED, conn.hello_id, _describe(exc).encode())
            with contextlib.suppress(OSError):
                conn.sock.shutdown(socket.SHUT_RDWR)
            return
        conn.opened = True
        with self._cond:
            self._open.add(conn)
        logger.info("session opened", slot=conn.slot, generation=conn.generation, pid=conn.peer_pid)
        welcome = {
            **self._shared,
            **conn.welcome,
            "slot": conn.slot,
            "generation": conn.generation,
            "max_batch": self._max_batch,
            "max_request_bytes": self._limit,
            "shm_slot_bytes": type(self._worker).shm_slot_size(),
            "signals": sorted(self._signals),
        }
        self._reply(conn, Kind.WELCOME, conn.hello_id, wire.to_json(welcome))

    def _close(self, conn: _Conn) -> None:
        with self._cond:
            self._open.discard(conn)
        if conn.opened:
            conn.opened = False
            try:
                self._worker.session_closed(conn.slot, conn.generation)
            except Exception:
                logger.exception("session_closed() raised", slot=conn.slot)
            logger.info("session closed", slot=conn.slot, generation=conn.generation)
        conn.carrier.close()

    def _reset(self, conn: _Conn, request_id: int) -> None:
        if conn.closed:
            return
        if conn.opened:
            with self._cond:
                self._waiting = [r for r in self._waiting if r.conn is not conn]
            try:
                self._worker.session_closed(conn.slot, conn.generation)
            except Exception:
                logger.exception("session_closed() raised", slot=conn.slot)
        conn.generation += 1
        try:
            self._worker.session_opened(conn.slot, conn.generation)
        except Exception as exc:
            logger.exception("session_opened() refused a reset", slot=conn.slot)
            conn.opened = False
            with self._cond:
                self._open.discard(conn)
            self._reply(conn, Kind.REJECTED, request_id, _describe(exc).encode())
            return
        conn.opened = True
        with self._cond:
            self._open.add(conn)
        self._reply(conn, Kind.ACK, request_id, wire.to_json({"generation": conn.generation}))

    def _run(self, batch: list[_Request]) -> None:
        started = time.monotonic()
        requests = [WorkerRequest(r.conn.slot, r.conn.generation, r.payload) for r in batch]
        self._step_started = started
        try:
            outputs = self._worker.step(requests)
            if not isinstance(outputs, list | tuple) or len(outputs) != len(batch):
                count = (
                    len(outputs) if isinstance(outputs, list | tuple) else type(outputs).__name__
                )
                raise RuntimeError(f"step() returned {count} outputs for {len(batch)} requests")
        except Exception as exc:
            logger.exception("step() raised", size=len(batch))
            outputs = [exc] * len(batch)
        finally:
            self._step_started = None
        ran = time.monotonic() - started
        del requests
        for request, output in zip(batch, outputs, strict=True):
            request.payload = None
            self._answer(request, output)
        logger.debug(
            "step ran",
            size=len(batch),
            slots=[r.conn.slot for r in batch],
            step_ms=round(ran * 1000, 2),
            wait_ms=[round((started - r.arrived) * 1000, 2) for r in batch],
        )

    def _answer(self, request: _Request, output: Any) -> None:
        conn = request.conn
        if conn.closed:
            return
        if isinstance(output, BaseException):
            self._reply(conn, Kind.FAILED, request.request_id, _failure(output))
            return
        try:
            header = conn.carrier.encode(output)
        except Exception as exc:
            logger.exception("an output could not be sent", slot=conn.slot)
            self._reply(conn, Kind.FAILED, request.request_id, _failure(exc))
            return
        self._reply(conn, Kind.RESULT, request.request_id, header)

    def _shutdown(self) -> None:
        """Stop accepting, fail what waits, close every session, and free their memory."""
        with self._cond:
            waiting, self._waiting = self._waiting, []
        stopping = _failure(WorkerStopping("worker_died: the model worker is stopping"))
        for request in waiting:
            self._reply(request.conn, Kind.FAILED, request.request_id, stopping)
        if self._listener is not None:
            with contextlib.suppress(OSError):
                self._listener.close()
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()
        with self._cond:
            conns = list(self._open) + list(self._joining)
            for conn in list(self._by_slot.values()):
                self._leave(conn)
        for conn in conns:
            with contextlib.suppress(OSError):
                conn.sock.shutdown(socket.SHUT_RDWR)
        for conn in {*conns, *self._leaving}:
            self._close(conn)
        self._transport.worker_stopped()


class WorkerStopping(WorkerCrashed, ModelUnavailable):
    """The answer to a request still queued when the worker was stopped."""


def _peer_pid(sock: socket.socket) -> int:
    """The pid of the process on the other end, or ``0`` where the platform does not say."""
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        return 0
    try:
        pid, _, _ = _PEERCRED.unpack(sock.getsockopt(socket.SOL_SOCKET, option, _PEERCRED.size))
    except OSError:
        return 0
    return pid


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _failure(exc: BaseException) -> bytes:
    """Describe *exc* for the session as JSON: its type's name, its message, its traceback."""
    kind = type(exc)
    return wire.to_json(
        {
            "type": f"{kind.__module__}.{kind.__qualname__}",
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
            "worker_lost": isinstance(exc, WorkerStopping),
        }
    )
