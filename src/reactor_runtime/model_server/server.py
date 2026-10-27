"""The model worker's process: one :class:`ModelServer`, many session processes.

:class:`WorkerServer` listens on its transport's socket (a Unix socket, with
shared memory for the arrays, by default). Each session process that connects
gets a reader thread, which decodes its frames, rebuilds its requests (mapping
their arrays out of shared memory without copying them), and queues them. One
step thread, the thread that called :meth:`WorkerServer.serve`, forms batches
from the queue by the policy :class:`ModelServer` documents, runs ``step()``,
and sends each answer back the same way.

A fault stays with the session it belongs to. A request that does not decode,
does not match the worker's ``request_type``, names another session's memory,
or is too large is answered as rejected; a frame that breaks the protocol (an
undeclared signal included) closes that session's connection alone; an output
that fails, or holds a value that does not travel, fails that request alone. A
``step()`` that raises fails every request of its batch with the same error,
and the worker goes on: it never steps a request twice.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import select
import socket
import struct
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reactor_runtime.distributed.errors import WorkerCrashed
from reactor_runtime.log import get_logger
from reactor_runtime.model_server import codec, schema, wire
from reactor_runtime.model_server.contract import (
    ModelServer,
    ServerRequest,
    ServerSettings,
    check_declarations,
)
from reactor_runtime.model_server.shm import RequestRefused
from reactor_runtime.model_server.transport import Transport, UnixShmTransport, WorkerCarrier
from reactor_runtime.model_server.wire import Kind

logger = get_logger(__name__)

# How long a step thread waits for work before it checks whether it was asked to stop.
_STOP_POLL = 0.2
# How long a new connection may take to say hello.
_HELLO_TIMEOUT = 10.0
# A STEP frame is its timeout and a codec header; its arrays are in shared memory.
_STEP_FRAME_LIMIT = wire.TIMEOUT.size + codec.MAX_HEADER_BYTES
_PEERCRED = struct.Struct("3i")
_TOKEN = re.compile(r"[0-9a-f]{4,16}")
# How long an answer may wait for a session to read it before the session is cut off.
_SEND_TIMEOUT = 5.0


@dataclass(eq=False)
class _Conn:
    slot: int
    peer_pid: int
    sock: socket.socket
    carrier: WorkerCarrier
    request_prefix: str
    transport: dict[str, str]
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
    """Serve one loaded :class:`ModelServer` to the session processes that connect.

    Args:
        worker: The worker, already loaded.
        socket_path: Where sessions connect. A stale file there is replaced.
        shm_prefix: Every shared block a session writes, and every one this
            server writes, is named under this prefix.
        transport: How frames and payloads travel; a Unix socket and shared
            memory by default.
        settings: The serving settings, as ``reactor.yaml`` adjusted the
            class's (:meth:`ServerSettings.resolve`); the class's own when
            omitted.

    Raises:
        TypeError: The worker class's declared types are not valid declarations.
        ValueError: A serving setting is out of range.
    """

    def __init__(
        self,
        worker: ModelServer,
        socket_path: str | Path,
        *,
        shm_prefix: str,
        transport: Transport | None = None,
        settings: ServerSettings | None = None,
    ) -> None:
        cls = type(worker)
        check_declarations(cls)
        settings = settings or ServerSettings.resolve(cls, {})
        settings.check()
        self._settings = settings
        self._worker = worker
        self._path = Path(socket_path)
        self._prefix = shm_prefix
        self._request_type = cls.request_type
        self._result_type = cls.result_type
        self._signal_types = cls.signals
        self._schema = schema.fingerprint(
            request_type=cls.request_type, result_type=cls.result_type, signals=cls.signals
        )
        self._contract = schema.contract_text(cls.request_type, cls.result_type, cls.signals)
        self._transport = transport or UnixShmTransport()
        self._shared: dict[str, str] = {}
        self._max_batch = settings.max_batch
        self._window = settings.batch_window
        self._limit = settings.max_request_bytes
        self._cond = threading.Condition()
        self._waiting: list[_Request] = []
        self._joining: list[_Conn] = []
        self._leaving: list[_Conn] = []
        self._resets: list[tuple[_Conn, int]] = []
        self._by_slot: dict[int, _Conn] = {}
        # Slots whose connection is being set up, outside the lock.
        self._opening: set[int] = set()
        # How many sessions each slot has opened, across connections.
        self._generations: dict[int, int] = {}
        self._open: set[_Conn] = set()
        self._arrivals = itertools.count()
        self._conn_ids = itertools.count()
        self._stopping = False
        self._listener: socket.socket | None = None
        self._step_started: float | None = None

    @property
    def settings(self) -> ServerSettings:
        """The serving settings in effect."""
        return self._settings

    @property
    def schema(self) -> str:
        """The fingerprint of the worker's request, result, and signal types."""
        return self._schema

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
            protocol=wire.PROTOCOL,
            schema=self._schema,
            contract=self._contract,
            max_batch=self._max_batch,
            batch_window=self._window,
            max_request_bytes=self._limit,
            shm_input_bytes=self._settings.shm_input_bytes,
            shm_output_bytes=self._settings.shm_output_bytes,
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

    def close(self) -> None:
        """Release what :meth:`bind` set up, for a server that never served."""
        self._shutdown()

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

            def limit(kind: Kind) -> int:
                return _STEP_FRAME_LIMIT if kind is Kind.STEP else wire.CONTROL_LIMIT

            while True:
                kind, request_id, payload = wire.recv(sock, limit)
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
            if conn is None:
                with contextlib.suppress(OSError):
                    sock.close()
            else:
                # Closed on the step thread once the connection is: a send
                # still under way must not find the descriptor reused.
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                with self._cond:
                    self._leave(conn)
                    self._cond.notify_all()

    def _hello(self, sock: socket.socket) -> _Conn | None:
        """Read a session's hello and queue its connection to be opened."""
        sock.settimeout(_HELLO_TIMEOUT)
        kind, request_id, payload = wire.recv(sock, wire.CONTROL_LIMIT)
        if kind is not Kind.HELLO:
            raise wire.ProtocolError(f"expected a HELLO, got {kind.name}")
        raw = wire.json_object(payload)
        protocol = raw.get("protocol")
        if protocol != wire.PROTOCOL:
            self._refuse(
                sock,
                request_id,
                f"the session speaks model worker protocol {protocol!r} and this worker speaks "
                f"protocol {wire.PROTOCOL}: run both halves on the same reactor-runtime",
            )
            return None
        try:
            hello = wire.decode_message(payload, wire.Hello)
        except wire.ProtocolError as exc:
            self._refuse(sock, request_id, str(exc))
            return None
        if hello.schema != self._schema:
            self._refuse(
                sock,
                request_id,
                f"the session's request, result, and signal types (fingerprint {hello.schema}) "
                f"differ from the worker's (fingerprint {self._schema}: {self._contract})",
            )
            return None
        slot = hello.slot
        token = hello.token
        if slot is not None and slot < 0:
            self._refuse(sock, request_id, f"slot must be a non-negative integer, got {slot!r}")
            return None
        if not _TOKEN.fullmatch(token):
            self._refuse(sock, request_id, "the hello carries no valid token")
            return None
        sock.settimeout(None)
        peer_pid = _peer_pid(sock)
        with self._cond:
            if self._stopping:
                # Closed without a refusal: a session waiting in connect()
                # goes on waiting for the worker that replaces this one.
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
                if not _gone(held):
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
            carrier, extra = self._transport.worker_carrier(
                self._prefix,
                slot,
                token,
                conn_id,
                self._limit,
                self._settings.shm_input_bytes,
                self._settings.shm_output_bytes,
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
            request_prefix=extra.pop("request_prefix"),
            transport=extra,
            hello_id=request_id,
        )
        with self._cond:
            self._opening.discard(slot)
            if self._stopping:
                carrier.close()
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
            self._reject(conn, request_id, "the request has no timeout field")
            return
        (timeout,) = wire.TIMEOUT.unpack_from(payload)
        if not conn.opened:
            self._reject(conn, request_id, "the session is not open on the worker")
            return
        try:
            value = conn.carrier.decode(payload[wire.TIMEOUT.size :])
            if self._request_type is not None:
                value = schema.load(value, self._request_type)
        except (RequestRefused, schema.SchemaError) as exc:
            logger.warning("rejected a session's request", slot=conn.slot, reason=str(exc))
            self._reject(conn, request_id, f"the request does not decode: {exc}")
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
        """Deliver one signal; one that is not a declared type refuses the session.

        Raises:
            wire.ProtocolError: The signal's tag is not in the worker's table,
                or its fields do not match the type the tag names. The
                session is sent the reason, then its connection is closed.
        """
        try:
            if len(payload) < wire.SIGNAL_TAG.size:
                raise wire.ProtocolError("a signal has no tag")
            (tag,) = wire.SIGNAL_TAG.unpack_from(payload)
            if tag >= len(self._signal_types):
                raise wire.ProtocolError(
                    f"a signal names tag {tag}, which the worker does not declare "
                    f"(it declares {len(self._signal_types)} signal types)"
                )
            declared = self._signal_types[tag]
            try:
                tree = json.loads(payload[wire.SIGNAL_TAG.size :])
                signal = schema.load(codec.from_tree(tree), declared)
            except (UnicodeDecodeError, ValueError, RecursionError, schema.SchemaError) as exc:
                raise wire.ProtocolError(
                    f"a {declared.__qualname__} signal does not decode: {exc}"
                ) from None
        except wire.ProtocolError as exc:
            with conn.send_lock, contextlib.suppress(OSError):
                _send_within(
                    conn.sock,
                    wire.encode(Kind.REFUSED, 0, wire.encode_message(wire.Refusal(str(exc)))),
                    _SEND_TIMEOUT,
                )
            raise
        if not conn.opened:
            return
        try:
            self._worker.signal(conn.slot, signal)
        except Exception:
            logger.exception("signal() raised", slot=conn.slot, signal=declared.__qualname__)

    def _refuse(self, sock: socket.socket, request_id: int, reason: str) -> None:
        logger.warning("refused a session", reason=reason)
        with contextlib.suppress(OSError):
            wire.send(sock, Kind.REFUSED, request_id, wire.encode_message(wire.Refusal(reason)))

    def _reject(self, conn: _Conn, request_id: int, reason: str) -> None:
        self._reply(conn, Kind.REJECTED, request_id, wire.encode_message(wire.Refusal(reason)))

    def _reply(self, conn: _Conn, kind: Kind, request_id: int, payload: bytes = b"") -> None:
        """Send one frame to *conn*; a session that does not read it in time is cut off.

        The socket's send timeout bounds the wait, so a session that stops
        reading cannot hold the step thread, and with it every other session.
        """
        if conn.closed:
            return
        try:
            with conn.send_lock:
                _send_within(conn.sock, wire.encode(kind, request_id, payload), _SEND_TIMEOUT)
        except OSError as exc:
            # The reader sees the socket end and closes the connection.
            logger.warning(
                "a session stopped taking its answers; closing it", slot=conn.slot, error=str(exc)
            )
            with contextlib.suppress(OSError):
                conn.sock.shutdown(socket.SHUT_RDWR)

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
                self._guarded(self._close, conn)
            for conn in joining:
                self._guarded(self._open_conn, conn)
            for conn, request_id in resets:
                self._guarded(self._reset, conn, request_id)
            if leaving or joining or resets:
                continue
            batch = self._take_batch()
            if batch:
                self._step_started = time.monotonic()
                try:
                    self._run(batch)
                finally:
                    self._step_started = None
                continue
            with self._cond:
                if not (self._joining or self._leaving or self._resets or self._ready()):
                    self._cond.wait(_STOP_POLL)

    def _guarded(self, call: Any, conn: _Conn, *args: Any) -> None:
        """Run one connection's open, close, or reset; a failure closes that connection alone.

        Each call is timed on its own for the stuck check, so many sessions
        opening at once do not add up to one long step.
        """
        self._step_started = time.monotonic()
        try:
            call(conn, *args)
        except Exception:
            logger.exception("a session's open, close, or reset failed; closing it", slot=conn.slot)
            with contextlib.suppress(OSError):
                conn.sock.shutdown(socket.SHUT_RDWR)
        finally:
            self._step_started = None

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
            expired = [r for r in self._waiting if r.deadline is not None and r.deadline <= now]
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
        conn.generation = self._next_generation(conn.slot)
        try:
            self._worker.session_opened(conn.slot, conn.generation)
        except Exception as exc:
            logger.exception("session_opened() refused a session", slot=conn.slot)
            self._reply(
                conn,
                Kind.REFUSED,
                conn.hello_id,
                wire.encode_message(wire.Refusal(f"session_opened() refused it: {_describe(exc)}")),
            )
            with contextlib.suppress(OSError):
                conn.sock.shutdown(socket.SHUT_RDWR)
            return
        conn.opened = True
        with self._cond:
            self._open.add(conn)
        logger.info("session opened", slot=conn.slot, generation=conn.generation, pid=conn.peer_pid)
        welcome = wire.Welcome(
            slot=conn.slot,
            generation=conn.generation,
            max_batch=self._max_batch,
            max_request_bytes=self._limit,
            shm_slot_bytes=self._settings.shm_slot_bytes,
            signals=tuple(s.__qualname__ for s in self._signal_types),
            request_prefix=conn.request_prefix,
            transport={**self._shared, **conn.transport},
        )
        self._reply(conn, Kind.WELCOME, conn.hello_id, wire.encode_message(welcome))

    def _next_generation(self, slot: int) -> int:
        self._generations[slot] = self._generations.get(slot, 0) + 1
        return self._generations[slot]

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
        try:
            conn.carrier.close()
        finally:
            with conn.send_lock, contextlib.suppress(OSError):
                conn.sock.close()

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
        conn.generation = self._next_generation(conn.slot)
        try:
            self._worker.session_opened(conn.slot, conn.generation)
        except Exception as exc:
            logger.exception("session_opened() refused a reset", slot=conn.slot)
            conn.opened = False
            with self._cond:
                self._open.discard(conn)
            self._reject(conn, request_id, f"session_opened() refused the reset: {_describe(exc)}")
            return
        conn.opened = True
        with self._cond:
            self._open.add(conn)
        self._reply(conn, Kind.ACK, request_id, wire.encode_message(wire.ResetAck(conn.generation)))

    def _run(self, batch: list[_Request]) -> None:
        started = time.monotonic()
        requests = [ServerRequest(r.conn.slot, r.conn.generation, r.payload) for r in batch]
        failed: bytes | None = None
        outputs: list[Any] = []
        try:
            returned = self._worker.step(requests)
            if not isinstance(returned, list | tuple) or len(returned) != len(batch):
                count = (
                    len(returned) if isinstance(returned, list | tuple) else type(returned).__name__
                )
                raise RuntimeError(f"step() returned {count} outputs for {len(batch)} requests")
            outputs = list(returned)
        except Exception as exc:
            logger.exception("step() raised; failing its batch", size=len(batch))
            failed = _failure(exc, batch_failed=True)
        ran = time.monotonic() - started
        for index, request in enumerate(batch):
            request.payload = None
            try:
                if failed is not None:
                    self._reply(request.conn, Kind.FAILED, request.request_id, failed)
                else:
                    self._answer(request, outputs[index])
            except Exception:
                logger.exception("an answer could not be sent", slot=request.conn.slot)
                self._reply(request.conn, Kind.FAILED, request.request_id, _UNDESCRIBED)
        logger.debug(
            "step ran",
            size=len(batch),
            slots=[r.conn.slot for r in batch],
            raised=failed is not None,
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
            if self._result_type is not None:
                output = schema.dump(output, self._result_type)
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


class WorkerStopping(WorkerCrashed):
    """The answer to a request still queued when the worker was stopped."""


def _send_within(sock: socket.socket, data: bytes, seconds: float) -> None:
    """Send all of *data* within *seconds* in all, or raise ``TimeoutError``.

    Non-blocking sends against one deadline, so a session that reads a little
    now and then cannot stretch one frame past it; the socket's own reads stay
    blocking for its reader.
    """
    view = memoryview(data)
    deadline = time.monotonic() + seconds
    poller = select.poll()
    poller.register(sock, select.POLLOUT)
    while view:
        try:
            sent = sock.send(view, socket.MSG_DONTWAIT)
        except BlockingIOError:
            sent = 0
        view = view[sent:]
        if not view:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"the session took no answer for {seconds} s")
        poller.poll(remaining * 1000)


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


def _gone(conn: _Conn) -> bool:
    """Whether the process on *conn*'s other end has gone, closing its socket.

    Read from the socket itself, which works across PID namespaces (separate
    containers of one pod), where the peer's pid reads as 0.
    """
    if conn.peer_pid > 0 and not _alive(conn.peer_pid):
        return True
    poller = select.poll()
    poller.register(conn.sock, select.POLLIN | select.POLLHUP | select.POLLERR)
    try:
        events = poller.poll(0)
    except (OSError, ValueError):
        return True
    for _, mask in events:
        if mask & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
            return True
        if mask & select.POLLIN:
            try:
                return conn.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
            except BlockingIOError:
                return False
            except OSError:
                return True
    return False


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


def _text(call: Any, fallback: str) -> str:
    try:
        return str(call())
    except Exception:
        return fallback


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {_text(lambda: str(exc), '<unprintable>')}"


def _failure(exc: BaseException, *, batch_failed: bool = False) -> bytes:
    """Describe *exc* for the session: its type's name, its message, its traceback.

    An exception that cannot be printed is still described, by its type.
    """
    kind = type(exc)
    return wire.encode_message(
        wire.Failure(
            type=f"{kind.__module__}.{kind.__qualname__}",
            message=_text(lambda: str(exc), "<unprintable>"),
            traceback=_text(lambda: "".join(traceback.format_exception(exc)), ""),
            worker_lost=isinstance(exc, WorkerStopping),
            batch_failed=batch_failed,
        )
    )


_UNDESCRIBED = wire.encode_message(
    wire.Failure(
        type="builtins.RuntimeError",
        message="the answer could not be sent",
        traceback="",
        worker_lost=False,
        batch_failed=False,
    )
)
