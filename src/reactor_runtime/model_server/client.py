"""The application half's connection to the model worker.

A session process opens one :class:`ModelClient` in its model's ``load()`` and
calls :meth:`ModelClient.step` from ``generate()``. The client writes the
payload's arrays into a shared block of its own, sends the worker a small
header, and blocks until the answer, which it copies out of the block the
worker keeps for it.

The client reports and never decides. It holds no reference to the
application and does nothing in the background but read its socket: a worker
that goes away marks the connection lost, and from then on every call raises
:class:`WorkerLost` on the application's own call path until the application
calls :meth:`ModelClient.reconnect`. What a lost or a stuck worker means for
the session is the application's to decide, in ``process_output()`` (see
:mod:`reactor_runtime.model_server` for the two ways to do it).
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import itertools
import json
import os
import secrets
import select
import socket
import sys
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reactor_runtime.distributed.errors import WorkerCrashed, WorkerTimeout
from reactor_runtime.log import get_logger
from reactor_runtime.model_server import codec, schema, wire
from reactor_runtime.model_server.contract import check_signal_types
from reactor_runtime.model_server.transport import (
    ClientCarrier,
    NoListener,
    Transport,
    UnixShmTransport,
    WorkerUnreachable,
)
from reactor_runtime.model_server.wire import Kind

logger = get_logger(__name__)

SOCKET_ENV = "REACTOR_MODEL_WORKER_SOCKET"
SLOT_ENV = "REACTOR_SESSION_SLOT"
STEP_TIMEOUT_ENV = "REACTOR_WORKER_STEP_TIMEOUT"
CONNECT_TIMEOUT_ENV = "REACTOR_WORKER_CONNECT_TIMEOUT"
DEFAULT_SOCKET = "/run/reactor-worker/sock"

_DEFAULT_STEP_TIMEOUT = 30.0
# Long enough for a worker loading large weights after a restart.
_DEFAULT_CONNECT_TIMEOUT = 1800.0
_CONNECT_POLL_FIRST = 0.2
_CONNECT_POLL_MAX = 2.0
# How long one opened connection may take to be welcomed: a worker welcomes
# between steps, so a long step delays it.
_HELLO_TIMEOUT = 120.0


class WorkerRejected(ValueError):  # noqa: N818 (named for the event a caller catches)
    """The worker refused a request or a session: malformed, too large, or refused by the model.

    Only the session it names is affected; the worker and every other session
    go on.
    """


class WorkerError(RuntimeError):
    """The model failed this request in the worker.

    Its message is ``"<type>: <message>"`` of the worker's exception. Nothing
    of the exception object crosses: nothing on the socket is pickled, either
    way.

    Attributes:
        type_name: The exception's qualified type, such as ``"mymodel.NoFace"``.
        message: The exception's message.
        worker_traceback: Its traceback in the worker, as text.
    """

    def __init__(self, type_name: str, message: str, worker_traceback: str) -> None:
        """Record the worker's exception by name, message, and traceback."""
        super().__init__(f"{type_name}: {message}")
        self.type_name = type_name
        self.message = message
        self.worker_traceback = worker_traceback


class BatchFailed(WorkerError):  # noqa: N818 (named for the event a caller catches)
    """The worker's ``step()`` raised for the whole batch this request was in.

    Every request of the batch gets it, with the same cause: :attr:`type_name`,
    :attr:`message`, and :attr:`worker_traceback` describe the exception
    ``step()`` raised. The cause may be another session's input; a model that
    can tell which request is bad returns an exception for that request alone,
    which arrives as a plain :class:`WorkerError`.
    """


class StepUnanswered(WorkerTimeout):
    """A step the model did not answer: :class:`StepTimedOut` or :class:`StepExpired`.

    A :class:`~reactor_runtime.distributed.WorkerTimeout`. The connection is
    still up and the client stays usable.
    """


class StepTimedOut(StepUnanswered):
    """No answer came within the step's timeout: the worker is slow, or stuck.

    The request was cancelled. A worker that stays stuck is the worker's
    liveness probe's to restart.
    """


class StepExpired(StepUnanswered):
    """The worker dropped the request before stepping it.

    Its timeout passed while it waited in the worker's queue, or the same
    session sent another request that replaced it.
    """


class WorkerLost(WorkerCrashed):
    """The connection to the worker is gone: its message says why.

    Raised by every call once the connection is lost, until
    :meth:`ModelClient.reconnect` succeeds. A
    :class:`~reactor_runtime.distributed.WorkerCrashed`.
    """


class ClientClosed(WorkerLost):
    """The client was closed: every call raises it, :meth:`ModelClient.wait_lost` included."""


class ResetRequired(RuntimeError):  # noqa: N818 (named for the state a caller catches)
    """A step on a connection whose slot the application has not reset yet.

    Raised by :meth:`ModelClient.step` after :meth:`ModelClient.connect` and
    after :meth:`ModelClient.reconnect`, and after a :meth:`ModelClient.reset`
    that did not complete, until a reset succeeds. The application calls
    ``reset()`` when its session starts, so a session never runs on state the
    worker kept for another one.
    """


@dataclass(eq=False)
class _Link:
    """One live connection to the worker, and the shared memory that goes with it."""

    sock: socket.socket
    slot: int
    generation: int
    max_request_bytes: int
    carrier: ClientCarrier
    replies: dict[int, tuple[Kind, bytes]] = field(default_factory=dict)
    lost: WorkerLost | None = None
    closing: bool = False
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        with contextlib.suppress(OSError):
            self.sock.close()
        self.carrier.close()


# Every client that is open, closed by the interpreter's exit before it joins
# the threads still running: a client blocked in connect() or reconnect() on a
# pool thread (asyncio.to_thread) then returns instead of holding the exit.
_OPEN: weakref.WeakSet[ModelClient] = weakref.WeakSet()


def _close_all() -> None:
    for client in list(_OPEN):
        with contextlib.suppress(Exception):
            client.close()


_register_exit: Callable[[Callable[[], None]], None] | None = getattr(
    threading, "_register_atexit", None
)
if _register_exit is not None:
    _register_exit(_close_all)
atexit.register(_close_all)


class ModelClient:
    """One session process's connection to the model worker.

    Build it with :meth:`connect`. One :meth:`step` or :meth:`reset` runs at a
    time; a second caller waits for the first. :meth:`signal`,
    :meth:`wait_lost`, and :meth:`close` may be called from any thread.

    Every failure is a raise from the call that sees it:

    * :class:`WorkerError` when the model failed this request, and its
      subclass :class:`BatchFailed` when ``step()`` raised for the whole batch.
    * :class:`WorkerRejected` when the worker refused the request (it did not
      decode, did not match the worker's ``request_type``, or was larger than
      ``max_request_bytes``) or ``session_opened()`` refused a reset.
    * :class:`StepTimedOut` when no answer came within the timeout (the request
      is cancelled), and :class:`StepExpired` when the worker dropped it from
      its queue; both are :class:`StepUnanswered`, and the client stays usable.
    * :class:`ResetRequired` from a step before the application has reset the
      slot on this connection.
    * :class:`WorkerLost` once the connection is lost, from every call, until
      :meth:`reconnect` succeeds; :class:`ClientClosed` after :meth:`close`.
    """

    def __init__(
        self,
        *,
        socket_path: str,
        slot: int | None,
        step_timeout: float,
        transport: Transport,
        request_type: type | None,
        result_type: type | None,
        signals: tuple[type, ...],
    ) -> None:
        """Hold the settings; :meth:`connect` builds and connects one."""
        self._path = socket_path
        self._transport = transport
        self._wanted_slot = slot
        self._step_timeout = step_timeout
        self._request_type = request_type
        self._result_type = result_type
        self._signal_types = signals
        self._signal_tags = {cls: tag for tag, cls in enumerate(signals)}
        self._schema = schema.fingerprint(
            request_type=request_type, result_type=result_type, signals=signals
        )
        self._contract = schema.contract_text(request_type, result_type, signals)
        self._ids = itertools.count(1)
        self._call_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._link: _Link | None = None
        self._lost_reason: str | None = None
        self._reset_required = True
        self._closed = False
        self._last_slot = slot
        _OPEN.add(self)

    @classmethod
    def connect(
        cls,
        *,
        socket_path: str | Path | None = None,
        slot: int | None = None,
        request_type: type | None = None,
        result_type: type | None = None,
        signals: tuple[type, ...] = (),
        connect_timeout: float | None = None,
        step_timeout: float | None = None,
        transport: Transport | None = None,
    ) -> ModelClient:
        """Connect to the worker and block until it has opened this session.

        Called from the session half's ``load()``: while it waits, the session
        runtime is in its loading state, so a cold start and a worker restart
        both look like a slow model load. It waits through a worker that is
        not listening yet, or that goes away before it welcomes the session,
        and returns once one does.

        The types are the session half's view of the contract and must be
        the worker's own (``ModelServer.request_type``, ``result_type``, and
        ``signals``), imported from the model's code; the worker refuses a
        session whose types differ.

        Args:
            socket_path: The worker's socket; ``REACTOR_MODEL_WORKER_SOCKET``
                when omitted, which must then be set.
            slot: This session's slot, the number the worker keeps its state
                under; ``REACTOR_SESSION_SLOT``, else the lowest free slot the
                worker assigns (read it back from :attr:`slot`).
            request_type: The frozen dataclass :meth:`step` sends, or ``None``
                for plain values.
            result_type: The frozen dataclass :meth:`step` returns, built from
                the fields the worker sent, or ``None`` for plain values.
            signals: The frozen dataclasses :meth:`signal` sends, in the
                worker's order.
            connect_timeout: Seconds to wait for a welcome in all;
                ``REACTOR_WORKER_CONNECT_TIMEOUT``, else 1800. ``0`` waits
                without bound.
            step_timeout: Seconds a :meth:`step` or :meth:`reset` waits for
                its answer by default; ``REACTOR_WORKER_STEP_TIMEOUT``, else
                30. ``0`` waits without bound.
            transport: How frames and payloads travel; a Unix socket and
                shared memory by default, which needs the session and the
                worker on one host sharing the socket's directory and
                ``/dev/shm``.

        Raises:
            WorkerUnreachable: At once, for a deployment that cannot reach a
                worker however long it waits: no socket is named, the socket's
                directory is missing, the socket is not open to this user, or
                the worker's shared memory is not visible from here.
            WorkerRejected: The worker refused this session: another protocol
                version, other types, a slot held by a live process, no shared
                memory for one more session, or ``session_opened()`` refused it.
            TypeError: A declared type is not a data-only frozen dataclass.
            WorkerTimeout: No worker welcomed the session within *connect_timeout*.
            ClientClosed: :meth:`close` was called while it waited, as the
                interpreter's exit does on SIGTERM.
        """
        path = str(socket_path or os.getenv(SOCKET_ENV, "").strip())
        if not path:
            raise WorkerUnreachable(
                f"{SOCKET_ENV} is not set: this model's session half needs a model worker "
                "(python -m reactor_runtime.model_server) and the path of its socket"
            )
        if slot is None:
            raw = os.getenv(SLOT_ENV, "").strip()
            slot = int(raw) if raw else None
        if step_timeout is None:
            step_timeout = _float_env(STEP_TIMEOUT_ENV, _DEFAULT_STEP_TIMEOUT)
        if connect_timeout is None:
            connect_timeout = _float_env(CONNECT_TIMEOUT_ENV, _DEFAULT_CONNECT_TIMEOUT)
        for name, declared in (("request_type", request_type), ("result_type", result_type)):
            if declared is not None:
                try:
                    schema.validate(declared)
                except schema.SchemaError as exc:
                    raise TypeError(f"{name}: {exc}") from None
        if not isinstance(signals, tuple):
            raise TypeError(f"signals must be a tuple of frozen dataclasses, got {signals!r}")
        check_signal_types(signals, "signals")
        _cap_malloc_arenas()
        transport = transport or UnixShmTransport()
        transport.check(path)
        client = cls(
            socket_path=path,
            slot=slot,
            step_timeout=step_timeout,
            transport=transport,
            request_type=request_type,
            result_type=result_type,
            signals=signals,
        )
        try:
            client._connect_within(connect_timeout)
        except BaseException:
            client.close()
            raise
        return client

    @property
    def slot(self) -> int | None:
        """This session's slot, ``None`` until a worker has assigned one."""
        return self._last_slot

    @property
    def generation(self) -> int:
        """The worker's count of openings of this slot, moved on by :meth:`reset`.

        ``0`` while the connection is lost.
        """
        link = self._link
        return link.generation if link is not None else 0

    @property
    def available(self) -> bool:
        """Whether the connection is up. A stuck worker's connection is up."""
        return self._link is not None

    @property
    def lost_reason(self) -> str | None:
        """Why the connection was lost, or ``None`` while it is up."""
        return self._lost_reason

    def step(self, payload: Any, *, timeout: float | None = None) -> Any:
        """Send *payload* to the worker's next step and return this session's output.

        Pass contiguous numpy arrays for the bulk of the payload: they travel
        through shared memory. The output's arrays are copies this session
        owns. The header that carries the rest (plain values, the arrays'
        shapes) is at most 1 MiB.

        Args:
            payload: An instance of the ``request_type`` passed to
                :meth:`connect`, or plain values, lists, tuples, dicts, bytes,
                and numpy arrays without one (see
                :mod:`reactor_runtime.model_server.codec`).
            timeout: Seconds to wait for the answer; the client's step timeout
                when omitted, and ``0`` waits without bound.

        Returns:
            An instance of ``result_type``, or plain values without one.

        Raises:
            WorkerError: The model failed this request.
            BatchFailed: ``step()`` raised for the batch this request was in.
            TypeError: The payload does not match ``request_type``, or holds a
                value that does not travel.
            WorkerRejected: The request is too large, or the worker refused it.
            StepTimedOut: No answer in time; the request was cancelled.
            StepExpired: The worker dropped the request from its queue.
            ResetRequired: The application has not reset the slot on this
                connection yet.
            WorkerLost: The connection is lost.
        """
        with self._call_lock:
            link = self._usable()
            if self._reset_required:
                raise ResetRequired(
                    "reset() has not completed on this connection to the model worker: the "
                    "application resets its slot when its session starts"
                )
            try:
                return self._step(link, payload, timeout)
            finally:
                if link.lost is not None:
                    link.release()

    def reset(self, *, timeout: float | None = None) -> int:
        """Close this slot's session on the worker and open a new one.

        The worker runs ``session_closed()`` then ``session_opened()`` for the
        slot, between steps. Only the application calls it, when its session
        starts, so nothing of the last session reaches it; until it completes
        on a connection, :meth:`step` raises :class:`ResetRequired`.

        Returns:
            The slot's new generation.

        Raises:
            WorkerRejected: ``session_opened()`` refused the new session.
            StepTimedOut: No answer in time.
            WorkerLost: The connection is lost.
        """
        with self._call_lock:
            link = self._usable()
            self._reset_required = True
            try:
                generation = self._reset(link, timeout)
            finally:
                if link.lost is not None:
                    link.release()
            self._reset_required = False
            return generation

    def signal(self, signal: Any) -> None:
        """Deliver *signal* to the worker's ``signal()`` now, even while a step waits.

        Never waits for the worker. Safe from any thread.

        Args:
            signal: An instance of one of the ``signals`` types passed to
                :meth:`connect`.

        Raises:
            ValueError: Its type is not one of the declared signals.
            TypeError: Its fields do not match its declaration.
            WorkerLost: The connection is lost.
        """
        tag = self._signal_tags.get(type(signal))
        if tag is None:
            names = [cls.__qualname__ for cls in self._signal_types]
            raise ValueError(
                f"{type(signal).__qualname__} is not a declared signal; this client "
                f"declares {names}"
            )
        try:
            fields = schema.dump(signal, type(signal))
        except schema.SchemaError as exc:
            raise TypeError(str(exc)) from None
        body = wire.SIGNAL_TAG.pack(tag) + _json(codec.to_tree(fields))
        link = self._usable()
        self._send(link, Kind.SIGNAL, 0, body)

    def wait_lost(self, timeout: float | None = None) -> str:
        """Block until the connection is lost, and return why.

        Returns at once while it is lost. Meant for one task of the
        application that watches the worker between steps and while idle (see
        :mod:`reactor_runtime.model_server`); call it with a timeout, in a
        loop, so the task can be cancelled.

        Args:
            timeout: Seconds to wait; ``None`` waits without bound.

        Raises:
            TimeoutError: The connection is still up after *timeout*.
            ClientClosed: :meth:`close` was called, before or while it waited.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if self._closed:
                    raise ClientClosed("the worker client is closed")
                if self._link is None and self._lost_reason is not None:
                    return self._lost_reason
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"the connection to the model worker is up after {timeout} s"
                    )
                self._cond.wait(remaining)

    def reconnect(self, timeout: float) -> None:
        """Drop the connection, if any, and connect again, blocking until the worker welcomes it.

        The new connection is a new session on the worker, with a new
        generation; :meth:`step` raises :class:`ResetRequired` until the
        application calls :meth:`reset`. For a stuck worker this waits until
        the worker is restarted, since a worker opens a session only between
        steps.

        Args:
            timeout: Seconds to wait in all; call it in a loop to wait longer,
                so shutdown is never held by it.

        Raises:
            WorkerTimeout: No worker welcomed the session within *timeout*.
            WorkerUnreachable: The deployment cannot reach a worker at all.
            WorkerRejected: The worker refused the session.
            ClientClosed: :meth:`close` was called, before or while it waited.
        """
        with self._call_lock:
            link = self._link
            if link is not None:
                self._lose(link, "the application reconnects")
                link.release()
        self._connect_within(timeout)

    def close(self) -> None:
        """Leave the worker and release this client's shared memory. Idempotent.

        A :meth:`connect`, :meth:`reconnect`, or :meth:`wait_lost` blocked in
        another thread raises :class:`ClientClosed`, and so does every call
        after it.
        """
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            link, self._link = self._link, None
            if self._lost_reason is None:
                self._lost_reason = "the worker client is closed"
            if link is not None and link.lost is None:
                link.lost = ClientClosed("the worker client is closed")
            self._cond.notify_all()
        _OPEN.discard(self)
        if link is not None:
            link.closing = True
            with contextlib.suppress(OSError):
                link.sock.shutdown(socket.SHUT_RDWR)
            with self._call_lock:
                link.release()

    # -- one step ----------------------------------------------------------------------------

    def _step(self, link: _Link, payload: Any, timeout: float | None) -> Any:
        request_id = next(self._ids)
        wait = self._step_timeout if timeout is None else timeout
        limit = link.max_request_bytes
        if self._request_type is not None:
            try:
                payload = schema.dump(payload, self._request_type)
            except schema.SchemaError as exc:
                raise TypeError(str(exc)) from None
        size = link.carrier.size(payload)
        if size > limit:
            raise WorkerRejected(
                f"the request carries {size} bytes, over the worker's {limit} limit"
            )
        try:
            header, total = link.carrier.encode(payload)
        except codec.HeaderTooLarge as exc:
            link.carrier.renew()
            raise WorkerRejected(str(exc)) from None
        if total > limit:
            link.carrier.renew()
            raise WorkerRejected(
                f"the request carries {total} bytes, over the worker's {limit} limit"
            )
        self._send(link, Kind.STEP, request_id, wire.TIMEOUT.pack(wait) + header)
        kind, reply = self._await(link, request_id, wait)
        if kind in (Kind.RESULT, Kind.FAILED, Kind.EXPIRED):
            # Answered by the worker's step thread, which is done with every
            # request before this one; a rejection comes from its reader.
            link.carrier.settled()
        if kind is None:
            self._send(link, Kind.CANCEL, request_id, best_effort=True)
            # The worker may still read the abandoned request, so the next
            # one is written elsewhere.
            link.carrier.renew()
            raise StepTimedOut(f"the model worker gave no answer within {wait} s")
        if kind is Kind.RESULT:
            try:
                value = link.carrier.decode(reply)
                if self._result_type is not None:
                    value = schema.load(value, self._result_type)
            except (ValueError, schema.SchemaError) as exc:
                raise RuntimeError(f"the model worker's output did not decode: {exc}") from None
            return value
        if kind is Kind.FAILED:
            raise _failure(reply)
        if kind is Kind.REJECTED:
            raise WorkerRejected(_reason(reply))
        if kind is Kind.EXPIRED:
            raise StepExpired("the request expired in the model worker's queue")
        raise self._protocol_break(link, f"answered a step with {kind.name}")

    def _reset(self, link: _Link, timeout: float | None) -> int:
        request_id = next(self._ids)
        wait = self._step_timeout if timeout is None else timeout
        self._send(link, Kind.RESET, request_id)
        kind, reply = self._await(link, request_id, wait)
        if kind is None:
            raise StepTimedOut(f"the model worker did not reset within {wait} s")
        if kind is Kind.REJECTED:
            raise WorkerRejected(_reason(reply))
        if kind is not Kind.ACK:
            raise self._protocol_break(link, f"answered a reset with {kind.name}")
        try:
            ack = wire.decode_message(reply, wire.ResetAck)
        except wire.ProtocolError as exc:
            raise self._protocol_break(link, str(exc)) from None
        link.generation = ack.generation
        return ack.generation

    def _protocol_break(self, link: _Link, reason: str) -> WorkerLost:
        """Drop a connection whose worker broke the protocol; return what the caller raises."""
        self._lose(link, f"the model worker broke the protocol: {reason}")
        assert link.lost is not None
        return link.lost

    # -- connecting --------------------------------------------------------------------------

    def _connect_within(self, timeout: float) -> None:
        """Open a connection, waiting through a worker that is not there yet, until *timeout*."""
        deadline = None if timeout <= 0 else time.monotonic() + timeout
        poll = _CONNECT_POLL_FIRST
        last = "no model worker listens yet"
        while True:
            if self._closed:
                raise ClientClosed("the worker client is closed")
            try:
                self._open(deadline)
                return
            except NoListener as exc:
                last = f"no model worker listens at {self._path}: {exc}"
            except WorkerLost as exc:
                if isinstance(exc, ClientClosed):
                    raise
                last = str(exc)
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise WorkerTimeout(
                    f"no model worker welcomed this session within {timeout} s: {last}"
                )
            if self._stop.wait(poll if remaining is None else min(poll, remaining)):
                raise ClientClosed("the worker client is closed")
            poll = min(poll * 2, _CONNECT_POLL_MAX)

    def _open(self, deadline: float | None) -> None:
        """Connect once and install the link, or raise why not."""
        sock = self._transport.connect(self._path)
        # A slot the worker assigned is not asked for again: another session
        # may hold it by now. Its state is a new session's either way.
        hello = wire.Hello(
            slot=self._wanted_slot,
            pid=os.getpid(),
            token=secrets.token_hex(4),
            protocol=wire.PROTOCOL,
            schema=self._schema,
        )
        wait = _HELLO_TIMEOUT if deadline is None else max(0.1, deadline - time.monotonic())
        try:
            wire.send(sock, Kind.HELLO, 0, wire.encode_message(hello))
            kind, payload = self._await_welcome(sock, min(wait, _HELLO_TIMEOUT))
        except (wire.Closed, OSError, wire.ProtocolError) as exc:
            sock.close()
            if self._closed:
                raise ClientClosed("the worker client is closed") from None
            raise WorkerLost(f"worker_died: lost while opening the session: {exc}") from None
        if kind is Kind.REFUSED:
            sock.close()
            reason = _reason(payload)
            raise WorkerRejected(
                f"the model worker refused the session: {reason}"
                + (f" (this session declares: {self._contract})" if "fingerprint" in reason else "")
            )
        if kind is not Kind.WELCOME:
            sock.close()
            raise WorkerLost(f"worker_died: answered a hello with {kind.name}")
        try:
            welcome = wire.decode_message(payload, wire.Welcome)
        except wire.ProtocolError as exc:
            sock.close()
            raise WorkerRejected(f"the model worker's welcome does not decode: {exc}") from None
        try:
            carrier = self._transport.client_carrier(
                {**welcome.transport, "request_prefix": welcome.request_prefix}
            )
        except BaseException:
            sock.close()
            raise
        link = _Link(
            sock=sock,
            slot=welcome.slot,
            generation=welcome.generation,
            max_request_bytes=welcome.max_request_bytes,
            carrier=carrier,
        )
        with self._cond:
            if self._closed:
                link.release()
                raise ClientClosed("the worker client is closed")
            self._link = link
            self._lost_reason = None
            self._reset_required = True
            self._last_slot = link.slot
            self._cond.notify_all()
        reader = threading.Thread(
            target=self._read_loop, args=(link,), name="worker-client", daemon=True
        )
        reader.start()
        logger.info(
            "connected to the model worker",
            slot=link.slot,
            generation=link.generation,
            socket=self._path,
            max_batch=welcome.max_batch,
            max_request_bytes=welcome.max_request_bytes,
            shm_slot_bytes=welcome.shm_slot_bytes,
        )

    def _await_welcome(self, sock: socket.socket, wait: float) -> tuple[Kind, bytes]:
        """Read the answer to a hello, waking every so often to see whether the client closed."""
        deadline = time.monotonic() + wait
        while True:
            readable, _, _ = select.select([sock], [], [], 0.5)
            if self._closed:
                raise wire.Closed("the worker client is closed")
            if readable:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no welcome within {wait:.0f} s")
        # The frame has started: read it whole, bounded so a stalled worker cannot hold it.
        sock.settimeout(max(1.0, deadline - time.monotonic()))
        kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
        sock.settimeout(None)
        return kind, payload

    # -- one connection ----------------------------------------------------------------------

    def _usable(self) -> _Link:
        with self._cond:
            if self._closed:
                raise ClientClosed("the worker client is closed")
            link = self._link
            if link is None:
                reason = self._lost_reason or "not connected"
                raise WorkerLost(
                    reason if reason.startswith("worker_") else f"worker_died: {reason}"
                )
            return link

    def _send(
        self,
        link: _Link,
        kind: Kind,
        request_id: int,
        payload: bytes = b"",
        *,
        best_effort: bool = False,
    ) -> None:
        try:
            with self._send_lock:
                wire.send(link.sock, kind, request_id, payload)
        except OSError as exc:
            if best_effort:
                return
            self._lose(link, f"the connection failed: {exc}")
            assert link.lost is not None
            raise link.lost from None

    def _await(self, link: _Link, request_id: int, wait: float) -> tuple[Kind | None, bytes]:
        deadline = None if wait <= 0 else time.monotonic() + wait
        with self._cond:
            while True:
                reply = link.replies.pop(request_id, None)
                if reply is not None:
                    link.replies.clear()
                    return reply
                if link.lost is not None:
                    raise link.lost
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None, b""
                self._cond.wait(remaining)

    def _read_loop(self, link: _Link) -> None:
        reason = "the model worker closed the connection"
        try:
            while True:
                kind, request_id, payload = wire.recv(link.sock, _reply_limit)
                if kind is Kind.REFUSED:
                    reason = f"the model worker refused this session: {_reason(payload)}"
                    break
                with self._cond:
                    link.replies[request_id] = (kind, payload)
                    self._cond.notify_all()
        except wire.Closed:
            pass
        except (OSError, wire.ProtocolError) as exc:
            reason = f"the connection failed: {exc}"
        if not link.closing:
            self._lose(link, reason)
        elif self._call_lock.acquire(blocking=False):
            try:
                link.release()
            finally:
                self._call_lock.release()

    def _lose(self, link: _Link, reason: str) -> None:
        """Retire *link*: fail what waits on it and record why; nothing reconnects.

        The link's shared memory is released by whichever comes last: this,
        when no call is using it, or the call that is, as it returns.
        """
        with self._cond:
            if link.lost is not None:
                return
            link.lost = WorkerLost(f"worker_died: {reason}")
            if self._link is link:
                self._link = None
                if not self._closed:
                    self._lost_reason = f"worker_died: {reason}"
            self._cond.notify_all()
        logger.error("lost the model worker", slot=link.slot, reason=reason)
        link.closing = True
        with contextlib.suppress(OSError):
            link.sock.shutdown(socket.SHUT_RDWR)
        if self._call_lock.acquire(blocking=False):
            try:
                link.release()
            finally:
                self._call_lock.release()


def _reply_limit(kind: Kind) -> int:
    return codec.MAX_HEADER_BYTES if kind is Kind.RESULT else wire.CONTROL_LIMIT


def _json(tree: Any) -> bytes:
    return json.dumps(tree, separators=(",", ":"), allow_nan=False).encode()


def _reason(payload: bytes) -> str:
    """The reason a REFUSED or REJECTED frame carries."""
    try:
        return wire.decode_message(payload, wire.Refusal).reason
    except wire.ProtocolError as exc:
        return f"(the refusal does not decode: {exc})"


def _failure(payload: bytes) -> Exception:
    """The exception a FAILED answer stands for."""
    try:
        failure = wire.decode_message(payload, wire.Failure)
    except wire.ProtocolError as exc:
        return RuntimeError(
            f"the model worker failed this request; its answer did not decode: {exc}"
        )
    if failure.worker_lost:
        return WorkerLost(failure.message)
    cls = BatchFailed if failure.batch_failed else WorkerError
    return cls(failure.type, failure.message, failure.traceback)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} {raw!r} must be a number of seconds") from None
    if not value >= 0:
        raise ValueError(f"{name} must be at least 0")
    return value


# glibc's mallopt option for the arena count.
_M_ARENA_MAX = -8
_ARENA_ENV = "REACTOR_SESSION_MALLOC_ARENAS"
_arenas_capped = False


def _cap_malloc_arenas() -> None:
    """Cap glibc's malloc arenas in a session process, unless the environment chose.

    glibc opens arenas as threads contend, up to eight per core, and keeps
    each at the most it ever held, so a session process serving session after
    session grows by its arenas, not by what it holds. A session process holds
    no weights, so two arenas cost it nothing. ``MALLOC_ARENA_MAX``, when set,
    is glibc's own and wins; ``REACTOR_SESSION_MALLOC_ARENAS`` sets the cap
    (``2`` by default, ``0`` leaves glibc's default). Linux with glibc only.
    """
    global _arenas_capped
    if _arenas_capped or os.getenv("MALLOC_ARENA_MAX"):
        return
    _arenas_capped = True
    raw = os.getenv(_ARENA_ENV, "").strip()
    try:
        arenas = int(raw) if raw else 2
    except ValueError:
        logger.warning("ignored a malformed arena cap", variable=_ARENA_ENV, value=raw)
        return
    if arenas <= 0 or not sys.platform.startswith("linux"):
        return
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None).mallopt(_M_ARENA_MAX, arenas)
