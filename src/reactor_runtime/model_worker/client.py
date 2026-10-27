"""The session half's connection to the model worker.

A session process opens one :class:`WorkerClient` in its model's ``load()`` and
calls :meth:`WorkerClient.step` from ``generate()``. The client writes the
payload's arrays into a shared block of its own, sends the worker a small
header, and blocks until the answer, which it copies out of the block the
worker keeps for it. A thread of the client reads the socket, so a worker that
dies is seen at once, even between steps; the client then connects again in
the background until a worker answers.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import itertools
import os
import secrets
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from reactor_runtime.distributed.errors import WorkerCrashed, WorkerTimeout
from reactor_runtime.interface.events.errors import ModelUnavailable
from reactor_runtime.log import get_logger
from reactor_runtime.model_worker import wire
from reactor_runtime.model_worker.transport import (
    ClientCarrier,
    NoListener,
    Transport,
    UnixShmTransport,
    WorkerUnreachable,
)
from reactor_runtime.model_worker.wire import Kind


class AvailabilitySink(Protocol):
    """What the client reports the worker's comings and goings to: a ``ReactorApp``."""

    def set_unavailable(self, reason: str) -> None:
        """The worker is gone, for *reason*."""

    def set_available(self) -> None:
        """A worker is connected again."""


logger = get_logger(__name__)

SOCKET_ENV = "REACTOR_MODEL_WORKER_SOCKET"
SLOT_ENV = "REACTOR_SESSION_SLOT"
STEP_TIMEOUT_ENV = "REACTOR_WORKER_STEP_TIMEOUT"
DEFAULT_SOCKET = "/run/reactor-worker/sock"

_DEFAULT_STEP_TIMEOUT = 30.0
_CONNECT_POLL = 0.2
_HELLO_TIMEOUT = 30.0
_RECONNECT_FIRST = 0.5
_RECONNECT_MAX = 5.0
# Outputs carry their arrays in shared memory; this bounds the header around them.
_REPLY_LIMIT = 1 << 30


class WorkerRejected(ValueError):  # noqa: N818 (named for the event a caller catches)
    """The worker refused a request or a session: malformed, too large, or refused by the model.

    Only the session it names is affected; the worker and every other session
    go on.
    """


class WorkerError(RuntimeError):
    """The model raised for this request in the worker.

    Its message is ``"<type>: <message>"`` of the worker's exception. Nothing
    of the exception object crosses: nothing on the socket is pickled, either
    way.

    Attributes:
        type_name: The exception's qualified type, such as ``"mymodel.NoFace"``.
        message: The exception's message.
        worker_traceback: Its traceback in the worker, as text.
    """

    def __init__(self, type_name: str, message: str, worker_traceback: str) -> None:
        super().__init__(f"{type_name}: {message}")
        self.type_name = type_name
        self.message = message
        self.worker_traceback = worker_traceback


class WorkerLost(WorkerCrashed, ModelUnavailable):
    """No worker is connected: its message starts with ``worker_died:`` and says why.

    A :class:`~reactor_runtime.ModelUnavailable`, so the default step loop
    drops the step it ends instead of crashing the session's process.
    """


@dataclass(eq=False)
class _Link:
    """One live connection to the worker, and the shared memory that goes with it."""

    sock: socket.socket
    slot: int
    generation: int
    max_request_bytes: int | None
    carrier: ClientCarrier
    signals: frozenset[str] | None = None
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


class WorkerClient:
    """One session process's connection to the model worker.

    Build it with :meth:`connect`. One :meth:`step` runs at a time; a second
    caller waits for the first. :meth:`signal` may be called from any thread at
    any time.

    The client keeps itself connected. When the worker goes away it notices at
    once, and connects again in the background, every half second at first and
    every five seconds at most, until a worker answers. A client built with the
    session's app reports both moves to it, with
    :meth:`~reactor_runtime.ReactorApp.set_unavailable` (reason
    ``worker_died: <why>``) and :meth:`~reactor_runtime.ReactorApp.set_available`:
    the runtime then ends the live session with that reason, answers
    ``/health`` with 503 and refuses new sessions until the worker is back,
    and the process stays up. A reconnected session is a new one on the worker:
    its ``session_opened()`` runs again.

    Every failure is a raise from the call that sees it:

    * :class:`WorkerError` when the model raised for this request; it names
      the worker's exception type and carries its message and traceback.
    * :class:`WorkerRejected` when the worker refused the request (it did not
      decode, was larger than the worker's ``max_request_bytes``) or the
      model's ``session_opened()`` refused the session.
    * :class:`~reactor_runtime.distributed.WorkerTimeout` when no answer came
      within the timeout. The request is cancelled; the client stays usable.
    * :class:`WorkerLost`, whose message starts ``worker_died:``, while no
      worker is connected.
    """

    def __init__(
        self,
        *,
        socket_path: str,
        slot: int | None,
        step_timeout: float,
        app: AvailabilitySink | None,
        transport: Transport,
    ) -> None:
        self._path = socket_path
        self._transport = transport
        self._wanted_slot = slot
        self._step_timeout = step_timeout
        self._app = app
        self._ids = itertools.count(1)
        self._call_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._cond = threading.Condition()
        self._link: _Link | None = None
        self._lost_reason = "worker_unavailable: not connected yet"
        self._closed = False
        self._reconnecting = False
        self._last_slot = slot
        atexit.register(self.close)

    @classmethod
    def connect(
        cls,
        app: AvailabilitySink | None = None,
        *,
        socket_path: str | Path | None = None,
        slot: int | None = None,
        connect_timeout: float = 60.0,
        step_timeout: float | None = None,
        transport: Transport | None = None,
    ) -> WorkerClient:
        """Connect to the worker and wait until it has opened this session.

        Args:
            app: The session's app, passed from its ``load()`` as
                ``WorkerClient.connect(self)``. With it, a worker that is not
                there yet is no error: the client returns at once with the app
                marked unavailable, and connects in the background.
            socket_path: The worker's socket; ``REACTOR_MODEL_WORKER_SOCKET``,
                else ``/run/reactor-worker/sock``.
            slot: This session's slot, the number the worker keeps its state
                under; ``REACTOR_SESSION_SLOT``, else the lowest free slot the
                worker assigns (read it back from :attr:`slot`).
            connect_timeout: Without an app, seconds to keep trying while no
                worker listens.
            step_timeout: Seconds a :meth:`step` waits for its answer by
                default; ``REACTOR_WORKER_STEP_TIMEOUT``, else 30. ``0`` waits
                without bound.
            transport: How frames and payloads travel; a Unix socket and
                shared memory by default, which needs the session and the
                worker on one host sharing the socket's directory and
                ``/dev/shm``.

        Raises:
            WorkerUnreachable: The deployment cannot reach a worker at all: the
                socket's directory is missing, or the worker's shared memory is
                not visible from here. Raised with or without an app, since
                waiting does not fix it.
            WorkerRejected: The worker refused this session.
            WorkerTimeout: Without an app, no worker accepted within
                *connect_timeout*.
            WorkerLost: Without an app, the worker closed the connection
                before opening it.
        """
        path = str(socket_path or os.getenv(SOCKET_ENV, "").strip() or DEFAULT_SOCKET)
        if slot is None:
            raw = os.getenv(SLOT_ENV, "").strip()
            slot = int(raw) if raw else None
        if step_timeout is None:
            step_timeout = float(os.getenv(STEP_TIMEOUT_ENV, "") or _DEFAULT_STEP_TIMEOUT)
        _cap_malloc_arenas()
        transport = transport or UnixShmTransport()
        transport.check(path)
        client = cls(
            socket_path=path, slot=slot, step_timeout=step_timeout, app=app, transport=transport
        )
        if app is None:
            deadline = time.monotonic() + connect_timeout
            while True:
                try:
                    client._open()
                    return client
                except NoListener as exc:
                    if time.monotonic() >= deadline:
                        client.close()
                        raise WorkerTimeout(f"no model worker accepted at {path}: {exc}") from None
                    time.sleep(_CONNECT_POLL)
                except BaseException:
                    client.close()
                    raise
        try:
            client._open()
        except (WorkerRejected, WorkerUnreachable):
            client.close()
            raise
        except NoListener as exc:
            client._went_down(f"worker_unavailable: no model worker at {path}: {exc}")
        except WorkerLost as exc:
            client._went_down(str(exc))
        return client

    @property
    def slot(self) -> int | None:
        """This session's slot, ``None`` until a worker has assigned one."""
        return self._last_slot

    @property
    def generation(self) -> int:
        """The worker's count of openings of this slot, moved on by :meth:`reset`."""
        link = self._link
        return link.generation if link is not None else 0

    @property
    def available(self) -> bool:
        """Whether a worker is connected now."""
        return self._link is not None

    def wait_available(self, timeout: float | None = None) -> bool:
        """Block until a worker is connected; return whether one is.

        Args:
            timeout: Seconds to wait; ``None`` waits without bound.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._link is None and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return self._link is not None

    def step(self, payload: Any, *, timeout: float | None = None) -> Any:
        """Send *payload* to the worker's next step and return this session's output.

        Pass contiguous numpy arrays for the bulk of the payload: they travel
        through shared memory. The output's arrays are copies this session owns.

        Args:
            payload: Plain values, lists, tuples, dicts, bytes, and numpy
                arrays (see :mod:`reactor_runtime.model_worker.codec`).
            timeout: Seconds to wait for the answer; the client's step timeout
                when omitted, and ``0`` waits without bound.

        Raises:
            WorkerError: The model raised for this request.
            TypeError: The payload holds a value that does not travel.
            WorkerRejected: The worker refused the request.
            WorkerTimeout: No answer in time; the request was cancelled.
            WorkerLost: No worker is connected.
        """
        with self._call_lock:
            link = self._usable()
            try:
                return self._step(link, payload, timeout)
            finally:
                if link.lost is not None:
                    link.release()

    def reset(self, *, timeout: float | None = None) -> int:
        """Close this slot's session on the worker and open a new one.

        The worker runs ``session_closed()`` then ``session_opened()`` for the
        slot, between steps. Call it when the session process starts a new
        session, so nothing of the last one reaches it.

        Returns:
            The slot's new generation.

        Raises:
            WorkerRejected: ``session_opened()`` refused the new session.
            WorkerTimeout: No answer in time.
            WorkerLost: No worker is connected.
        """
        with self._call_lock:
            link = self._usable()
            try:
                request_id = next(self._ids)
                wait = self._step_timeout if timeout is None else timeout
                self._send(link, Kind.RESET, request_id)
                kind, reply = self._await(link, request_id, wait)
                if kind is None:
                    raise WorkerTimeout(f"the model worker did not reset within {wait} s")
                if kind is Kind.REJECTED:
                    raise WorkerRejected(reply.decode(errors="replace"))
                if kind is not Kind.ACK:
                    raise WorkerLost(f"worker_died: answered a reset with {kind.name}")
                link.generation = int(wire.from_json(reply)["generation"])
                return link.generation
            finally:
                if link.lost is not None:
                    link.release()

    def signal(self, name: str, payload: Any = None) -> None:
        """Deliver *name* to the worker's ``signal()`` now, even while a step waits.

        Never blocks on the worker and never raises for a worker that is gone:
        the signal is dropped then. Safe from any thread.

        Args:
            name: One of the names the worker's ``ModelWorker.signals`` declares.
            payload: A JSON value.

        Raises:
            ValueError: The worker declares no signal of that name.
            TypeError: The payload is not a JSON value.
        """
        link = self._link
        if link is None:
            return
        if link.signals is not None and name not in link.signals:
            raise ValueError(
                f"the model worker declares no signal {name!r}; it takes {sorted(link.signals)}"
            )
        data = wire.to_json({"name": name, "payload": payload})
        self._send(link, Kind.SIGNAL, 0, data, best_effort=True)

    def close(self) -> None:
        """Leave the worker and release this client's shared memory. Idempotent."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            link, self._link = self._link, None
            if link is not None and link.lost is None:
                link.lost = WorkerLost("worker_unavailable: the worker client is closed")
            self._cond.notify_all()
        atexit.unregister(self.close)
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
        header, size = link.carrier.encode(payload)
        limit = link.max_request_bytes
        if limit is not None and size > limit:
            raise WorkerRejected(
                f"the request carries {size} bytes, over the worker's {limit} limit"
            )
        self._send(link, Kind.STEP, request_id, wire.TIMEOUT.pack(wait) + header)
        kind, reply = self._await(link, request_id, wait)
        if kind is None:
            self._send(link, Kind.CANCEL, request_id, best_effort=True)
            # The worker may still read the abandoned request, so the next
            # one is written elsewhere.
            link.carrier.renew()
            raise WorkerTimeout(f"the model worker gave no answer within {wait} s")
        if kind is Kind.RESULT:
            try:
                return link.carrier.decode(reply)
            except ValueError as exc:
                raise RuntimeError(f"the model worker's output did not decode: {exc}") from None
        if kind is Kind.FAILED:
            raise _failure(reply)
        if kind is Kind.REJECTED:
            raise WorkerRejected(reply.decode(errors="replace"))
        if kind is Kind.EXPIRED:
            raise WorkerTimeout("the request expired in the model worker's queue")
        raise WorkerLost(f"worker_died: answered a step with {kind.name}")

    # -- connecting --------------------------------------------------------------------------

    def _open(self) -> None:
        """Connect once and install the link, or raise why not."""
        sock = self._transport.connect(self._path)
        hello: dict[str, Any] = {"pid": os.getpid(), "token": secrets.token_hex(4)}
        wanted = self._wanted_slot if self._wanted_slot is not None else self._last_slot
        if wanted is not None:
            hello["slot"] = wanted
        try:
            wire.send(sock, Kind.HELLO, 0, wire.to_json(hello))
            sock.settimeout(_HELLO_TIMEOUT)
            kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
            sock.settimeout(None)
            welcome = wire.from_json(payload) if kind is Kind.WELCOME else {}
        except (wire.Closed, OSError, wire.ProtocolError) as exc:
            sock.close()
            raise WorkerLost(f"worker_died: lost while opening the session: {exc}") from None
        if kind is Kind.REFUSED:
            sock.close()
            raise WorkerRejected(f"the model worker refused the session: {payload.decode()}")
        if kind is not Kind.WELCOME:
            sock.close()
            raise WorkerLost(f"worker_died: answered a hello with {kind.name}")
        try:
            carrier = self._transport.client_carrier(welcome)
        except BaseException:
            sock.close()
            raise
        link = _Link(
            sock=sock,
            slot=int(welcome["slot"]),
            generation=int(welcome.get("generation", 1)),
            max_request_bytes=int(welcome.get("max_request_bytes", 0)) or None,
            carrier=carrier,
            signals=frozenset(welcome["signals"]) if "signals" in welcome else None,
        )
        with self._cond:
            if self._closed:
                link.release()
                return
            self._link = link
            self._last_slot = link.slot
        reader = threading.Thread(
            target=self._read_loop, args=(link,), name="worker-client", daemon=True
        )
        reader.start()
        logger.info("connected to the model worker", slot=link.slot, socket=self._path)
        if self._app is not None:
            self._app.set_available()
        with self._cond:
            self._cond.notify_all()

    def _went_down(self, reason: str) -> None:
        """Report the worker gone and keep connecting in the background."""
        with self._cond:
            if self._closed:
                return
            self._lost_reason = reason
            start = not self._reconnecting
            self._reconnecting = True
        if self._app is not None:
            self._app.set_unavailable(reason)
        if start:
            threading.Thread(
                target=self._reconnect_loop, name="worker-reconnect", daemon=True
            ).start()

    def _reconnect_loop(self) -> None:
        backoff = _RECONNECT_FIRST
        try:
            while not self._closed:
                time.sleep(backoff)
                if self._closed:
                    return
                try:
                    self._open()
                    return
                except (WorkerRejected, WorkerUnreachable) as exc:
                    logger.warning("could not reopen this session on the worker", error=str(exc))
                except (NoListener, WorkerLost):
                    pass
                backoff = min(backoff * 2, _RECONNECT_MAX)
        finally:
            with self._cond:
                self._reconnecting = False

    # -- one connection ----------------------------------------------------------------------

    def _usable(self) -> _Link:
        if self._closed:
            raise RuntimeError("the worker client is closed")
        link = self._link
        if link is None:
            raise WorkerLost(
                self._lost_reason
                if self._lost_reason.startswith("worker_")
                else f"worker_died: {self._lost_reason}"
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
                kind, request_id, payload = wire.recv(link.sock, _REPLY_LIMIT)
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
        """Retire *link*: fail what waits on it, and start connecting again.

        The link's shared memory is released by whichever comes last: this,
        when no call is using it, or the call that is, as it returns.
        """
        with self._cond:
            if link.lost is not None:
                return
            link.lost = WorkerLost(f"worker_died: {reason}")
            current = self._link is link
            if current:
                self._link = None
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
        if current:
            self._went_down(f"worker_died: {reason}")


def _failure(payload: bytes) -> Exception:
    """The exception a FAILED answer stands for."""
    try:
        failure = wire.from_json(payload)
    except wire.ProtocolError:
        return RuntimeError("the model worker failed this request and its answer did not decode")
    message = str(failure.get("message", ""))
    if failure.get("worker_lost"):
        return WorkerLost(message)
    return WorkerError(
        str(failure.get("type", "Exception")), message, str(failure.get("traceback", ""))
    )


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
