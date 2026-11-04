"""The model server's process core: sessions in, batches stepped, answers out.

:class:`ModelServerCore` serves one loaded :class:`~reactor_runtime.model_server.ModelServer`
to the session runtimes that connect through a
:class:`~reactor_runtime.model_server.LinkTransport`. It is a service component,
started, drained and stopped by the runtime's service like every other.

Each session's link has a reader thread, which receives its messages and queues
them, and a writer thread behind its outbox (:mod:`.connection`). One step
thread runs every call into the model: ``session_opened()``,
``session_closed()``, and ``step()`` for the batches the policy forms
(:mod:`.batching`), so the model's state needs no lock against itself. Only
``signal()`` runs on a session's reader thread, and ``health()`` on the
caller's.

A fault stays with the session it belongs to. A second request while one
waits is refused for that session alone; a session that breaks the protocol,
or stops reading its answers, is closed alone; an output that is an
exception, or a result the link cannot carry, fails that request alone. A
``step()`` that raises fails every request of its batch with
:class:`~reactor_runtime.model_server.BatchFailed`, and the model server goes on: it
never steps a request twice.

A request whose output is a :class:`concurrent.futures.Future` stays running
after ``step()`` returns, and keeps its session busy until the future is
done. The future's done-callback, which runs on whichever thread completes
it, finishes the request under the server's lock and posts the answer to the
session's outbox; the server never calls ``cancel()`` or
``add_done_callback()`` while it holds that lock. A request whose timeout
passes, or whose session sends a cancel, is answered at once, but its session
stays busy until the future is done, unless ``cancel()`` shows the work never
started. A reset or a leave releases the session at once and leaves the old
generation's work to ``session_closed()``; until its future is done, it still
counts toward :meth:`ModelServerCore.health`.

A session's messages are applied in the order it sent them. While a
:class:`~reactor_runtime.model_server.protocol.Reset` is pending, the session's
steps are refused with :class:`~reactor_runtime.model_server.ResetRequired`:
a session sends the steps of its new generation only after its
:class:`~reactor_runtime.model_server.protocol.ResetDone`, as
:class:`~reactor_runtime.model_server.ModelClient` does.

The model server's own life is a state machine too, ``serving``, then ``draining``,
then ``stopped``, and every message is answered in every state. Once the
model server drains, a step or a reset is answered at once with
:class:`~reactor_runtime.model_server.ModelServerStopping`, a hello is refused, and
whatever was still queued is answered before the step thread ends. A
connection is tracked from the moment it is accepted: one that does not send
its hello within :data:`HELLO_TIMEOUT` is closed, at most ``max_sessions``
connections may be opening at once, and draining or stopping closes the ones
still opening. An exception that escapes the step thread latches the model server
unhealthy with the reason, so the deployment restarts it.
"""

from __future__ import annotations

import asyncio
import collections.abc
import enum
import functools
import itertools
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from reactor_runtime.core import Health, HealthStatus
from reactor_runtime.log import get_logger
from reactor_runtime.model_server.batching import BatchPolicy, Waiting
from reactor_runtime.model_server.connection import Connection, ConnectionState
from reactor_runtime.model_server.errors import (
    BatchFailed,
    ModelServerRejected,
    ModelServerStopping,
    ResetRequired,
    StepExpired,
    StepStillRunning,
)
from reactor_runtime.model_server.link import LinkListener, LinkTransport
from reactor_runtime.model_server.protocol import (
    Cancel,
    Closed,
    Hello,
    ProtocolError,
    Refused,
    Reset,
    ResetDone,
    Result,
    Signal,
    Step,
    Welcome,
    check_hello,
    failure,
)
from reactor_runtime.model_server.server import ModelServer, ServerRequest, ServerSettings

logger = get_logger(__name__)

DEFAULT_STUCK_SECONDS = 120.0
"""How long one call into the model, or one request's future, may run before health fails."""

HELLO_TIMEOUT = 10.0
"""Seconds an accepted connection may take to send its hello before it is closed."""

_IDLE_WAIT = 0.5

_STOPPING = "model_server_lost: the model server is stopping"


class ServerState(enum.Enum):
    """Where the model server is in its life: serving, draining, or stopped."""

    SERVING = "serving"
    DRAINING = "draining"
    STOPPED = "stopped"


_SERVER_MOVES: dict[ServerState, frozenset[ServerState]] = {
    ServerState.SERVING: frozenset({ServerState.DRAINING, ServerState.STOPPED}),
    ServerState.DRAINING: frozenset({ServerState.STOPPED}),
    ServerState.STOPPED: frozenset(),
}


class ServerMoveError(RuntimeError):
    """The model server was asked to move to a state its sequence does not allow from here."""


@dataclass(frozen=True)
class _Open:
    connection: Connection


@dataclass(frozen=True)
class _Leave:
    connection: Connection


@dataclass(frozen=True)
class _Reset:
    connection: Connection
    request_id: int
    session_id: str | None


@dataclass
class _Pending:
    """A request handed to ``step()`` that has no answer from the model yet.

    ``answered`` means its session already got an answer (its timeout passed,
    it was cancelled), so the model's result is dropped; ``detached`` means
    its session reset or left, so its session is no longer held for it.
    """

    ticket: int
    request: ServerRequest
    waiting: Waiting
    connection: Connection
    since: float
    future: Future[Any] | None = None
    answered: bool = False
    detached: bool = False

    @property
    def awaited(self) -> bool:
        """Whether a session still waits for this request's answer."""
        return not self.answered and not self.detached


class ModelServerCore:
    """Serve one loaded model to the session runtimes that connect, as a service component.

    Args:
        model: The model half, already loaded.
        transport: Opens the links sessions connect through.
        address: Where sessions connect.
        settings: The serving settings (:meth:`ServerSettings.resolve`).
        max_sessions: The deployment's session count: only slots ``0`` to
            ``max_sessions - 1`` are served, and at most that many connections
            may be opening at once.
        stuck_after: Seconds one call into the model, or one request's
            future, may run before :meth:`health` reports the model server
            stuck.
        drain_timeout: Seconds :meth:`drain` waits for running requests
            before it fails the ones left.
    """

    name = "model_server"
    depends_on: tuple[str, ...] = ()

    def __init__(
        self,
        model: ModelServer,
        transport: LinkTransport,
        address: str,
        *,
        settings: ServerSettings,
        max_sessions: int,
        stuck_after: float = DEFAULT_STUCK_SECONDS,
        drain_timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._transport = transport
        self._address = address
        self._settings = settings
        self._max_sessions = max_sessions
        self._stuck_after = stuck_after
        self._drain_timeout = drain_timeout
        self._drain_deadline: float | None = None
        self._policy = BatchPolicy(settings.max_batch, settings.batch_window)
        self._cond = threading.Condition()
        self._state = ServerState.SERVING
        self._events: list[_Open | _Leave | _Reset] = []
        self._connections: dict[int, Connection] = {}
        self._by_slot: dict[int, Connection] = {}
        self._opening: set[Connection] = set()
        self._reset_pending: set[int] = set()
        self._generations: dict[int, int] = {}
        self._numbers = itertools.count(1)
        self._listener: LinkListener | None = None
        self._call: tuple[str, float] | None = None
        self._step_thread: threading.Thread | None = None
        self._broken: str | None = None
        self._pending: dict[int, _Pending] = {}
        self._tickets = itertools.count(1)

    @property
    def state(self) -> ServerState:
        """Where the model server is in its life."""
        return self._state

    # -- the service component ----------------------------------------------------------------

    async def start(self) -> None:
        """Listen for sessions and start the step thread."""
        self._listener = self._transport.listen(self._address)
        self._step_thread = threading.Thread(
            target=self._step_loop, name="model-server-step", daemon=True
        )
        self._step_thread.start()
        threading.Thread(target=self._accept_loop, name="model-server-accept", daemon=True).start()
        logger.info(
            "model server serving",
            address=self._address,
            max_sessions=self._max_sessions,
            max_batch=self._settings.max_batch,
            batch_window=self._settings.batch_window,
        )

    async def drain(self) -> None:
        """Stop taking work, wait for running requests, and fail what is left.

        From the moment this is called, a step or a reset is answered at once
        with :class:`~reactor_runtime.model_server.ModelServerStopping`, and a
        connection that sends its hello is closed without a refusal, so its
        session waits for the model server that replaces this one. The running
        step finishes, and a request whose future is not done yet may still be
        answered, until its own timeout or ``drain_timeout`` seconds after this
        call, whichever comes first. Then every request still running or
        queued, and every pending reset, is answered with
        ``ModelServerStopping``, a future not done yet is cancelled, and every
        connection still opening is closed. Open sessions stay connected until
        :meth:`stop`.
        """
        with self._cond:
            if self._state is ServerState.SERVING:
                self._move(ServerState.DRAINING)
                self._drain_deadline = time.monotonic() + self._drain_timeout
            self._cond.notify_all()
        if self._listener is not None:
            self._listener.close()
        if self._step_thread is not None:
            await asyncio.to_thread(self._step_thread.join)
        self._answer_in_flight()

    async def stop(self) -> None:
        """Fail the requests left and close every session's connection.

        A request still running is answered with ``ModelServerStopping``
        through its session's outbox before the link closes, and its future
        is cancelled; a result that arrives later is dropped.
        """
        with self._cond:
            if self._state is not ServerState.STOPPED:
                self._move(ServerState.STOPPED)
            self._cond.notify_all()
        if self._listener is not None:
            self._listener.close()
        if self._step_thread is not None and self._step_thread.is_alive():
            await asyncio.to_thread(self._step_thread.join)
        self._answer_in_flight()
        with self._cond:
            connections = list(self._connections.values())
        for connection in connections:
            self._leave(connection)

    def health(self) -> Health:
        """Report the model's own health, the model server's own faults, and stuck work.

        Unhealthy when the step thread stopped, when one call into the model
        (``step()``, ``session_opened()``, ``session_closed()``) has run longer
        than ``stuck_after``, or when a request's future has not been done for
        longer than ``stuck_after`` since ``step()`` returned it, whether or not
        its session is still open. A ``health()`` of the model that raises is
        reported as unhealthy.
        """
        if self._broken is not None:
            return Health(HealthStatus.UNHEALTHY, self._broken)
        now = time.monotonic()
        call = self._call
        if call is not None and now - call[1] > self._stuck_after:
            return Health(
                HealthStatus.UNHEALTHY, f"stuck: one {call[0]} has run for {now - call[1]:.0f} s"
            )
        with self._cond:
            oldest = min((p.since for p in self._pending.values() if p.future), default=None)
        if oldest is not None and now - oldest > self._stuck_after:
            return Health(
                HealthStatus.UNHEALTHY,
                f"stuck: a request's future has not been done for {now - oldest:.0f} s",
            )
        try:
            return self._model.health()
        except Exception as exc:
            return Health(HealthStatus.UNHEALTHY, f"health() raised {type(exc).__name__}: {exc}")

    def _move(self, state: ServerState) -> None:
        if state not in _SERVER_MOVES[self._state]:
            raise ServerMoveError(
                f"the model server cannot move from {self._state.value} to {state.value}"
            )
        self._state = state

    def _answer_in_flight(self) -> None:
        """Fail queued and running requests and finish pending events and opening links.

        Runs once the step thread has ended, or on it as it ends after a fault,
        so it is the only thread that calls into the model.
        """
        stopping = ModelServerStopping(_STOPPING)
        with self._cond:
            events, self._events = self._events, []
            dropped = [
                (connection, self._policy.drop(connection.number))
                for connection in self._connections.values()
            ]
            pending = list(self._pending.values())
            self._pending.clear()
            self._policy.finished([p.waiting.connection for p in pending if not p.detached])
            self._cond.notify_all()
            opening = list(self._opening)
            self._opening.clear()
        for item in pending:
            if item.awaited:
                self._answer(item.connection, failure(item.waiting.request_id, stopping))
        _cancel(pending)
        for event in events:
            self._refuse_while_stopping(event)
        for connection, requests in dropped:
            for request in requests:
                self._answer(connection, failure(request.request_id, stopping))
        for connection in opening:
            # Closed, not refused: the session waits for the model server that
            # replaces this one (see _hello).
            if connection.state in (ConnectionState.CONNECTING, ConnectionState.OPENING):
                connection.move(ConnectionState.CLOSING)
                connection.close()

    def _refuse_while_stopping(self, event: _Open | _Leave | _Reset) -> None:
        """Finish a lifecycle event that arrives while the model server stops.

        An open stays among the opening connections, which are closed with the
        others; a leave closes the session; a reset is refused.
        """
        if isinstance(event, _Leave):
            self._leave(event.connection)
        elif isinstance(event, _Reset):
            self._reset_answered(event.connection)
            self._answer(
                event.connection, failure(event.request_id, ModelServerStopping(_STOPPING))
            )

    # -- connections --------------------------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while True:
            try:
                link = self._listener.accept()
            except OSError:
                return
            connection = Connection(link, next(self._numbers))
            with self._cond:
                serving = self._state is ServerState.SERVING
                room = len(self._opening) < self._max_sessions
                if serving and room:
                    self._opening.add(connection)
            if not (serving and room):
                logger.warning(
                    "closing a connection the model server cannot take",
                    reason="stopping" if not serving else "too many sessions opening at once",
                )
                connection.close()
                continue
            threading.Thread(
                target=self._read_loop,
                args=(connection,),
                name=f"model-server-reader-{connection.number}",
                daemon=True,
            ).start()

    def _read_loop(self, connection: Connection) -> None:
        timer = threading.Timer(HELLO_TIMEOUT, self._hello_expired, (connection,))
        timer.daemon = True
        timer.start()
        try:
            hello = connection.link.receive()
            timer.cancel()
            if self._hello(connection, hello):
                while True:
                    self._dispatch(connection, connection.link.receive())
        except (Closed, OSError):
            pass
        except ProtocolError as exc:
            logger.warning(
                "closing a session that broke the protocol", error=str(exc), **connection.where()
            )
        except Exception:
            logger.exception("a session's reader failed", **connection.where())
        finally:
            timer.cancel()
        self._queue(_Leave(connection))

    def _hello_expired(self, connection: Connection) -> None:
        if connection.state is ConnectionState.CONNECTING:
            logger.warning("closing a connection that sent no hello in time", **connection.where())
            connection.link.close()

    def _hello(self, connection: Connection, hello: Any) -> bool:
        if not isinstance(hello, Hello):
            raise ProtocolError(f"a session opened with a {type(hello).__name__}, not a Hello")
        try:
            check_hello(hello)
        except ProtocolError as exc:
            return self._refuse(connection, str(exc))
        if type(hello.slot) is not int or not 0 <= hello.slot < self._max_sessions:
            return self._refuse(
                connection,
                f"slot {hello.slot!r} is out of range: this model server serves slots 0 to "
                f"{self._max_sessions - 1}",
            )
        connection.slot = hello.slot
        with self._cond:
            serving = self._state is ServerState.SERVING
            if serving:
                connection.move(ConnectionState.OPENING)
                self._events.append(_Open(connection))
                self._cond.notify_all()
        if not serving:
            # Closed, not refused: a refusal tells the session it will never be
            # served, while a model server that stops is replaced, and a session
            # whose link closes while it opens waits for the next one.
            logger.info("closing a session that opened while stopping", **connection.where())
            connection.close()
        return serving

    def _refuse(self, connection: Connection, reason: str) -> bool:
        logger.warning("refused a session", reason=reason, **connection.where())
        connection.post(Refused(reason))
        return False

    def _dispatch(self, connection: Connection, message: Any) -> None:
        if isinstance(message, Step):
            self._take(connection, message)
        elif isinstance(message, Cancel):
            self._cancel_request(connection, message.request_id)
        elif isinstance(message, Signal):
            self._signal(connection, message)
        elif isinstance(message, Reset):
            self._take_reset(connection, message)
        else:
            raise ProtocolError(f"a session sent a {type(message).__name__}")

    def _take(self, connection: Connection, step: Step) -> None:
        now = time.monotonic()
        waiting = Waiting(
            connection=connection.number,
            request_id=step.request_id,
            payload=step.payload,
            arrived=now,
            deadline=None if step.timeout is None else now + step.timeout,
        )
        refusal: BaseException | None = None
        with self._cond:
            if self._state is not ServerState.SERVING:
                refusal = ModelServerStopping(_STOPPING)
            elif connection.state is not ConnectionState.OPEN:
                refusal = ModelServerRejected("the session is not open")
            elif connection.number in self._reset_pending:
                refusal = ResetRequired(
                    "a reset is still pending: wait for ResetDone before stepping"
                )
            elif not self._policy.add(waiting):
                refusal = StepStillRunning(
                    "the session's previous request is still waiting or running: retry once it ends"
                )
            else:
                self._cond.notify_all()
        if refusal is not None:
            self._answer(connection, failure(step.request_id, refusal))

    def _cancel_request(self, connection: Connection, request_id: int) -> None:
        """Withdraw a waiting request, or stop answering a running one.

        The session sent the cancel because it gave up on the answer, so a
        running request gets none. Its session stays busy until its future is
        done, unless ``cancel()`` shows the work never started.
        """
        with self._cond:
            if self._policy.cancel(connection.number, request_id):
                return
            running = [
                p
                for p in self._pending.values()
                if p.connection is connection
                and p.waiting.request_id == request_id
                and p.future is not None
                and p.awaited
            ]
            for item in running:
                item.answered = True
        _cancel(running)

    def _take_reset(self, connection: Connection, reset: Reset) -> None:
        with self._cond:
            serving = self._state is ServerState.SERVING
            if serving:
                self._reset_pending.add(connection.number)
                self._events.append(_Reset(connection, reset.request_id, reset.session_id))
                self._cond.notify_all()
        if not serving:
            self._answer(connection, failure(reset.request_id, ModelServerStopping(_STOPPING)))

    def _reset_answered(self, connection: Connection) -> None:
        with self._cond:
            self._reset_pending.discard(connection.number)

    def _signal(self, connection: Connection, message: Signal) -> None:
        if connection.state is not ConnectionState.OPEN:
            return
        try:
            self._model.signal(connection.slot or 0, message.signal)
        except Exception:
            logger.exception("signal() raised", **connection.where())

    def _queue(self, event: _Open | _Leave | _Reset) -> None:
        with self._cond:
            self._events.append(event)
            self._cond.notify_all()

    def _answer(self, connection: Connection, message: Any) -> None:
        if not connection.post(message):
            logger.warning(
                "a session stopped reading its answers; closing it", **connection.where()
            )
            connection.link.close()

    # -- the step thread ----------------------------------------------------------------------

    def _step_loop(self) -> None:
        try:
            self._step_until_drained()
        except Exception as exc:
            logger.exception("the model server's step loop stopped")
            # With no step thread left, nothing would take what is queued or
            # what arrives: stop taking work and answer it all.
            with self._cond:
                self._broken = f"the step loop stopped: {type(exc).__name__}: {exc}"
                if self._state is ServerState.SERVING:
                    self._move(ServerState.DRAINING)
                self._cond.notify_all()
            self._answer_in_flight()

    def _step_until_drained(self) -> None:
        while True:
            with self._cond:
                if self._state is ServerState.STOPPED:
                    return
                now = time.monotonic()
                draining = self._state is ServerState.DRAINING
                if draining:
                    # What is left when this returns is answered by
                    # _answer_in_flight(), events included.
                    deadline = self._drain_deadline
                    assert deadline is not None
                    if now >= deadline or not any(p.awaited for p in self._pending.values()):
                        return
                overdue = self._expire_running(now)
                events, self._events = self._events, []
                turn = None
                if draining:
                    # Draining applies lifecycle events but takes no new batches.
                    if not events and not overdue:
                        self._cond.wait(max(min(self._next_wake(_IDLE_WAIT), deadline - now), 0.0))
                        continue
                elif not events:
                    open_sessions = sum(
                        1 for c in self._connections.values() if c.state is ConnectionState.OPEN
                    )
                    turn = self._policy.take(now, open_sessions)
                    if not turn.batch and not turn.expired and not overdue:
                        wait = _IDLE_WAIT if turn.wake_at is None else turn.wake_at - now
                        self._cond.wait(max(self._next_wake(wait), 0.0))
                        continue
            for item in overdue:
                self._answer(
                    item.connection,
                    failure(
                        item.waiting.request_id,
                        StepExpired("the request's timeout passed while it ran"),
                    ),
                )
            _cancel(overdue)
            for event in events:
                if draining:
                    self._refuse_while_stopping(event)
                else:
                    self._apply(event)
            if turn is None:
                continue
            for request in turn.expired:
                self._post_to(
                    request.connection,
                    failure(
                        request.request_id,
                        StepExpired("the request's timeout passed while it waited in the queue"),
                    ),
                )
            if turn.batch:
                self._run(turn.batch)

    def _expire_running(self, now: float) -> list[_Pending]:
        """Mark running requests whose timeout passed as answered; hold the condition."""
        overdue = [
            p
            for p in self._pending.values()
            if p.future is not None
            and p.awaited
            and p.waiting.deadline is not None
            and p.waiting.deadline <= now
        ]
        for item in overdue:
            item.answered = True
        return overdue

    def _next_wake(self, wait: float) -> float:
        """Shorten *wait* to the earliest timeout of a running request; hold the condition."""
        deadlines = [
            p.waiting.deadline
            for p in self._pending.values()
            if p.future is not None and p.awaited and p.waiting.deadline is not None
        ]
        if not deadlines:
            return wait
        return min(wait, min(deadlines) - time.monotonic())

    def _calling(self, what: str) -> _Timed:
        return _Timed(self, what)

    def _apply(self, event: _Open | _Leave | _Reset) -> None:
        try:
            if isinstance(event, _Open):
                self._open(event.connection)
            elif isinstance(event, _Leave):
                self._leave(event.connection)
            else:
                self._reset(event)
        except Exception:
            logger.exception("a session's open, close or reset failed", **event.connection.where())
            event.connection.link.close()

    def _open(self, connection: Connection) -> None:
        # The connection stays among the opening ones until its open succeeds
        # or fails, so a slow session_opened() still counts toward max_sessions
        # and a drain still sees it.
        if connection.state is not ConnectionState.OPENING:
            with self._cond:
                self._opening.discard(connection)
            return
        assert connection.slot is not None
        held = self._by_slot.get(connection.slot)
        if held is not None and held is not connection:
            # A slot belongs to one session runtime: a new hello for it means the
            # process that held it is gone, or was replaced.
            self._leave(held)
        connection.generation = self._next_generation(connection.slot)
        try:
            with self._calling("session_opened()"):
                self._model.session_opened(connection.slot, connection.generation)
        except Exception as exc:
            with self._cond:
                self._opening.discard(connection)
                serving = self._state is ServerState.SERVING
            logger.exception("session_opened() refused a session", **connection.where())
            if serving:
                # Closed, not refused, once the server stops, as _hello does:
                # the session then waits for the model server that replaces it.
                connection.post(
                    Refused(f"session_opened() refused it: {type(exc).__name__}: {exc}")
                )
            connection.move(ConnectionState.CLOSING)
            connection.close()
            return
        with self._cond:
            self._opening.discard(connection)
            serving = self._state is ServerState.SERVING
            if serving:
                connection.move(ConnectionState.OPEN)
                self._connections[connection.number] = connection
                self._by_slot[connection.slot] = connection
        if not serving:
            # The model server began to stop while the model opened the session:
            # close it, not refuse it, as _hello does, after the model closed the
            # state it just opened.
            try:
                with self._calling("session_closed()"):
                    self._model.session_closed(connection.slot, connection.generation)
            except Exception:
                logger.exception("session_closed() raised", **connection.where())
            connection.move(ConnectionState.CLOSING)
            connection.close()
            return
        logger.info("session opened", generation=connection.generation, **connection.where())
        self._answer(
            connection,
            Welcome(
                slot=connection.slot,
                generation=connection.generation,
                max_batch=self._settings.max_batch,
            ),
        )

    def _leave(self, connection: Connection) -> None:
        with self._cond:
            self._opening.discard(connection)
        if connection.state in (ConnectionState.CLOSING, ConnectionState.CLOSED):
            connection.close()
            return
        was_open = connection.state is ConnectionState.OPEN
        self._forget(connection)
        if was_open:
            assert connection.slot is not None
            try:
                with self._calling("session_closed()"):
                    self._model.session_closed(connection.slot, connection.generation)
            except Exception:
                logger.exception("session_closed() raised", **connection.where())
            logger.info("session closed", generation=connection.generation, **connection.where())
        connection.move(ConnectionState.CLOSED)
        connection.close()

    def _forget(self, connection: Connection) -> None:
        """Move *connection* to closing and drop everything the model server holds for it."""
        connection.move(ConnectionState.CLOSING)
        with self._cond:
            self._policy.drop(connection.number)
            running = self._drop_pending(connection)
            self._reset_pending.discard(connection.number)
            self._connections.pop(connection.number, None)
            if connection.slot is not None and self._by_slot.get(connection.slot) is connection:
                del self._by_slot[connection.slot]
        _cancel(running)

    def _reset(self, event: _Reset) -> None:
        connection = event.connection
        if connection.state is not ConnectionState.OPEN:
            self._reset_answered(connection)
            return
        assert connection.slot is not None
        with self._cond:
            dropped = self._policy.drop(connection.number)
            running = self._drop_pending(connection)
            dropped.extend(p.waiting for p in running if not p.answered)
        _cancel(running)
        for request in dropped:
            self._answer(connection, failure(request.request_id, StepExpired("the session reset")))
        try:
            with self._calling("session_closed()"):
                self._model.session_closed(connection.slot, connection.generation)
        except Exception:
            logger.exception("session_closed() raised", **connection.where())
        connection.generation = self._next_generation(connection.slot)
        connection.session_id = event.session_id
        try:
            with self._calling("session_opened()"):
                self._model.session_opened(connection.slot, connection.generation)
        except Exception as exc:
            # The old state is closed and the new one never opened: the session
            # is refused, as a raise from session_opened() is documented to do,
            # and session_closed() does not run again for this generation.
            logger.exception("session_opened() refused a reset", **connection.where())
            self._reset_answered(connection)
            self._answer(
                connection,
                failure(
                    event.request_id,
                    ModelServerRejected(
                        f"session_opened() refused the reset: {type(exc).__name__}: {exc}"
                    ),
                ),
            )
            self._forget(connection)
            connection.move(ConnectionState.CLOSED)
            connection.close()
            return
        connection.move(ConnectionState.OPEN)
        self._reset_answered(connection)
        logger.info("session reset", generation=connection.generation, **connection.where())
        self._answer(connection, ResetDone(event.request_id, connection.generation))

    def _next_generation(self, slot: int) -> int:
        self._generations[slot] = self._generations.get(slot, 0) + 1
        return self._generations[slot]

    def _post_to(self, number: int, message: Any) -> None:
        with self._cond:
            connection = self._connections.get(number)
        if connection is not None:
            self._answer(connection, message)

    def _run(self, batch: list[Waiting]) -> None:
        with self._cond:
            live = [
                (request, connection)
                for request in batch
                if (connection := self._connections.get(request.connection)) is not None
            ]
        if not live:
            # Every session of the batch left before it ran: step() is never
            # called with no request.
            with self._cond:
                self._policy.finished([r.connection for r in batch])
            return
        now = time.monotonic()
        pending = [
            _Pending(
                next(self._tickets),
                ServerRequest(c.slot or 0, c.generation, r.payload, c.session_id),
                r,
                c,
                now,
            )
            for r, c in live
        ]
        with self._cond:
            self._pending.update((p.ticket, p) for p in pending)
        try:
            with self._calling("step()"):
                outputs = self._model.step([p.request for p in pending])
            if not _is_sequence(outputs) or len(outputs) != len(pending):
                count = len(outputs) if _is_sequence(outputs) else type(outputs).__name__
                raise RuntimeError(f"step() returned {count} outputs for {len(pending)} requests")
        except Exception as exc:
            logger.exception("step() raised; failing its batch", size=len(pending))
            with self._cond:
                for item in pending:
                    if self._pending.get(item.ticket) is item:
                        self._complete(item, BatchFailed(exc))
            return
        futures: list[tuple[int, Future[Any]]] = []
        with self._cond:
            returned = time.monotonic()
            for item, output in zip(pending, outputs, strict=True):
                if self._pending.get(item.ticket) is not item:
                    continue
                if isinstance(output, Future):
                    # Registered before its callback, which runs at once on a
                    # future that is already done.
                    item.future = output
                    item.since = returned
                    futures.append((item.ticket, output))
                else:
                    self._complete(item, output)
            self._cond.notify_all()
        for ticket, future in futures:
            future.add_done_callback(functools.partial(self._resolve, ticket))

    def _resolve(self, ticket: int, future: Future[Any]) -> None:
        """Finish the request behind *ticket* from its done future, on the completing thread."""
        try:
            if future.cancelled():
                output: Any = StepExpired("the request was cancelled")
            else:
                error = future.exception()
                output = future.result() if error is None else error
            with self._cond:
                item = self._pending.get(ticket)
                if item is not None:
                    self._complete(item, output)
        except Exception as exc:
            # concurrent.futures only logs what a callback raises, and the
            # session would stay busy for good: make the fault visible.
            logger.exception("finishing a request's future failed")
            with self._cond:
                self._broken = f"finishing a request's future failed: {type(exc).__name__}: {exc}"

    def _complete(self, item: _Pending, output: Any) -> None:
        """Finish a pending request while holding the server's condition."""
        del self._pending[item.ticket]
        connection = item.connection
        if (
            item.awaited
            and connection.state is ConnectionState.OPEN
            and connection.generation == item.request.generation
            and self._connections.get(connection.number) is connection
        ):
            self._answer(connection, self._outcome(item.waiting.request_id, output))
        else:
            logger.debug(
                "dropping a result its session no longer waits for",
                generation=item.request.generation,
                request_id=item.waiting.request_id,
                **connection.where(),
            )
        if not item.detached:
            self._policy.finished([connection.number])
        self._cond.notify_all()

    def _drop_pending(self, connection: Connection) -> list[_Pending]:
        """Detach a generation's running requests while holding the server's condition.

        They stay pending until their futures are done, so :meth:`health`
        still sees work the model never finished.
        """
        # Answered requests too: their session was held for them, and a result
        # that arrives after the reset must not free the next generation's.
        running = [
            p for p in self._pending.values() if p.connection is connection and not p.detached
        ]
        for item in running:
            item.detached = True
        self._cond.notify_all()
        return [p for p in running if p.future is not None]

    def _outcome(self, request_id: int, output: Any) -> Any:
        if isinstance(output, BaseException):
            return failure(request_id, output)
        return Result(request_id, output)


class _Timed:
    """Time one call into the model, for :meth:`ModelServerCore.health`."""

    def __init__(self, server: ModelServerCore, what: str) -> None:
        self._server = server
        self._what = what

    def __enter__(self) -> None:
        self._server._call = (self._what, time.monotonic())

    def __exit__(self, *exc: object) -> None:
        self._server._call = None


def _cancel(items: list[_Pending]) -> None:
    """Cancel the futures of *items*, never while holding the server's condition."""
    for item in items:
        if item.future is not None:
            item.future.cancel()


def _is_sequence(outputs: Any) -> bool:
    return isinstance(outputs, collections.abc.Sequence) and not isinstance(
        outputs, str | bytes | bytearray
    )
