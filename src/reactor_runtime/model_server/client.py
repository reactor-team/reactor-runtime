"""The application half's connection to its model server.

A session runtime's application opens one :class:`ModelClient` in its
``load()`` and calls :meth:`ModelClient.step` from ``generate()``. The client
reports and never decides: it holds no reference to the application, does
nothing in the background but read its link, and never reconnects. A failure
reaches the application on its own call path, in ``process_output()``, and a
lost link also reaches the ``on_lost`` callback the application passed to
:meth:`ModelClient.connect`, so the application decides what each means.
"""

from __future__ import annotations

import enum
import itertools
import threading
import time
from collections.abc import Callable
from typing import Any

from reactor_runtime import log
from reactor_runtime.distributed.errors import WorkerTimeout
from reactor_runtime.log import get_logger, get_session_id
from reactor_runtime.model_server.config import LinkConfig, link_config
from reactor_runtime.model_server.errors import (
    ClientClosed,
    ResetRequired,
    StepTimedOut,
    WorkerLost,
    WorkerRejected,
)
from reactor_runtime.model_server.link import LinkTransport, NoListener, WorkerLink
from reactor_runtime.model_server.protocol import (
    Cancel,
    Closed,
    Failed,
    Hello,
    Refused,
    Reset,
    ResetDone,
    Result,
    Signal,
    Step,
    Welcome,
    error_of,
)
from reactor_runtime.model_server.shm_link import UnixShmTransport

logger = get_logger(__name__)

_POLL_FIRST = 0.2
_POLL_MAX = 2.0


class LinkState(enum.Enum):
    """Where a client's link to its model server is."""

    OPEN = "open"
    LOST = "lost"
    CLOSED = "closed"


class ModelClient[RequestT, ResultT]:
    """One session runtime's connection to its model server.

    Build it with :meth:`connect`. One :meth:`step` or :meth:`reset` runs at a
    time; a second caller waits for the first. :meth:`signal` and
    :meth:`close` may be called from any thread. Its threads are daemons, and
    every wait it makes is bounded or ended by :meth:`close`.

    What :meth:`step` sends and returns, and what :meth:`signal` sends, is
    pickled as it is. The type parameters are for type checkers alone, and
    nothing checks them at run time.

    Every failure is a raise from the call that sees it:

    * the model's own exception for a request it failed, as itself, and
      :class:`~reactor_runtime.model_server.BatchFailed` when the model's
      ``step()`` raised for the whole batch;
    * :class:`~reactor_runtime.model_server.WorkerRejected` when the worker
      refused the request or the session, and its subclass
      :class:`~reactor_runtime.model_server.StepStillRunning` when a step this
      client abandoned after a timeout is still running there (retry once it
      ends);
    * :class:`~reactor_runtime.model_server.StepTimedOut` when no answer came
      in time: the request is cancelled, which withdraws it if it is still
      queued, and the client stays usable; and
      :class:`~reactor_runtime.model_server.StepExpired` when the request's
      timeout passed in the worker's queue;
    * :class:`~reactor_runtime.model_server.ResetRequired` from a step before
      the application has reset its slot;
    * :class:`~reactor_runtime.model_server.PayloadTooLarge` for a request, or
      its result, larger than the shared memory ``reactor.yaml`` gives it, and
      the pickling error for a request that does not pickle (the client stays
      usable in both cases);
    * :class:`~reactor_runtime.model_server.WorkerLost` once the link is lost,
      from every call from then on.

    A lost link stays lost: the client does not reconnect. The application
    re-raises in ``process_output()``, its session ends, the session runtime
    exits, and its restart waits in ``load()`` until the model server is back::

        class Plasma(ReactorApp):
            def load(self, config_path):
                # Waits for the model server; a loss between steps ends the runtime.
                self.model: ModelClient[Tick, Frame] = ModelClient.connect(on_lost=self.crash)

            @session_started
            def begin(self):
                self.model.reset()                         # fresh state for this session

            def generate(self, input):
                return PlasmaOutput(main_video=self.model.step(Tick(tick=self.tick)).pixels)

            async def process_output(self, outcome):
                if isinstance(outcome.error, (WorkerLost, StepUnanswered)):
                    raise outcome.error                    # down; the restart waits
                ...

    A link lost while no step runs, between sessions for one, raises from no
    call. ``on_lost=self.crash`` is how the application hears of that loss and
    goes down the same way, through ``ReactorApp.crash()``.

    Once connected, every line the session runtime writes carries
    ``runtime_role="session"`` and its ``slot``, and the model server's lines
    about the slot carry the same slot and the session id of its last reset,
    so one session's lines and the model server's lines for it can be joined.

    The model server's own health closes the loop: a model that reports itself
    unhealthy is restarted by its deployment, which drops every session's link,
    and each session runtime then goes down and comes back as above.
    """

    def __init__(
        self,
        link: WorkerLink,
        welcome: Welcome,
        config: LinkConfig,
        on_lost: Callable[[WorkerLost], None] | None = None,
    ) -> None:
        """Hold an open link; use :meth:`connect`."""
        self._link = link
        self._config = config
        self._on_lost = on_lost
        self._slot = welcome.slot
        self._generation = welcome.generation
        self._state = LinkState.OPEN
        # What every call raises once the link is lost or closed: a fresh
        # exception per call, so no traceback chain grows on a stored one.
        self._ended: tuple[type[WorkerLost], str] | None = None
        self._reset_required = True
        self._ids = itertools.count(1)
        self._replies: dict[int, Any] = {}
        self._waiting: int | None = None
        self._cond = threading.Condition()
        self._call_lock = threading.Lock()
        threading.Thread(target=self._read, name="model-client-reader", daemon=True).start()
        log.set_process(role="session", slot=self._slot)
        logger.info("model server link", generation=self._generation)

    @classmethod
    def connect(
        cls,
        config: LinkConfig | None = None,
        *,
        on_lost: Callable[[WorkerLost], None] | None = None,
        transport: LinkTransport | None = None,
    ) -> ModelClient[RequestT, ResultT]:
        """Connect to the model server and block until it has opened this session.

        Called from the application's ``load()``: while it waits, the session
        runtime is loading, so a cold start and a model server restart both look
        like a slow model load. It waits through a model server that is not
        listening yet, or that closes the link before it opens the session.

        Args:
            config: Where the model server is and which slot this session
                serves; :func:`~reactor_runtime.model_server.config.link_config`
                reads it from the environment when omitted.
            on_lost: Called once, with a :class:`WorkerLost` carrying the message
                every later call raises, when the open link is lost; never for
                :meth:`close`. It
                runs on the thread that saw the loss, which can be the client's
                reader or the caller of a :meth:`step`, so it must be safe to
                call from any thread and return quickly. An exception it raises
                is logged and dropped.
            transport: How the link is made; the shared-memory link over a Unix
                socket when omitted.

        The configured connect timeout bounds the whole wait, the link's own
        setup with the model server included.

        Raises:
            WorkerUnreachable: At once, for a deployment that cannot reach a
                model server however long it waits.
            WorkerRejected: The model server refused this session.
            WorkerTimeout: No model server opened the session within the
                configured connect timeout.
        """
        config = config or link_config()
        transport = transport or UnixShmTransport(session_directory=config.shm_dir)
        deadline = (
            None if config.connect_timeout is None else time.monotonic() + config.connect_timeout
        )
        poll = _POLL_FIRST
        last = "no model server listens yet"
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise WorkerTimeout(
                    f"no model server opened this session within {config.connect_timeout} s: {last}"
                )
            try:
                link = transport.connect(config.socket, config.slot, timeout=remaining)
            except NoListener as exc:
                last = str(exc)
            else:
                try:
                    link.send(Hello(slot=config.slot))
                    answer = _receive_until(link, deadline)
                except TimeoutError:
                    link.close()
                    last = "the model server accepted the link but did not open the session"
                except Exception as exc:
                    link.close()
                    last = f"the model server closed the link before it opened the session: {exc}"
                else:
                    if isinstance(answer, Welcome):
                        return cls(link, answer, config, on_lost)
                    link.close()
                    if isinstance(answer, Refused):
                        raise WorkerRejected(
                            f"the model server refused this session: {answer.reason}"
                        )
                    last = f"the model server answered the hello with a {type(answer).__name__}"
            if deadline is not None and time.monotonic() >= deadline:
                raise WorkerTimeout(
                    f"no model server opened this session within {config.connect_timeout} s: {last}"
                )
            wait = poll if deadline is None else min(poll, max(deadline - time.monotonic(), 0.0))
            time.sleep(wait)
            poll = min(poll * 2, _POLL_MAX)

    @property
    def slot(self) -> int:
        """The slot this session holds on the model server."""
        return self._slot

    @property
    def generation(self) -> int:
        """The slot's generation: it moves on at every reset."""
        return self._generation

    @property
    def state(self) -> LinkState:
        """Whether the link is open, lost, or closed."""
        return self._state

    def step(self, request: RequestT, *, timeout: float | None = None) -> ResultT:
        """Send *request* to the model's next step and return this session's result.

        A step that times out is cancelled, which withdraws it if it is still
        queued on the model server. One the model is already running keeps the
        session's one request until it ends, and a :meth:`step` meanwhile
        raises :class:`~reactor_runtime.model_server.StepStillRunning`.

        Args:
            request: What the model's ``step()`` receives as this session's payload.
            timeout: Seconds to wait for the answer. ``None`` uses the configured
                step timeout, and ``0`` waits without a bound, as ``0`` does in
                the environment.

        Returns:
            What the model's ``step()`` returned for this request, whose arrays
            are its own.

        Raises:
            ValueError: *timeout* is negative; nothing is sent.
        """
        wait = self._wait(timeout)
        with self._call_lock:
            self._usable()
            if self._reset_required:
                raise ResetRequired(
                    "reset() has not completed on this link: the application resets its slot "
                    "when its session starts"
                )
            request_id = next(self._ids)
            answer = self._call(request_id, Step(request_id, wait, request), wait)
            if answer is None:
                self._cancel(request_id)
                raise StepTimedOut(f"the model server gave no answer within {wait} s")
            if isinstance(answer, Result):
                return answer.output
            raise error_of(answer)

    def reset(self, *, session_id: str | None = None, timeout: float | None = None) -> int:
        """Close this slot's session on the model server and open a new one.

        The model server runs ``session_closed()`` then ``session_opened()``
        for the slot. Only the application calls it, when its session starts;
        until it completes, :meth:`step` raises
        :class:`~reactor_runtime.model_server.ResetRequired`.

        Args:
            session_id: The id the model server puts on its log lines about this
                slot until the next reset; the live session's id when omitted.
            timeout: Seconds to wait. ``None`` uses the configured step timeout,
                and ``0`` waits without a bound.

        Returns:
            The slot's new generation.

        Raises:
            ValueError: *timeout* is negative; nothing is sent.
        """
        wait = self._wait(timeout)
        with self._call_lock:
            self._usable()
            self._reset_required = True
            request_id = next(self._ids)
            reset = Reset(request_id, session_id or get_session_id())
            answer = self._call(request_id, reset, wait)
            if answer is None:
                raise StepTimedOut(f"the model server did not reset within {wait} s")
            if isinstance(answer, ResetDone):
                self._generation = answer.generation
                self._reset_required = False
                logger.info("model server link", generation=self._generation)
                return answer.generation
            raise error_of(answer)

    def signal(self, signal: Any) -> None:
        """Deliver *signal* to the model's ``signal()`` now, even while a step waits."""
        self._usable()
        self._send(Signal(signal))

    def close(self) -> None:
        """Leave the model server and release this client's memory. Idempotent."""
        with self._cond:
            if self._state is LinkState.CLOSED:
                return
            self._state = LinkState.CLOSED
            self._ended = (ClientClosed, "the model client is closed")
            self._cond.notify_all()
        self._link.close()

    def _wait(self, timeout: float | None) -> float | None:
        if timeout is None:
            return self._config.step_timeout
        if timeout < 0:
            raise ValueError(f"timeout must be at least 0 seconds, or None, got {timeout!r}")
        return None if timeout == 0 else timeout

    def _usable(self) -> None:
        with self._cond:
            self._raise_if_ended()

    def _raise_if_ended(self) -> None:
        """Raise a fresh exception for a lost or closed link; the caller holds the condition."""
        if self._ended is not None:
            kind, message = self._ended
            raise kind(message)

    def _send(self, message: Any) -> None:
        """Send *message*; only a link that is gone makes the client lost.

        ``PayloadTooLarge``, or any error encoding the message (one that does
        not pickle), was raised before anything was sent: it reaches the caller
        and the client stays open.
        """
        try:
            self._link.send(message)
        except (Closed, OSError) as exc:
            self._lose(f"the link failed: {exc}")
            self._usable()

    def _cancel(self, request_id: int) -> None:
        """Withdraw a request that timed out; a cancel that cannot be sent is logged."""
        try:
            self._send(Cancel(request_id))
        except WorkerLost:
            raise
        except Exception as exc:
            logger.warning(
                "could not cancel a request that timed out",
                slot=self._slot,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _call(self, request_id: int, message: Any, wait: float | None) -> Any:
        with self._cond:
            self._waiting = request_id
        try:
            self._send(message)
            deadline = None if wait is None else time.monotonic() + wait
            with self._cond:
                while request_id not in self._replies:
                    self._raise_if_ended()
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return None
                    self._cond.wait(remaining)
                return self._replies.pop(request_id)
        finally:
            with self._cond:
                self._waiting = None
                self._replies.clear()

    def _read(self) -> None:
        reason = "the model server closed the link"
        try:
            while True:
                message = self._link.receive()
                if isinstance(message, Result | Failed | ResetDone):
                    with self._cond:
                        if message.request_id == self._waiting:
                            self._replies[message.request_id] = message
                            self._cond.notify_all()
        except Exception as exc:
            if self._state is not LinkState.CLOSED:
                reason = f"{reason}: {exc}" if str(exc) else reason
        self._lose(reason)

    def _lose(self, reason: str) -> None:
        with self._cond:
            if self._state is not LinkState.OPEN:
                return
            self._state = LinkState.LOST
            message = f"worker_died: {reason}"
            self._ended = (WorkerLost, message)
            lost = WorkerLost(message)
            self._cond.notify_all()
        logger.error("lost the model server", slot=self._slot, reason=reason)
        self._link.close()
        if self._on_lost is not None:
            try:
                self._on_lost(lost)
            except Exception:
                logger.exception("the on_lost callback raised", slot=self._slot)


def _receive_until(link: WorkerLink, deadline: float | None) -> Any:
    """Receive one message, giving up at *deadline*: a model server opens a session between steps.

    Raises:
        TimeoutError: Nothing arrived by *deadline*; the link is closed.
    """
    if deadline is None:
        return link.receive()
    outcome: list[Any] = []
    failed: list[BaseException] = []

    def wait() -> None:
        try:
            outcome.append(link.receive())
        except BaseException as exc:
            failed.append(exc)

    waiter = threading.Thread(target=wait, name="model-client-welcome", daemon=True)
    waiter.start()
    waiter.join(max(deadline - time.monotonic(), 0.0))
    if waiter.is_alive():
        link.close()
        waiter.join(1.0)
        raise TimeoutError("no answer to the hello in time")
    if failed:
        raise failed[0]
    return outcome[0]
