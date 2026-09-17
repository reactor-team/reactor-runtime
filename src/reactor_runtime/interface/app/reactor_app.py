"""The application authoring base, :class:`ReactorApp`.

What an author subclasses. It joins the two halves of the model layer: the
:class:`ReactorCore` engine it inherits (thread, loop, buffers, queues) and the
:class:`ModelContract` it assembles. Declaring a subclass resolves the contract
once, from a single traversal of the class, and caches it on the class — the
commands its ``@event`` handlers expose, the messages they return, its tracks,
and its lifecycle hooks.

This class supplies the *what* the engine leaves open: the two dispatch loops
that drain the engine's typed queues into handlers. The command loop validates
nothing — that happened at the bridge — and turns each :class:`CommandEnvelope`
back into a handler call, replying with the handler's returned message to the
one connection that sent the command. The reactor loop runs the lifecycle hooks
and maintains :attr:`connected` from the live client count.

It also owns the typed, client-settable state. An application that declares
``state: MyState`` (an :class:`InputState` subclass) gets one ``set_<field>``
command per public field, stamped onto the class before the contract is built
so the same contract that powers ``@event`` handlers discovers, validates, and
documents it. ``self.state`` is session-scoped: built from field defaults when
a session starts, before ``@session_started`` runs, and cleared after
``@session_ended`` returns.

Two pieces of loop-bound state go with it. The step lock serializes every
command handler and lifecycle hook, so a ``run()`` that takes the same lock
around one unit of work never has a handler land inside it. The live gate is
set while a session has started and at least one client is connected; a
``run()`` waits on it and checks it between units of work.

The default ``run()`` is the step loop. Each turn takes the step lock and calls
``process_input()``, ``generate()``, and ``process_output()`` in that order, then
emits the media the step produced. An author who needs a different loop
overrides ``run()``. That replaces the loop and only the loop: the dispatch
layer above stays, and the three hooks are never called for that class.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Coroutine
from typing import Any, ClassVar, get_type_hints

from reactor_runtime.codes import INTERNAL_ERROR
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    FileUploaded,
    ReactorEvent,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import CommandFailure, ConnId
from reactor_runtime.interface.app.input_state import InputState
from reactor_runtime.interface.app.outcome import StepOutcome
from reactor_runtime.interface.client import ClientInfo
from reactor_runtime.interface.events.decorators import (
    EVENT_ATTR,
    RESERVED_PARAMS,
    EventHandler,
    make_command,
)
from reactor_runtime.interface.events.errors import ApplicationError, CommandError
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.reactor_core import (
    CommandEnvelope,
    ReactorCore,
    RequestId,
)
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.interface.tracks import Output
from reactor_runtime.log import get_logger, release_session_id

logger = get_logger(__name__)

# A refused step waits this long before the loop asks again. Refusing is one
# call and one raise, so without the wait a paused application spins a core
# on the model thread; with it, a change of state is noticed within a frame
# period. The same pause ReactorPipeline takes on an idle yield.
_REFUSED_SLEEP = 0.005


class ReactorApp(ReactorCore):
    """Base class an author subclasses to define the application the runtime drives.

    Write ``generate()`` and the runtime drives it one step at a time. Override
    ``process_input()`` to refuse a step or shape what the model gets, and
    ``process_output()`` to turn the model's result into media and messages.
    Decorate methods with ``@event`` to expose commands, and with the lifecycle
    decorators to hook session and connection events — ``@session_started`` is
    the hook for once-per-session initialization. Declaring the subclass
    resolves the contract and caches it on the class, reachable through
    :meth:`ModelContract.of`.

    Override ``run()`` to write your own loop against ``emit()``, ``send()``,
    ``@event``, :attr:`connected`, and the tracks. The three step hooks are then
    not called.

    Class attributes:
        fps: The nominal rate, in frames per second, an emitted chunk plays out
            at. Declare it to pin playout; leave it out and the step loop paces
            playout from the measured ``generate()`` time. A hand-written
            ``run()`` that passes ``compute_time`` to :meth:`emit` paces itself
            and this is only the fallback.
        state: Optional. Annotate with an :class:`InputState` subclass to declare
            the client-settable state. Every public field becomes a
            ``set_<field>`` command; a hand-written ``@event`` of the same name
            wins over the generated one.

    Lifecycle:
        connected: An :class:`asyncio.Event` set while at least one client is
            connected and cleared when the last one leaves, so a ``run`` loop can
            gate generation on having an audience.
        state: The live :class:`InputState` instance while a session is live
            and ``None`` between sessions, on a class that declares ``state:``.
            A class that declares no ``state:`` owns the attribute itself; the
            base class never writes it.
    """

    __reactor_contract__: ClassVar[ModelContract]
    __app_state__: ClassVar[type[InputState] | None] = None

    state: Any = None
    connected: asyncio.Event
    _clients: dict[ConnId, ClientInfo]
    _session_active: bool
    _live: asyncio.Event
    _step_lock: asyncio.Lock
    _step_requested: asyncio.Event
    _gate_drops: int

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Stamp the state's auto-setters before the contract is built, so they
        # are discovered as ordinary commands.
        state_cls = _resolve_state_class(cls)
        if state_cls is not None:
            cls.__app_state__ = state_cls
            _stamp_auto_setters(cls, state_cls)
        cls.__reactor_contract__ = ModelContract.build(cls)

    def __init__(self) -> None:
        super().__init__()
        # Only an app that declared `state:` has its attribute owned here. One
        # that did not may use the name for its own purposes, and the base
        # class never touches it.
        if self.__app_state__ is not None:
            self.state = None

    # -- the step, three calls ------------------------------------------------

    async def process_input(self) -> Any:
        """Decide whether a step can happen now and what the model gets.

        The application half of a step. Read ``self.state``, drain the
        :class:`MediaInput` holder the class declared with ``try_read()``, and
        return the step input ``generate()`` receives. Raise
        :class:`ApplicationError` with the reason to refuse the step; the model
        is not called and the loop asks again.

        Runs under the step lock. Does not call the model, ``emit()``,
        ``send()``, or ``flush()``.

        Returns:
            The step input. The type is the author's. The default returns
            ``self.state``, ``None`` when no state is declared, and never refuses.
        """
        return getattr(self, "state", None)

    def generate(self, input: Any, /) -> Any:
        """Run one step of inference.

        The model half of a step. Synchronous; blocking GPU work is expected.
        Reads its argument and its own attributes, never ``self.state`` or the
        media tracks. Returns the step result, an :class:`Output` when the
        default ``process_output()`` is used, or raises the model's own exception
        when the step is invalid for the model. A raise is not a refusal;
        refusing is :class:`ApplicationError` in :meth:`process_input`.

        Args:
            input: What :meth:`process_input` returned.

        Returns:
            The step result. The type is the author's.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define generate(), or override run() to drive "
            "the model with its own loop."
        )

    async def process_output(self, outcome: StepOutcome, /) -> Output | None:
        """Turn what ``generate()`` did into what the client receives.

        The application half again. Receives the :class:`StepOutcome` the
        runtime built: ``outcome.result`` when ``generate()`` returned,
        ``outcome.error`` when it raised. Run the step's effects here: ``await
        self.send()`` for a message, which goes on the wire before the step's
        media; ``self.output.flush()``; recovery from a model error. Return the
        :class:`Output` to emit, or ``None`` to emit nothing.

        Runs under the step lock.

        This is the one place a model failure is decided. When ``outcome.error``
        is set, either recover or re-raise:

        * Recover an error the model is known to raise: reset the model half,
          send a message, ``flush()`` if the picture must cut, and return
          ``None`` or an :class:`Output`. The loop continues with the next step.
        * Re-raise anything else. A raise out of this method is a crash of the
          model, not of the step: the runtime logs the traceback, stops the
          command and lifecycle dispatchers, and ends the session with an error
          the client sees. The loop is not restarted; whatever runs the process
          decides whether to restart it. This is the same outcome an uncaught
          exception in a hand-written ``run()`` has.

        The default re-raises, so a model whose ``generate()`` fails ends the
        session loudly instead of serving a dead model in silence. A refusal is
        not a failure: :class:`ApplicationError` from :meth:`process_input` never
        reaches this method.

        Args:
            outcome: What ``generate()`` did.

        Returns:
            The media to emit on the declared tracks, or ``None``. The default
            re-raises an error and otherwise returns ``outcome.to_output()``.
        """
        if outcome.error is not None:
            raise outcome.error
        return outcome.to_output()

    async def run(self) -> None:
        """Drive the model one step at a time. Override for a different loop.

        Steps run while a session is live and at least one client is connected.
        Each turn waits for a step request, takes the step lock, runs the three
        hooks, releases the lock, emits the media the step produced, yields once
        so handlers already waiting get their turn, and requests the next step.
        A productive step is not paced here: a fast model waits in :meth:`emit`
        on a full wire. A refused step waits a few milliseconds before the next
        request, so a paused application does not spin a core.

        Playout is paced from the measured ``generate()`` time unless the author
        declares ``fps``. Whether ``fps`` is pinned is read when the loop starts,
        so a ``load()`` that assigns it counts.

        When the gate drops, the input buffers reset, so the next session or
        client starts from empty tracks. A drop and a re-set that both land
        while a step is blocked in :meth:`emit` still count as a drop: the loop
        compares the number of drops, not the gate's current value.

        Raises:
            Exception: Whatever :meth:`process_output` raised, which by default is
                the error :meth:`generate` raised. It ends the model loop: the
                runtime reports the crash, ends the session with an error, and
                does not start the loop again. A model that expects an error
                recovers from it in :meth:`process_output` instead.
        """
        fps_pinned = _fps_is_author_pinned(type(self))
        last_refusal: str | None = None
        self._step_requested.set()
        while True:
            await self._live.wait()
            drops = self._gate_drops
            try:
                while self._live.is_set() and self._gate_drops == drops:
                    # 1. Wait for a step request. While this await is pending the
                    #    event loop runs the handlers that arrived; each takes the
                    #    step lock, so none can run once the lock below is held.
                    await self._step_requested.wait()
                    self._step_requested.clear()

                    # The step lock is held from the first line of process_input()
                    # to the return of process_output(). Both hooks are async and
                    # may await; the lock is what stops a handler from landing in
                    # one of those gaps.
                    async with self._step_lock:
                        # 2. The application gate.
                        try:
                            input = await self.process_input()
                        except ApplicationError as refused:
                            # One record per change of reason, not one per turn.
                            reason = f"{type(refused).__name__}: {refused}"
                            if reason != last_refusal:
                                logger.debug(
                                    "step refused",
                                    reason=str(refused),
                                    kind=type(refused).__name__,
                                )
                                last_refusal = reason
                            media = None
                            outcome = None
                        else:
                            last_refusal = None
                            # 3. The model. Synchronous on purpose: nothing changes
                            #    under it. This is the one place a StepOutcome is
                            #    built.
                            started = time.perf_counter()
                            try:
                                result = self.generate(input)
                            except Exception as error:
                                outcome = StepOutcome(
                                    error=error, elapsed=time.perf_counter() - started
                                )
                            else:
                                outcome = StepOutcome(
                                    result=result, elapsed=time.perf_counter() - started
                                )

                            # 4. The application collects the outcome into media,
                            #    sends its messages, or recovers. A raise ends the
                            #    loop.
                            media = await self.process_output(outcome)

                    # 5. Emit outside the lock, so a handler can run while the
                    #    wire is full.
                    if media is not None and outcome is not None:
                        pace = None if fps_pinned else outcome.elapsed
                        await self.emit(media, compute_time=pace)

                    # A refused turn waits a little before asking again; a
                    # productive turn yields once so handler tasks already
                    # runnable get their turn, then asks again right away.
                    await asyncio.sleep(_REFUSED_SLEEP if outcome is None else 0)
                    self._step_requested.set()
            finally:
                for buffer in self._input_buffers.values():
                    buffer.reset()

    # -- engine hooks ---------------------------------------------------------

    def _on_loop_ready(self) -> None:
        """Create the loop-bound state the dispatchers and the step loop share."""
        self.connected = asyncio.Event()
        self._clients = {}
        self._session_active = False
        self._live = asyncio.Event()
        self._step_lock = asyncio.Lock()
        self._step_requested = asyncio.Event()
        self._gate_drops = 0

    def _background_coros(self) -> list[Coroutine[Any, Any, None]]:
        """Run the two queue-drain loops alongside ``run()``."""
        return [self._command_loop(), self._reactor_loop()]

    # -- command dispatch -----------------------------------------------------

    async def _command_loop(self) -> None:
        """Drain validated commands and dispatch each to its handler."""
        queue = self._command_q
        assert queue is not None
        while True:
            envelope = await queue.get()
            await self._dispatch_command(envelope)

    async def _dispatch_command(self, envelope: CommandEnvelope) -> None:
        """Invoke a command's handler and reply to the sender with its return.

        The command arrived validated, so its fields populate the handler call
        directly; any reserved parameter the handler declares is injected. When
        the handler returns a :class:`ModelMessage`, it is sent addressed to the
        connection that issued the command — never broadcast — correlated with
        the command's request id; a handler that returns nothing is answered
        with a bodyless acknowledgement so an awaiting client still resolves.

        A handler that raises answers with a failure instead. A
        :class:`CommandError` carries the author's own code and message to the
        client unchanged. Any other exception answers with ``internal_error`` and
        keeps its detail in the log, because the text of an unplanned exception
        can name paths, queries, or credentials. Either way the loop survives,
        and the client resolves rather than waits.

        The handler runs under the step lock, so it lands between two units of
        a ``run()`` loop that holds the same lock, never inside one.
        """
        async with self._step_lock:
            await self._run_command(envelope)

    async def _run_command(self, envelope: CommandEnvelope) -> None:
        """Call the handler and answer the sender. Caller holds the step lock."""
        command = envelope.command
        spec = self.__reactor_contract__.commands.get(type(command).name)
        if spec is None:
            return
        kwargs: dict[str, Any] = {
            name: getattr(command, name) for name in type(command).__command_fields__
        }
        for name in spec.reserved:
            kwargs[name] = self._reserved(name, envelope.conn_id)
        try:
            if spec.is_async:
                result = await spec.handler(self, **kwargs)
            else:
                result = spec.handler(self, **kwargs)
        except CommandError as exc:
            logger.warning(
                "command handler reported a failure",
                command=spec.name,
                code=exc.code,
            )
            self._fail(envelope, CommandFailure(exc.code, exc.message))
            return
        except Exception:
            logger.exception("error in command handler", command=spec.name)
            self._fail(
                envelope,
                CommandFailure(INTERNAL_ERROR, "The handler raised an unexpected error."),
            )
            return
        if envelope.conn_id is None:
            return
        if isinstance(result, ModelMessage):
            self._reply(envelope.conn_id, result, envelope.request_id)
        elif envelope.request_id is not None:
            self._reply(envelope.conn_id, None, envelope.request_id)

    # -- reactor-event dispatch -----------------------------------------------

    async def _reactor_loop(self) -> None:
        """Drain authoritative reactor events and run their lifecycle hooks."""
        queue = self._reactor_q
        assert queue is not None
        while True:
            event = await queue.get()
            await self._dispatch_reactor_event(event)

    async def _dispatch_reactor_event(self, event: ReactorEvent) -> None:
        """Run the lifecycle hook for one reactor event and track liveness.

        Connection events keep :attr:`connected` and the per-client registry in
        step before and after the hook runs, so a ``@connected`` hook sees its
        client and a ``@disconnected`` hook can still address it. A session end
        clears occupancy outright — the session's connections are torn down
        wholesale without a per-connection close — so :attr:`connected` reads
        false for a ``run`` loop gating on it and the client registry does not
        leak across sessions. Upload events run their hooks directly.

        The session's :attr:`state` is built from field defaults before the
        ``@session_started`` hook runs, so once-per-session initialization can
        write to it, and cleared only after the ``@session_ended`` hook returns,
        which may still read the ending session's values. A client leaving and
        rejoining within one session sees the same instance.

        The live gate, :attr:`_live`, is a session that has started and a client
        that is connected. A session end drops the gate before its hook runs, so
        a ``run()`` loop that checks the gate between units of work stops at the
        next boundary instead of waiting for the hook to take the step lock.
        """
        hooks = self.__reactor_contract__.lifecycle
        if isinstance(event, ClientConnected):
            self._clients[event.conn_id] = self._make_client(event.conn_id, time.monotonic())
            self._set_connected(event.total)
            await self._invoke_hook(hooks.connected, event.conn_id)
        elif isinstance(event, ClientDisconnected):
            self._set_connected(event.total)
            await self._invoke_hook(hooks.disconnected, event.conn_id)
            self._clients.pop(event.conn_id, None)
        elif isinstance(event, SessionStarted):
            if self.__app_state__ is not None:
                self.state = self.__app_state__()
            self._session_active = True
            self._update_live()
            await self._invoke_hook(hooks.session_started, None)
        elif isinstance(event, SessionEnded):
            self._session_active = False
            self._set_connected(0)
            await self._invoke_hook(hooks.session_ended, None)
            self._clients.clear()
            if self.__app_state__ is not None:
                self.state = None
            # The hook has returned, so its records were written while the
            # session's log binding was live; the session's last ambient writer
            # is done and the binding retires here, on the model thread.
            release_session_id(event._log_binding)
        elif isinstance(event, FileUploaded):
            await self._invoke_hook(hooks.file_uploaded, event.conn_id, uploaded_file=event.file)

    async def _invoke_hook(
        self, hook: Callable[..., Any] | None, conn_id: ConnId | None, **extra: Any
    ) -> None:
        """Call one lifecycle hook under the step lock, injecting reserved parameters.

        A hook that raises is logged and swallowed, so a faulty hook cannot tear
        the reactor loop down. The lock puts the hook between two units of a
        ``run()`` loop, like a command handler.
        """
        if hook is None:
            return
        kwargs = dict(extra)
        for name in _hook_reserved(hook):
            kwargs[name] = self._reserved(name, conn_id)
        async with self._step_lock:
            try:
                if inspect.iscoroutinefunction(hook):
                    await hook(self, **kwargs)
                else:
                    hook(self, **kwargs)
            except Exception:
                logger.exception("error in lifecycle handler", handler=_qualname(hook))

    # -- internals ------------------------------------------------------------

    def _set_connected(self, total: int) -> None:
        """Hold :attr:`connected` set while any client is connected."""
        if total > 0:
            self.connected.set()
        else:
            self.connected.clear()
        self._update_live()

    def _update_live(self) -> None:
        """Reconcile the live gate from session liveness and the client count.

        Every drop of the gate is counted, so a loop that was blocked while the
        gate dropped and came back still sees that a boundary passed.
        """
        if self.connected.is_set() and self._session_active:
            self._live.set()
        elif self._live.is_set():
            self._live.clear()
            self._gate_drops += 1

    def _reserved(self, name: str, conn_id: ConnId | None) -> Any:
        """Resolve a reserved handler parameter for the addressed connection."""
        if name == "client":
            return self._client_for(conn_id)
        return None

    def _client_for(self, conn_id: ConnId | None) -> ClientInfo | None:
        """Return the handle for *conn_id*, building one if the registry lacks it.

        A command can race ahead of its ``ClientConnected`` event since the two
        ride separate queues, so a missing entry is filled rather than dropped.
        """
        if conn_id is None:
            return None
        client = self._clients.get(conn_id)
        if client is None:
            client = self._make_client(conn_id, time.monotonic())
            self._clients[conn_id] = client
        return client

    def _make_client(self, conn_id: ConnId, joined_at: float) -> ClientInfo:
        """Build a client handle bound to the addressed sink for *conn_id*."""
        return ClientInfo(
            id=conn_id,
            joined_at=joined_at,
            _send=lambda message: self._reply(conn_id, message, None),
        )

    def _fail(self, envelope: CommandEnvelope, failure: CommandFailure) -> None:
        """Answer a command with a failure, when there is a client to answer.

        A bodyless acknowledgement without a request id says nothing, so the
        success path drops it. A failure carries a code and a message the client
        can act on, so it goes out whether or not the command was correlated.
        """
        if envelope.conn_id is None:
            return
        self._reply(envelope.conn_id, failure, envelope.request_id)

    def _reply(
        self,
        conn_id: ConnId,
        message: ModelMessage | CommandFailure | None,
        request_id: RequestId | None,
    ) -> None:
        """Send a reply to one connection through the addressed sink, if bound.

        A :class:`CommandFailure` is the reason a handler could not answer. A
        ``None`` message is the bodyless acknowledgement of a command whose
        handler returned nothing, correlated by *request_id*.
        """
        if self._out_addressed is not None:
            self._out_addressed(conn_id, message, request_id)


def _hook_reserved(hook: Callable[..., Any]) -> tuple[str, ...]:
    """Return the reserved parameters a lifecycle hook declares, in registry order."""
    try:
        sig = inspect.signature(hook)
    except (TypeError, ValueError):
        return ()
    return tuple(name for name in RESERVED_PARAMS if name in sig.parameters)


def _qualname(hook: Callable[..., Any]) -> str:
    """Best-effort readable name for a handler, for logging."""
    return getattr(hook, "__qualname__", repr(hook))


def _fps_is_author_pinned(cls: type) -> bool:
    """Return whether the application (or an intermediate base) pins ``fps`` itself.

    A loop that measures its own compute time paces playout from it unless the
    author declares ``fps``. The walk covers the author's own classes but stops
    at :class:`ReactorCore`, whose ``fps`` is the framework default rather than
    an author's choice, so a subclass that inherits a pinned rate from an
    intermediate base counts as pinned even without redeclaring it.
    """
    for klass in cls.__mro__:
        if klass is ReactorCore:
            break
        if "fps" in vars(klass):
            return True
    return False


# -- typed state: the `state:` annotation and its generated setters -----------


def _resolve_state_class(cls: type) -> type[InputState] | None:
    """Return the :class:`InputState` subclass named by the ``state`` annotation.

    A ``state`` annotation the class carries but that cannot be resolved, such
    as a forward reference to a class defined later in the module, is reported
    with a warning: the app would otherwise serve a schema with no ``set_``
    commands and no signal as to why.
    """
    raw = cls.__dict__.get("__annotations__", {}).get("state")
    if isinstance(raw, type):
        hint: Any = raw
    else:
        try:
            hint = get_type_hints(cls).get("state")
        except Exception as exc:
            if raw is not None:
                logger.warning(
                    "state annotation could not be resolved; no set_ commands generated",
                    app=cls.__qualname__,
                    annotation=str(raw),
                    error=str(exc),
                )
            return None
    if isinstance(hint, type) and issubclass(hint, InputState):
        return hint
    return None


def _existing_command_names(cls: type) -> set[str]:
    """Collect the command names already claimed by ``@event`` handlers on *cls*."""
    names: set[str] = set()
    for klass in cls.__mro__:
        for attr in vars(klass).values():
            handler = getattr(attr, EVENT_ATTR, None)
            if isinstance(handler, EventHandler):
                names.add(handler.name)
    return names


def _stamp_auto_setters(cls: type, state_cls: type[InputState]) -> None:
    """Stamp a ``set_<field>`` command handler for each public state field.

    Each handler carries the same :class:`EventHandler` metadata an ``@event``
    decorator produces, so the contract treats it identically. A field whose
    ``set_`` name is already claimed by a hand-written handler is skipped, so an
    author can override the generated setter.
    """
    existing = _existing_command_names(cls)
    try:
        hints = get_type_hints(state_cls)
    except Exception:
        hints = dict(getattr(state_cls, "__annotations__", {}))

    for field_name, info in state_cls._public_fields.items():
        command_name = f"set_{field_name}"
        if command_name in existing:
            continue
        field_type = hints.get(field_name, Any)
        command = make_command(command_name, [(field_name, field_type, info)])
        handler = _make_setter(field_name)
        setattr(
            handler,
            EVENT_ATTR,
            EventHandler(
                name=command_name,
                description=info.description or f"Set {field_name}.",
                command=command,
                is_async=False,
                reserved=(),
            ),
        )
        setattr(cls, command_name, handler)


def _make_setter(field_name: str) -> Callable[..., None]:
    """Build the handler that writes one field onto the live state."""

    def handler(self: Any, **kwargs: Any) -> None:
        if self.state is None:
            return
        setattr(self.state, field_name, kwargs[field_name])

    return handler
