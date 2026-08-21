"""The model authoring base — :class:`ReactorModel`.

What a model author subclasses. It joins the two halves of the model layer: the
:class:`ReactorCore` engine it inherits (thread, loop, buffers, queues) and the
:class:`ModelContract` it assembles. Declaring a subclass resolves the contract
once, from a single traversal of the class, and caches it on the class — the
commands its ``@event`` handlers expose, the messages they return, its tracks,
and its lifecycle hooks.

Two things live here and nowhere else. The *authoring surface*: the annotations
a model declares, the handful of methods it may override, and nothing besides.
And the *what* the engine leaves open: the two dispatch loops that drain the
engine's typed queues into handlers. The command loop validates nothing — that
happened at the bridge — and turns each :class:`CommandEnvelope` back into a
handler call, replying with the handler's returned message to the one connection
that sent the command. The reactor loop runs the lifecycle hooks, maintains
:attr:`connected` from the live client count, and builds the session's state.

The default generation loop is deliberately not here: it lives in
:class:`~reactor_runtime.interface.internal.step_driver.StepDriver`, which
``run()`` creates and hands the loop to. A model that overrides ``run()`` has no
driver at all, so it inherits no half-loop it has to reason about.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, ClassVar

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
from reactor_runtime.interface.client import ClientInfo
from reactor_runtime.interface.events.decorators import RESERVED_PARAMS
from reactor_runtime.interface.events.errors import CommandError
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.reactor_core import (
    CommandEnvelope,
    ReactorCore,
    RequestId,
)
from reactor_runtime.interface.internal.step_driver import StepDriver
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.interface.model.input_state import InputState
from reactor_runtime.interface.model.state_binding import (
    STATE_TYPE_ATTR,
    resolve_state_class,
    stamp_auto_setters,
)
from reactor_runtime.interface.model.stepping import StepStats
from reactor_runtime.interface.tracks.output import OUTPUT_REGISTRY, Output
from reactor_runtime.log import get_logger

logger = get_logger(__name__)


class ReactorModel(ReactorCore):
    """Base class an author subclasses to define a model.

    Declare what the model exchanges, then fill in as much of the step as it
    needs::

        class Helios(ReactorModel):
            state: HeliosState        # what a client may set, as set_<field> commands
            input: HeliosInput        # inbound tracks, as live frame buffers
            model: HeliosGenerator    # the model the runtime drives

    Decorate methods with ``@event`` to expose commands, and with the lifecycle
    decorators to hook session and connection events — ``@session_started`` is
    the hook for once-per-session initialization. Declaring the subclass
    resolves the contract and caches it on the class, reachable through
    :meth:`ModelContract.of`.

    Two ways to produce. Declare ``model:`` and the runtime drives it, calling
    :meth:`map_step` for its arguments and :meth:`to_output` with what it
    produced. Or override :meth:`run` and own the loop, keeping the state, the
    commands, the tracks, and the model binding exactly as they are.

    Class attributes:
        fps: The nominal rate, in frames per second, an emitted chunk plays out
            at when neither the output nor a measurement says otherwise
            (default 30). Declaring it pins playout to a fixed rate.

    Attributes:
        state: The session's :class:`InputState` instance, or ``None`` between
            sessions. Built before ``@session_started`` runs and cleared after
            ``@session_ended``, so it survives a client leaving and rejoining.
        model: The model the runtime drives, when the class declares one.

    Lifecycle:
        connected: An :class:`asyncio.Event` set while at least one client is
            connected and cleared when the last one leaves or the session ends,
            so a ``run`` loop can gate generation on having an audience.
    """

    __reactor_contract__: ClassVar[ModelContract]

    state: Any
    """The session's state instance, or ``None`` between sessions."""

    model: Any
    """The model the runtime drives, when the class declares one."""

    connected: asyncio.Event
    _clients: dict[ConnId, ClientInfo]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # The generated set_<field> commands are stamped before the contract is
        # assembled, so they are resolved as ordinary commands like any @event.
        state_cls = resolve_state_class(cls)
        if state_cls is not None:
            setattr(cls, STATE_TYPE_ATTR, state_cls)
            stamp_auto_setters(cls, state_cls)
        cls.__reactor_contract__ = ModelContract.build(cls)

    def __init__(self) -> None:
        super().__init__()
        self.state = None
        declared = getattr(type(self), "model", None)
        if isinstance(declared, type):
            # `model = TheModel` names a class for the runtime to build; an
            # instance assigned in load() is the other, fuller form.
            self.model = declared()

    # -- author surface -------------------------------------------------------

    def load(self, config_path: Path | None) -> None:
        """Load weights and allocate resources, once, before any client connects.

        The default hands *config_path* to the model bound as a class-level
        default (``model = TheModel``), which is all a model that takes no
        constructor arguments needs. Override it to build the model yourself::

            def load(self, config_path):
                self.model = HeliosGenerator()
                self.model.load(config_path)

        Args:
            config_path: Path to the model's config file (from ``runtime.config``
                in ``reactor.yaml``), or ``None`` when none is configured. The
                runtime does not parse it; the model reads and interprets it
                however it wants.
        """
        loader = getattr(getattr(self, "model", None), "load", None)
        if loader is not None:
            loader(config_path)

    def map_step(self, state: Any, input: Any) -> dict[str, Any]:
        """Build the arguments for one step. No side effects.

        Read the values a client set off *state*, drain the frames that arrived
        off *input*, and return the keyword arguments the model's ``generate``
        wants. Raise :class:`~reactor_runtime.NotReady` when there is nothing to
        step on yet — the driver holds the stream and tries again.

        Both arguments are passed rather than read off ``self``, so a step is
        visibly a function of exactly those two things and can be tested without
        a session. Do not call the model, ``emit``, ``send``, or ``flush`` from
        here; those belong to ``@event`` handlers and ``to_output``.

        The default passes every public state field, by name, which is all a
        model whose arguments *are* its state fields needs::

            def map_step(self, state, input):
                frames = input.webcam.try_read(4)
                if frames is None:
                    raise NotReady("waiting for webcam frames")
                return {"driving": np.stack([f.data for f in frames]), "prompt": state.prompt}

        Args:
            state: The session's state instance, or ``None`` if none is declared.
            input: The inbound track holder, or ``None`` if none is declared.

        Returns:
            The keyword arguments to call the model's ``generate`` with.

        Raises:
            NotReady: There is nothing to step on yet.
        """
        if state is None:
            return {}
        return {name: getattr(state, name) for name in type(state)._public_fields}

    def to_output(self, *produced: Any, **products: Any) -> Any:
        """Put what the model produced on the wire.

        Return an :class:`Output` to emit media, a
        :class:`~reactor_runtime.ModelMessage` to send one, a sequence of either
        when a step does both, or ``None`` to publish nothing. Messages are sent
        before the media, so an action never waits behind video.

        What the model returned arrives here by shape: a mapping spreads into
        keyword arguments, and anything else arrives as one positional argument.
        Declare a ``stats`` parameter to receive the runtime's
        :class:`~reactor_runtime.StepStats` for the step alongside it::

            def to_output(self, *, frames, drift, stats: StepStats) -> MyOutput:
                return MyOutput(
                    main_video=TrackPayload(frames, metadata=[{"drift": d} for d in drift]),
                    fps=CAMERA_FPS,
                )

        The default places a single produced value on the model's one declared
        output track, and a mapping onto the tracks its keys name. A model with
        several output classes, or with none because it only sends messages,
        implements this instead. Overrides are free to declare the exact
        parameters the model produces; the signature here accepts anything so
        that a narrower one is still a valid override.

        Args:
            produced: What ``generate`` returned, when it was not a mapping.
            products: The entries of the mapping ``generate`` returned.

        Returns:
            What to publish for this step.

        Raises:
            TypeError: If the default cannot tell which track a value belongs on.
        """
        output_cls = _sole_output_class(type(self))
        if products:
            return output_cls(**products)
        tracks = output_cls.__tracks__
        if len(tracks) != 1:
            raise TypeError(
                f"{output_cls.__name__} declares {len(tracks)} tracks, so "
                f"{type(self).__name__} must implement to_output() to say which product "
                f"goes on which track, or have generate() return a mapping keyed by track."
            )
        if not produced:
            raise TypeError(
                f"{type(self).__name__} produced nothing to place on "
                f"'{next(iter(tracks))}'. Return the frames from generate(), or None to "
                f"skip the step."
            )
        return output_cls(**{next(iter(tracks)): produced[0]})

    async def on_state_changed(self, state: Any) -> None:
        """React to the state having changed, after the change landed.

        Called around every step — once with everything the commands since the
        last step have written, and again if the step itself wrote to the state.
        The place to mirror the state to clients, which is one line::

            async def on_state_changed(self, state) -> None:
                await self.send(Controls.from_state(state))

        Left as a no-op by default. This is where a client is *told* about a
        change; doing work in reaction to one belongs to the model, which sees
        the values on the next step anyway.

        Args:
            state: The state as it now stands.
        """

    def on_step(self, stats: StepStats) -> None:
        """React to a step that produced, after its products are on the wire.

        The place for per-step application effects the wire contract does not
        carry — a progress message, a metric, a counter. Left as a no-op by
        default.

        Args:
            stats: What the runtime measured about the step.
        """

    async def run(self) -> None:
        """Drive the declared ``model:`` while a client is connected.

        Override this to own the generation loop, and everything else keeps
        working: the state and its commands, the ``@event`` handlers, the
        lifecycle hooks, the tracks, and the model binding. An override gates on
        :attr:`connected` and emits on its own schedule::

            async def run(self) -> None:
                while True:
                    await self.connected.wait()
                    while self.connected.is_set():
                        ...

        Raises:
            TypeError: If the class declares no ``model:`` to drive and does not
                override this method.
        """
        await StepDriver(self).run()

    # -- engine hooks ---------------------------------------------------------

    def _on_loop_ready(self) -> None:
        """Create the loop-bound state the dispatchers need."""
        self.connected = asyncio.Event()
        self._clients = {}

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
        """
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

        The session's state is built before the ``@session_started`` hook, so
        once-per-session initialization can write to it, and cleared after
        ``@session_ended``, which may still read the ending session's values.
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
            state_type: type[InputState] | None = getattr(type(self), STATE_TYPE_ATTR, None)
            if state_type is not None:
                self.state = state_type()
            await self._invoke_hook(hooks.session_started, None)
        elif isinstance(event, SessionEnded):
            self._set_connected(0)
            await self._invoke_hook(hooks.session_ended, None)
            self._clients.clear()
            self.state = None
        elif isinstance(event, FileUploaded):
            await self._invoke_hook(hooks.file_uploaded, event.conn_id, uploaded_file=event.file)

    async def _invoke_hook(
        self, hook: Callable[..., Any] | None, conn_id: ConnId | None, **extra: Any
    ) -> None:
        """Call one lifecycle hook, injecting reserved parameters it declares.

        A hook that raises is logged and swallowed, so a faulty hook cannot tear
        the reactor loop down.
        """
        if hook is None:
            return
        kwargs = dict(extra)
        for name in _hook_reserved(hook):
            kwargs[name] = self._reserved(name, conn_id)
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


def _sole_output_class(model_cls: type) -> type[Output]:
    """Return the one declared :class:`Output` class the default mapping can use.

    Args:
        model_cls: The model class, named in the error when there is no single
            unambiguous choice.

    Returns:
        The only registered output class.

    Raises:
        TypeError: If the model declares no output class, or more than one, so
            the default cannot tell which to build.
    """
    declared = list(OUTPUT_REGISTRY.values())
    if len(declared) == 1:
        return declared[0]
    if not declared:
        raise TypeError(
            f"{model_cls.__name__} declares no Output class, so there is no track to "
            f"emit on. Declare one, or implement to_output() to return the messages "
            f"this model publishes."
        )
    names = ", ".join(sorted(cls.__name__ for cls in declared))
    raise TypeError(
        f"{model_cls.__name__} declares several Output classes ({names}), so "
        f"to_output() has to say which one a step builds."
    )


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
