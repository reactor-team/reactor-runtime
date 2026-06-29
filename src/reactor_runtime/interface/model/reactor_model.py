"""The model authoring base — :class:`ReactorModel`.

What a model author subclasses. It joins the two halves of the model layer: the
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
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Coroutine
from typing import Any, ClassVar

from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    FileUploaded,
    ReactorEvent,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import ConnId
from reactor_runtime.interface.client import ClientInfo
from reactor_runtime.interface.events.decorators import RESERVED_PARAMS
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.reactor_core import (
    CommandEnvelope,
    ReactorCore,
    RequestId,
)
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.log import get_logger

logger = get_logger(__name__)


class ReactorModel(ReactorCore):
    """Base class an author subclasses to define a model.

    Decorate methods with ``@event`` to expose commands, and with the lifecycle
    decorators to hook session and connection events. Declaring the subclass
    resolves the contract and caches it on the class, reachable through
    :meth:`ModelContract.of`.

    Class attributes:
        fps: Target output frame rate of the emission buffer (default 30).
        buffer_size: Output buffer queue depth (default 10) — how many emitted
            frames may sit queued before :meth:`emit` blocks to pace the
            producer. A smaller value keeps latency low; a larger one absorbs
            more jitter.

    Lifecycle:
        connected: An :class:`asyncio.Event` set while at least one client is
            connected and cleared when the last one leaves, so a ``run`` loop can
            gate generation on having an audience.
    """

    __reactor_contract__: ClassVar[ModelContract]

    connected: asyncio.Event
    _clients: dict[ConnId, ClientInfo]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.__reactor_contract__ = ModelContract.build(cls)

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
        the command's request id. A handler that raises is logged and swallowed,
        so one bad command cannot stop the loop.
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
        except Exception:
            logger.exception("error in command handler", command=spec.name)
            return
        if isinstance(result, ModelMessage) and envelope.conn_id is not None:
            self._reply(envelope.conn_id, result, envelope.request_id)

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
            await self._invoke_hook(hooks.session_started, None)
        elif isinstance(event, SessionEnded):
            self._set_connected(0)
            await self._invoke_hook(hooks.session_ended, None)
            self._clients.clear()
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

    def _reply(self, conn_id: ConnId, message: ModelMessage, request_id: RequestId | None) -> None:
        """Send a message to one connection through the addressed sink, if bound."""
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
