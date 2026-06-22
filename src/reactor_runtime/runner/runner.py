"""The runner — one model and its session, composed.

The :class:`Runner` orchestrates a single model and the session around it. It
composes the session machinery rather than inheriting it, *is* the
:class:`~reactor_runtime.core.transport.ConnectionSink` transports push facts
through, and binds the model's outbound path back down onto its connections.
Process lifecycle lives elsewhere; the runner only owns session and model
orchestration, exposing the four :class:`~reactor_runtime.core.service.ServiceComponent`
verbs so the service can drive it.

The deliberate work is in :meth:`start`: the model class is resolved, its
contract read from the class, the model instantiated and loaded, the bridge
built and its outbound bound, and only then is the model spawned and the session
declared ready.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Mapping

from reactor_runtime.core import (
    Connection,
    ConnectionAnswered,
    ConnectionSink,
    ConnId,
    Health,
    HealthStatus,
    InputFrame,
    RuntimeConfig,
    ServiceComponent,
    SessionEvent,
    SessionState,
    Transition,
    TransitionEvent,
)
from reactor_runtime.event_stream import EventStream
from reactor_runtime.log import get_logger
from reactor_runtime.message_gateway import InboundCommand, MessageGateway
from reactor_runtime.model.bridge import ModelBridge
from reactor_runtime.model.contract import ModelContract
from reactor_runtime.model.message import ModelMessage
from reactor_runtime.model.reactor_core import ReactorCore
from reactor_runtime.protocol import Channel, ProtocolVersion, select
from reactor_runtime.runner.connection_manager import ConnectionManager
from reactor_runtime.runner.state_machine import SessionStateMachine

logger = get_logger(__name__)


def _no_shutdown() -> None:
    """Default process-shutdown hook — a no-op until the service wires one in."""


def import_model_class(model_ref: str) -> type[ReactorCore]:
    """Resolve a ``"module:Class"`` reference into the model class it names.

    Args:
        model_ref: An import reference of the form ``"package.module:Class"``.

    Returns:
        The referenced model class.

    Raises:
        ValueError: If the reference is not of the form ``"module:Class"``.
        TypeError: If the reference does not name a :class:`ReactorCore` subclass.
    """
    module_name, separator, class_name = model_ref.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f"model_ref must be 'module:Class', got {model_ref!r}")
    model_cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(model_cls, type) or not issubclass(model_cls, ReactorCore):
        raise TypeError(f"{model_ref} does not name a ReactorCore subclass")
    return model_cls


class Runner(ServiceComponent, ConnectionSink):
    """Orchestrates one model and its session, transport- and platform-neutral.

    Composes the session state machine, the connection multiplexer, the inbound
    message gateway, and the egress journal, and builds the model bridge in
    :meth:`start`. It is the connection sink every transport reports into, and
    funnels each session transition through one listener.
    """

    name = "runner"
    depends_on: tuple[str, ...] = ()

    def __init__(self, cfg: RuntimeConfig) -> None:
        """Wire the session machinery and local components that need no model.

        The bridge is built later, in :meth:`start`, once the model class is
        known; everything here is model-independent.

        Args:
            cfg: The configuration for this runtime process.
        """
        self._cfg = cfg
        self._sm = SessionStateMachine()
        self._sm.on_transition(self._dispatch_transition)
        self._events = EventStream()
        self._connections = ConnectionManager(state_machine=self._sm)
        self._codec = select(ProtocolVersion.V0)
        self._gateway = MessageGateway(
            sink=self, codec=self._codec, on_command=self._submit_command
        )
        self._bridge: ModelBridge | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inbound: set[asyncio.Task[None]] = set()
        self._used_conn_ids: set[ConnId] = set()
        self._accepting = True
        # The process-shutdown hook, wired by the assembly so the runner can ask
        # the service to bring the process down when the session is terminated
        # (a failed model load). A no-op until then, so the runner stays usable
        # on its own.
        self.request_shutdown: Callable[[], None] = _no_shutdown

    # -- lifecycle (ServiceComponent) -----------------------------------------

    async def start(self) -> None:
        """Resolve and bring up the model, then declare the session ready.

        The ordering is deliberate: resolve the class, read its contract from
        the class, instantiate and load it, build the bridge and bind the
        outbound path before the model thread spins, spawn it, and only then
        send ``INITIALIZATION_SUCCESS`` so the session leaves ``CREATED``.

        Bringing the model up is the one step that can fail outright. A failure
        here is terminal for the process — there is no model to serve — so it is
        caught and turned into ``INITIALIZATION_FAIL`` (moving the session to
        ``TERMINATED``) rather than raised; the dispatch on that move asks the
        service to bring the process down. ``start`` itself always returns.
        """
        self._loop = asyncio.get_running_loop()
        logger.info("loading model", model=self._cfg.model_ref)
        try:
            model_cls = import_model_class(self._cfg.model_ref)
            contract = ModelContract.of(model_cls)
            model = model_cls()
            model.load(dict(self._cfg.model_config))
            bridge = ModelBridge(model, contract)
            bridge.bind_outbound(
                broadcast=self._broadcast_message,
                addressed=self._send_addressed,
                media=self._connections.broadcast_media,
            )
            bridge.start()
        except Exception:
            logger.exception("model failed to load; terminating the session")
            self._sm.send(SessionEvent.INITIALIZATION_FAIL)
            return
        self._bridge = bridge
        self._sm.send(SessionEvent.INITIALIZATION_SUCCESS)
        logger.info(
            "model loaded; session ready",
            model=type(model).__name__,
            tracks=len(contract.tracks),
            commands=len(contract.commands),
        )

    async def drain(self) -> None:
        """Stop accepting new work."""
        self._accepting = False

    async def stop(self) -> None:
        """Release the model, bringing its thread down last."""
        if self._bridge is not None:
            await self._bridge.stop()

    def health(self) -> Health:
        """Report readiness from the session state and whether the model is up."""
        if self._sm.current_state is SessionState.TERMINATED:
            return Health(HealthStatus.UNHEALTHY, "session terminated")
        if self._bridge is None:
            return Health(HealthStatus.DEGRADED, "model not started")
        return Health.healthy()

    # -- inbound (ConnectionSink) ---------------------------------------------

    def connection_opened(self, conn: Connection) -> None:
        """Register a connection whose wire has reached its connected state."""
        self._connections.register(conn)

    def connection_closed(self, conn_id: ConnId) -> None:
        """Drop a previously opened connection that has gone away."""
        self._connections.drop(conn_id)

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        """Journal a transport's negotiation answer for a connection.

        The answer is an opaque, transport-agnostic payload (for WebRTC, the SDP
        answer as ``{"type", "sdp"}``); the runner does not parse it, only
        records it on the egress journal so a consumer can hand it back to the
        offering client. It arrives before the connection's wire connects, so it
        is journalled directly rather than riding a session transition.
        """
        self._events.emit(ConnectionAnswered(conn_id, dict(answer)))

    def message_received(self, conn_id: ConnId, payload: bytes | str) -> None:
        """Hand an inbound frame to the gateway for decoding and dispatch.

        The gateway decodes asynchronously, so the work is scheduled on the
        runtime loop and tracked until it completes. Model traffic rides the
        data channel, so the frame is decoded as such.
        """
        if self._loop is None:
            return
        task = self._loop.create_task(self._gateway.handle(conn_id, payload, Channel.DATA))
        self._inbound.add(task)
        task.add_done_callback(self._inbound.discard)

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        """Route an inbound media frame to its track on the model."""
        if self._bridge is not None:
            self._bridge.push_media(track, frame)

    def keepalive(self, conn_id: ConnId) -> None:
        """Note liveness for a connection."""
        self._connections.note_keepalive(conn_id)

    # -- internals ------------------------------------------------------------

    async def _submit_command(self, command: InboundCommand) -> None:
        """Submit a decoded client command to the model through the bridge."""
        if self._bridge is None:
            return
        await self._bridge.submit_command(
            command.name,
            dict(command.args),
            conn_id=command.conn_id,
            request_id=command.request_id,
        )

    def _broadcast_message(self, message: ModelMessage) -> None:
        """Encode a model message and broadcast it to every connection."""
        _channel, frame = self._codec.encode_model_message(
            message.name, message.to_wire_format()["data"]
        )
        self._connections.broadcast(frame)

    def _send_addressed(
        self, conn_id: ConnId, message: ModelMessage, request_id: str | None
    ) -> None:
        """Encode a model message and send it to one connection, correlating a reply."""
        _channel, frame = self._codec.encode_model_message(
            message.name, message.to_wire_format()["data"], request_id=request_id
        )
        self._connections.send(conn_id, frame)

    def _dispatch_transition(self, transition: Transition) -> None:
        """Journal each session transition on the egress stream."""
        self._events.emit(TransitionEvent(transition))
