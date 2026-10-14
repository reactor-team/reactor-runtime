"""The session host — several sessions, one model, one process.

A :class:`SessionHost` serves a :class:`~reactor_runtime.BatchedReactorApp`
that declares ``max_sessions`` above one. It loads the model once and keeps a
table of session slots, each a :class:`Runner` of its own: a state machine, a
connection manager, a recorder, an upload store, step counters, an orphan
timer, and runtime stats. A slot drives the shared model through a bridge face
that addresses one session, so a command, a lifecycle fact, and a media frame
reach that session's state and buffers. What the model emits, sends, or
reports comes back tagged with the session it was made for, and the host hands
it to the slot that owns that session. A slot whose session ends returns to
ready and takes the next start.

The host answers the same session-control surface a runner does, keyed by the
session id the caller chose at the start, or the id the host minted for it.
Every slot journals onto one egress stream, each fact tagged with its session.

Isolation follows the slots. A session that is stopped, times out, or fails in
its own hooks ends alone. A model loop that crashes ends every session and the
process, as it does under a runner.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from reactor_runtime.core import (
    ClientStatsBatch,
    CommandFailure,
    CompletedStep,
    Connection,
    ConnId,
    EndReason,
    Health,
    HealthStatus,
    InputFrame,
    MediaChunk,
    RuntimeConfig,
    RuntimeState,
    SessionEvent,
    SessionState,
)
from reactor_runtime.core.model import ReactorEvent, SessionStarted
from reactor_runtime.event_stream import EventStream
from reactor_runtime.interface.app.batched_app import BatchedReactorApp, current_session_key
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.bridge import CommandOutcome, ModelBridge
from reactor_runtime.interface.internal.reactor_core import MediaOps, RequestId
from reactor_runtime.interface.model.contract import ContractError, ModelContract
from reactor_runtime.log import get_logger, session_scope
from reactor_runtime.manifest import import_model_class
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.recording import ClipManifest, Gone, Pending, Recorder
from reactor_runtime.runner.runner import (
    _RUNNING_STATES,
    Runner,
    ShutdownHook,
    _no_shutdown,
)
from reactor_runtime.step_results import StepStore
from reactor_runtime.transport.router import (
    SessionNotRunningError,
    SessionTransitionError,
    UnknownSessionError,
)
from reactor_runtime.upload_store import UnknownUploadError, UploadAlreadyCompleteError, UploadStore

logger = get_logger(__name__)

# The most times a connection id is redrawn because another slot holds it.
_MAX_CONN_ID_DRAWS = 100


class SessionIdRequiredError(RuntimeError):
    """Raised when a request that must name a session does not.

    A host serving several sessions cannot tell which one a bare stop or
    descriptor read means.
    """


class NoFreeSlotError(SessionTransitionError):
    """Raised when a session start finds every slot serving.

    A :class:`SessionTransitionError`, so it answers with the status a busy
    single-session runtime answers a start with.

    Attributes:
        slots: How many sessions the process hosts at once.
    """

    def __init__(self, slots: int) -> None:
        """Record the capacity the start was refused against."""
        super().__init__("start", SessionState.STREAMING)
        self.slots = slots
        self.args = (f"cannot start session: all {slots} session slots are serving",)


def batched_capacity(model_ref: str) -> int:
    """Return how many sessions the model *model_ref* names serves at once.

    One for any model that is not a :class:`BatchedReactorApp`, and for one
    whose class cannot be imported: the single-session runner reports that
    failure the way it always does.
    """
    try:
        model_cls = import_model_class(model_ref)
    except Exception:
        return 1
    if isinstance(model_cls, type) and issubclass(model_cls, BatchedReactorApp):
        return model_cls.max_sessions
    return 1


@dataclasses.dataclass
class _Route:
    """Where a model-side session key's output goes: its slot and that slot's session count."""

    slot: _SessionSlot
    generation: int


class _SessionBridge(ModelBridge):
    """One slot's face onto the shared model, addressing the slot's current session."""

    def __init__(self, host: SessionHost, slot: _SessionSlot) -> None:
        self._host = host
        self._slot = slot
        self._key: int | None = None
        self._outbound_bound = False

    @property
    def contract(self) -> ModelContract:
        """The shared model's contract."""
        return self._host.contract

    async def submit_command(
        self,
        name: str,
        raw_args: dict[str, Any],
        *,
        conn_id: ConnId | None,
        request_id: RequestId,
    ) -> CommandOutcome:
        """Validate a client command and admit it to this slot's session."""
        try:
            command = self.contract.validate(name, raw_args)
        except ContractError as error:
            return CommandOutcome.reject(error.field, error.reason)
        if self._key is not None:
            self._host.model.submit_session_command(self._key, command, conn_id, request_id)
        return CommandOutcome.accept()

    def check_command(self, name: str, raw_args: dict[str, Any]) -> CommandOutcome:
        """Validate a command against the shared contract without admitting it."""
        try:
            self.contract.validate(name, raw_args)
        except ContractError as error:
            return CommandOutcome.reject(error.field, error.reason)
        return CommandOutcome.accept()

    def dispatch_reactor_event(self, event: ReactorEvent) -> None:
        """Hand the model a fact about this slot's session; a start opens a new session key."""
        if isinstance(event, SessionStarted):
            self._key = self._host.open_key(self._slot)
        if self._key is not None:
            self._host.model.post_session_event(self._key, event)

    def push_media(self, track: str, frame: InputFrame) -> None:
        """Route an inbound frame to this slot's session."""
        if self._key is not None:
            self._host.model.push_session_media(self._key, track, frame)

    def bind_outbound(self, **sinks: Any) -> None:
        """Accept the slot's sinks; the host routes the model's output to the slot itself."""
        self._outbound_bound = True

    def start(self) -> None:
        """Leave the shared model to the host, which started it."""

    async def stop(self) -> None:
        """Leave the shared model to the host, which stops it after every slot."""


class _SessionSlot(Runner):
    """A :class:`Runner` that hosts one session at a time on a shared model."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        metrics: RuntimeMetrics,
        *,
        host: SessionHost,
        events: EventStream,
        step_store: StepStore | None,
        load_started: float | None,
    ) -> None:
        super().__init__(cfg, metrics, load_started=load_started)
        self._host = host
        self._events = events
        self._step_store = step_store

    async def _open_model(self) -> tuple[ModelBridge, str]:
        """Attach to the model the host loads, instead of loading one."""
        model_name = await self._host.loaded()
        bridge = _SessionBridge(self._host, self)
        bridge.bind_outbound(**self._outbound_sinks())
        return bridge, model_name

    def _bind_log_session(self, session_id: str) -> int:
        """Leave the process-wide log binding alone; the host scopes each session's work."""
        return 0

    def start_session(self, params: Mapping[str, Any]) -> None:
        """Open a session under the id in *params*, which is also its transport id."""
        previous = self._session_id
        self._session_id = str(params["session_id"])
        try:
            super().start_session(params)
        except Exception:
            self._session_id = previous
            raise

    async def stop(self) -> None:
        """Release the slot; the shared step store is the host's to close."""
        self._step_store = None
        await super().stop()

    @property
    def sm_state(self) -> SessionState:
        """The slot's session state."""
        return self._sm.current_state

    @property
    def hosted_id(self) -> str:
        """The id of the session the slot runs, or ran last."""
        return self._session_id

    @property
    def generation(self) -> int:
        """How many sessions the slot has started."""
        return self._sessions_posted

    @property
    def upload_store(self) -> UploadStore:
        """The slot's own upload store."""
        return self._uploads

    @property
    def slot_recorder(self) -> Recorder:
        """The slot's own recorder."""
        return self._recorder

    def deliver_media(self, chunk: MediaChunk) -> None:
        """Fan a chunk the model emitted for this slot's session out to its consumers."""
        self._emit_media(chunk)

    def deliver_message(self, message: ModelMessage) -> None:
        """Broadcast a model message to this slot's clients."""
        self._broadcast_message(message)

    def deliver_reply(
        self, conn_id: ConnId, message: ModelMessage | CommandFailure | None, request_id: str | None
    ) -> None:
        """Send a model reply to one of this slot's connections."""
        self._send_addressed(conn_id, message, request_id)

    def deliver_step(self, step: CompletedStep) -> None:
        """Record a step the model reported for this slot's session."""
        self._on_step_completed(step)

    def flush(self) -> None:
        """Cut this slot's playout."""
        self._flush_media()

    def set_rate(self, fps: float) -> None:
        """Re-pace this slot's connections."""
        self._set_media_rate(fps)

    def set_depth(self, depth: int) -> None:
        """Bound this slot's connection queues."""
        self._set_media_depth(depth)

    def model_failed(self, error: BaseException) -> None:
        """End this slot's session for a model loop that crashed."""
        self._on_model_failure(error)

    def session_failed(self, generation: int, error: str) -> None:
        """End this slot's session, if it is still the one that failed, with *error*."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._end_failed_session, generation, error)

    def _end_failed_session(self, generation: int, error: str) -> None:
        if generation != self._sessions_posted or self._sm.current_state not in _RUNNING_STATES:
            return
        self._sm.send(SessionEvent.ERROR, message=f"session failed: {error}")
        self._sm.send(SessionEvent.STOP_SESSION, reason=EndReason.ERROR, close_reason=error[:64])


class _RecorderFanout:
    """The clip-reading face of every slot's recorder, addressed by session id."""

    def __init__(self, host: SessionHost) -> None:
        self._host = host

    def _recorder_for(self, session_id: str) -> Recorder:
        return self._host.slot_for_any(session_id).slot_recorder

    def manifest(self, session_id: str, start: float, end: float) -> ClipManifest | Pending | Gone:
        """Resolve a clip of *session_id* through the recorder that wrote it."""
        return self._recorder_for(session_id).manifest(session_id, start, end)

    def chunk_path(self, session_id: str, filename: str) -> Path | None:
        """Locate a segment of *session_id* through the recorder that wrote it.

        Raises:
            ClipSessionGoneError: If the recording is unknown or aged out.
        """
        return self._recorder_for(session_id).chunk_path(session_id, filename)


class SessionHost:
    """Serves up to ``max_sessions`` sessions from one batched model.

    A :class:`~reactor_runtime.core.ServiceComponent` and the
    :class:`~reactor_runtime.transport.router.SessionControl` every transport
    reports into, like a :class:`Runner`; each connection is handed to the
    slot whose session it joined.
    """

    name = "runner"
    depends_on: tuple[str, ...] = ()

    def __init__(
        self,
        cfg: RuntimeConfig,
        metrics: RuntimeMetrics,
        slots: int,
        *,
        load_started: float | None = None,
    ) -> None:
        """Build *slots* session slots over one shared journal and step store.

        The first slot observes on *metrics*, the registry the process serves;
        the others observe on registries of their own.

        Args:
            cfg: The configuration for this runtime process.
            metrics: The registry the process serves.
            slots: How many sessions to host at once, at least two.
            load_started: When the assembly began importing the model's
                class, as for :class:`Runner`.
        """
        self._cfg = cfg
        self._events = EventStream()
        self._step_store = (
            StepStore(cfg.step_results, self._on_step_saved) if cfg.step_results.enabled else None
        )
        self._slots = [
            _SessionSlot(
                cfg,
                metrics if index == 0 else RuntimeMetrics(version="", model=cfg.model_ref),
                host=self,
                events=self._events,
                step_store=self._step_store,
                load_started=load_started,
            )
            for index in range(slots)
        ]
        for slot in self._slots:
            slot.request_shutdown = self._forward_shutdown
        self._model: BatchedReactorApp | None = None
        self._bridge: ModelBridge | None = None
        self._loaded: asyncio.Future[str] | None = None
        self._keys = itertools.count(1)
        self._routes: dict[int, _Route] = {}
        self._slot_keys: dict[int, int] = {}
        self._conn_slots: dict[ConnId, _SessionSlot] = {}
        self._recorders = _RecorderFanout(self)
        self._draining = False
        self.request_shutdown: ShutdownHook = _no_shutdown

    # -- lifecycle (ServiceComponent) -----------------------------------------

    async def start(self) -> None:
        """Load the model once, then bring every slot up on it."""
        loop = asyncio.get_running_loop()
        self._loaded = loop.create_future()
        load = loop.create_task(self._load())
        await asyncio.gather(*(slot.start() for slot in self._slots))
        await load

    async def _load(self) -> None:
        """Load the model and bind its output to the host's routing sinks."""
        assert self._loaded is not None
        try:
            model_cls = import_model_class(self._cfg.model_ref)
            if not issubclass(model_cls, BatchedReactorApp):
                raise TypeError(f"{model_cls.__qualname__} is not a BatchedReactorApp")
            contract = ModelContract.of(model_cls)
            model = model_cls()
            await asyncio.to_thread(model.load, self._cfg.config_path)
            bridge = ModelBridge(model, contract)
            bridge.bind_outbound(
                broadcast=self._route_message,
                addressed=self._route_reply,
                media=self._route_media,
                media_ops=MediaOps(
                    flush=self._route_flush,
                    set_rate=self._route_rate,
                    set_depth=self._route_depth,
                ),
                failure=self._model_failed,
                step=self._route_step,
            )
            model.bind_session_failure(self._session_failed)
            bridge.start()
        except Exception as error:
            self._loaded.set_exception(error)
            # Every slot awaits the load and reports it; nothing else reads it.
            self._loaded.exception()
            return
        self._model = model
        self._bridge = bridge
        self._loaded.set_result(type(model).__name__)

    async def loaded(self) -> str:
        """Wait for the model to load, returning its class name, or raise why it did not."""
        assert self._loaded is not None
        return await asyncio.shield(self._loaded)

    async def drain(self) -> None:
        """Refuse new sessions, and let every running session end on grace."""
        self._draining = True
        await asyncio.gather(*(slot.drain() for slot in self._slots))

    async def stop(self) -> None:
        """Release every slot, then the shared step store, then the model."""
        await asyncio.gather(*(slot.stop() for slot in self._slots))
        if self._step_store is not None:
            await asyncio.to_thread(self._step_store.close)
        if self._bridge is not None:
            await self._bridge.stop()

    def health(self) -> Health:
        """Report unhealthy once any slot is terminated: the shared model is gone."""
        if any(slot.sm_state is SessionState.TERMINATED for slot in self._slots):
            return Health(HealthStatus.UNHEALTHY, "session terminated")
        return Health.healthy()

    def state(self) -> RuntimeState:
        """Report ``available`` while a slot is free, ``serving`` when none is or while draining."""
        states = [slot.sm_state for slot in self._slots]
        if SessionState.TERMINATED in states:
            return RuntimeState.TERMINATED
        if SessionState.CREATED in states:
            return RuntimeState.LOADING
        if SessionState.READY in states and not self._draining:
            return RuntimeState.AVAILABLE
        return RuntimeState.SERVING

    def _forward_shutdown(self, *, failure: bool = False) -> None:
        self.request_shutdown(failure=failure)

    # -- the model's side ---------------------------------------------------------

    @property
    def model(self) -> BatchedReactorApp:
        """The loaded model."""
        assert self._model is not None
        return self._model

    @property
    def contract(self) -> ModelContract:
        """The loaded model's contract."""
        assert self._bridge is not None
        return self._bridge.contract

    def open_key(self, slot: _SessionSlot) -> int:
        """Mint the model-side key of the session *slot* just started, retiring its last one."""
        key = next(self._keys)
        old = self._slot_keys.get(id(slot))
        if old is not None:
            self._routes.pop(old, None)
        self._slot_keys[id(slot)] = key
        self._routes[key] = _Route(slot, slot.generation)
        self._conn_slots = {
            cid: owner for cid, owner in self._conn_slots.items() if owner is not slot
        }
        return key

    def _route(self) -> _Route | None:
        key = current_session_key()
        route = self._routes.get(key) if key is not None else None
        if route is None:
            logger.debug("dropping model output for no live session", key=key)
        return route

    def _route_media(self, chunk: MediaChunk) -> None:
        route = self._route()
        if route is not None:
            route.slot.deliver_media(dataclasses.replace(chunk, session=route.generation))

    def _route_message(self, message: ModelMessage) -> None:
        route = self._route()
        if route is not None:
            route.slot.deliver_message(message)

    def _route_reply(
        self, conn_id: ConnId, message: ModelMessage | CommandFailure | None, request_id: str | None
    ) -> None:
        route = self._route()
        if route is not None:
            route.slot.deliver_reply(conn_id, message, request_id)

    def _route_step(self, step: CompletedStep) -> None:
        route = self._route()
        if route is not None:
            route.slot.deliver_step(dataclasses.replace(step, session=route.generation))

    def _for_each_or_current(self, act: Callable[[_SessionSlot], None]) -> None:
        """Apply a playout control to the current session's slot, or every slot outside one."""
        if current_session_key() is None:
            for slot in self._slots:
                act(slot)
            return
        route = self._route()
        if route is not None:
            act(route.slot)

    def _route_flush(self) -> None:
        self._for_each_or_current(lambda slot: slot.flush())

    def _route_rate(self, fps: float) -> None:
        self._for_each_or_current(lambda slot: slot.set_rate(fps))

    def _route_depth(self, depth: int) -> None:
        self._for_each_or_current(lambda slot: slot.set_depth(depth))

    def _model_failed(self, error: BaseException) -> None:
        for slot in self._slots:
            slot.model_failed(error)

    def _session_failed(self, key: int, error: str) -> None:
        route = self._routes.get(key)
        if route is not None:
            route.slot.session_failed(route.generation, error)

    def _on_step_saved(self, saved: Any) -> None:
        self.slot_for_any(saved.session_id)._on_step_saved(saved)

    # -- session control (driven by the HTTP routes) --------------------------

    def capacity(self) -> tuple[int, int]:
        """Return how many sessions the process hosts at once, and how many it will start."""
        if self._draining:
            return len(self._slots), 0
        free = sum(slot.sm_state is SessionState.READY for slot in self._slots)
        return len(self._slots), free

    def _running_slot(self, sid: str) -> _SessionSlot | None:
        for slot in self._slots:
            if slot.hosted_id == sid and slot.sm_state not in (
                SessionState.READY,
                SessionState.CREATED,
                SessionState.TERMINATED,
            ):
                return slot
        return None

    def slot_for_any(self, session_id: str) -> _SessionSlot:
        """Return the slot that runs or last ran *session_id*, else the first slot."""
        return next((s for s in self._slots if s.hosted_id == session_id), self._slots[0])

    def _require_slot(self, sid: str) -> _SessionSlot:
        slot = self._running_slot(sid)
        if slot is None:
            raise UnknownSessionError
        return slot

    def start_session(self, params: Mapping[str, Any]) -> str:
        """Open a session in a free slot, under the caller's ``session_id`` or a minted one.

        Returns:
            The session's id, which also addresses its transport routes.

        Raises:
            InvalidSessionStartError: If the body or a starting command is refused.
            SessionTransitionError: If the id is already running, the model
                is loading or terminated, or the host is draining, which is
                refused as a start against a closing session.
            NoFreeSlotError: If every slot is serving.
        """
        if self._draining:
            raise SessionTransitionError("start", SessionState.CLOSING)
        sid = str(params.get("session_id") or uuid.uuid4())
        running = self._running_slot(sid)
        if running is not None:
            raise SessionTransitionError("start", running.sm_state)
        states = [slot.sm_state for slot in self._slots]
        for blocking in (SessionState.TERMINATED, SessionState.CREATED):
            if blocking in states:
                raise SessionTransitionError("start", blocking)
        slot = next((s for s in self._slots if s.sm_state is SessionState.READY), None)
        if slot is None:
            raise NoFreeSlotError(len(self._slots))
        with session_scope(sid):
            slot.start_session({**params, "session_id": sid})
        return sid

    def stop_session(
        self, *, moderated: bool = False, reason: str = "", session_id: str | None = None
    ) -> None:
        """Close the session *session_id*, leaving the other sessions running.

        Raises:
            SessionIdRequiredError: If no session id is given.
            UnknownSessionError: If no running session has that id.
            SessionTransitionError: If the session is already closing.
        """
        if session_id is None:
            raise SessionIdRequiredError
        slot = self._require_slot(session_id)
        with session_scope(session_id):
            slot.stop_session(moderated=moderated, reason=reason)

    def descriptor(self, session_id: str | None = None) -> dict[str, Any]:
        """Describe the session *session_id*.

        Raises:
            SessionIdRequiredError: If no session id is given.
            UnknownSessionError: If no running session has that id.
        """
        if session_id is None:
            raise SessionIdRequiredError
        return self._require_slot(session_id).descriptor()

    def sessions(self) -> list[dict[str, Any]]:
        """Describe every running session."""
        return [slot.descriptor() for slot in self._slots if slot.sm_state in _RUNNING_STATES]

    def schema(self) -> dict[str, Any]:
        """Return the model's command contract as an OpenAPI document."""
        return self._slots[0].schema()

    @property
    def events(self) -> EventStream:
        """The egress journal every slot writes, each fact tagged with its session."""
        return self._events

    def event_session(self, sid: str) -> str:
        """Return the id the journal tags *sid*'s events with: *sid* itself."""
        return sid

    @property
    def recorder(self) -> _RecorderFanout:
        """The clip-reading face over every slot's recorder."""
        return self._recorders

    @property
    def step_store(self) -> StepStore | None:
        """The store every slot's saved steps are kept in, or ``None`` when off."""
        return self._step_store

    def uploads_for(self, sid: str) -> UploadStore:
        """Return the upload store of the running session *sid*.

        Raises:
            UnknownSessionError: If no running session has that id.
        """
        return self._require_slot(sid).upload_store

    def upload_store_holding(self, upload_id: str) -> UploadStore:
        """Return the upload store a slot named *upload_id* is reserved in."""
        for slot in self._slots:
            try:
                slot.upload_store.expected_size(upload_id)
            except UnknownUploadError:
                continue
            except UploadAlreadyCompleteError:
                pass
            return slot.upload_store
        return self._slots[0].upload_store

    # -- SessionControl -----------------------------------------------------------

    def require_session_running(self, sid: str) -> None:
        """Admit a request only against a running session this host serves.

        Raises:
            UnknownSessionError: If no slot runs *sid*.
            SessionNotRunningError: If its session is closing.
        """
        slot = self._require_slot(sid)
        if slot.sm_state not in _RUNNING_STATES:
            raise SessionNotRunningError

    def new_conn_id(self, sid: str | None = None) -> ConnId:
        """Mint a connection id for *sid*'s session that no other slot holds."""
        if sid is None:
            raise SessionIdRequiredError
        slot = self._require_slot(sid)
        for _ in range(_MAX_CONN_ID_DRAWS):
            conn_id = slot.new_conn_id()
            owner = self._conn_slots.get(conn_id)
            if owner is None or owner is slot:
                self._conn_slots[conn_id] = slot
                return conn_id
        return slot.new_conn_id()

    def track_map(self) -> Mapping[str, Any]:
        """Return the model's declared track manifest."""
        return self._slots[0].track_map()

    def _slot_of(self, conn_id: ConnId) -> _SessionSlot | None:
        return self._conn_slots.get(conn_id)

    def _to_slot(self, conn_id: ConnId, act: Callable[[_SessionSlot], None]) -> None:
        slot = self._slot_of(conn_id)
        if slot is None:
            logger.debug("dropping a fact for a connection no slot holds", conn_id=conn_id)
            return
        with session_scope(slot.hosted_id):
            act(slot)

    def offer_admitted(self, conn_id: ConnId) -> None:
        """Stamp an admitted offer in the slot that minted its id."""
        self._to_slot(conn_id, lambda slot: slot.offer_admitted(conn_id))

    def connection_opened(self, conn: Connection) -> None:
        """Register a live connection with its session's slot."""
        slot = self._slot_of(conn.id)
        if slot is None:
            logger.warning("refusing a connection no session minted", conn_id=conn.id)
            if self._slots[0]._loop is not None:
                self._slots[0]._spawn_teardown(conn.close())
            return
        with session_scope(slot.hosted_id):
            slot.connection_opened(conn)

    def connection_closed(self, conn_id: ConnId) -> None:
        """Drop a connection from its session's slot."""
        self._to_slot(conn_id, lambda slot: slot.connection_closed(conn_id))

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        """Record a negotiation answer with its session's slot."""
        self._to_slot(conn_id, lambda slot: slot.connection_answered(conn_id, answer))

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        """Hand an inbound frame to its session's slot."""
        self._to_slot(
            conn_id, lambda slot: slot.message_received(conn_id, payload, version, channel)
        )

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        """Route an inbound media frame to its session."""
        slot = self._slot_of(conn_id)
        if slot is not None:
            slot.media_received(conn_id, track, frame)

    def keepalive(self, conn_id: ConnId) -> None:
        """Note liveness for a connection."""
        self._to_slot(conn_id, lambda slot: slot.keepalive(conn_id))

    def resume_track(self, conn_id: ConnId, name: str) -> None:
        """Resume an outbound track for one connection."""
        self._to_slot(conn_id, lambda slot: slot.resume_track(conn_id, name))

    def pause_track(self, conn_id: ConnId, name: str) -> None:
        """Pause an outbound track for one connection."""
        self._to_slot(conn_id, lambda slot: slot.pause_track(conn_id, name))

    def publish_requested(self, conn_id: ConnId, name: str, request_id: str) -> None:
        """Arbitrate a publish-track claim within its session."""
        self._to_slot(conn_id, lambda slot: slot.publish_requested(conn_id, name, request_id))

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        """Release an inbound track a connection had claimed."""
        self._to_slot(conn_id, lambda slot: slot.unpublish_track(conn_id, name))

    def file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        """Hand an out-of-band upload to its session."""
        self._to_slot(conn_id, lambda slot: slot.file_uploaded(conn_id, upload_id))

    def schema_requested(self, conn_id: ConnId, request_id: str) -> None:
        """Answer a client's schema request."""
        self._to_slot(conn_id, lambda slot: slot.schema_requested(conn_id, request_id))

    def clip_requested(self, conn_id: ConnId, duration_seconds: float, request_id: str) -> None:
        """Resolve a snap-clip request against its session's recorder."""
        self._to_slot(
            conn_id, lambda slot: slot.clip_requested(conn_id, duration_seconds, request_id)
        )

    def recording_requested(self, conn_id: ConnId, request_id: str) -> None:
        """Resolve a full-session recording request against its session's recorder."""
        self._to_slot(conn_id, lambda slot: slot.recording_requested(conn_id, request_id))

    def client_stats_received(self, conn_id: ConnId, batch: ClientStatsBatch) -> None:
        """Journal a client quality batch under its session."""
        self._to_slot(conn_id, lambda slot: slot.client_stats_received(conn_id, batch))


__all__ = [
    "NoFreeSlotError",
    "SessionHost",
    "SessionIdRequiredError",
    "batched_capacity",
]
