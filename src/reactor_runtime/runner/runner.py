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
import importlib.metadata
import time
import uuid
from collections.abc import Callable, Coroutine, Mapping
from typing import Any

from reactor_runtime.codes import INVALID_COMMAND, UNRESOLVED_UPLOAD
from reactor_runtime.core import (
    JOURNAL_EVENTS,
    ClientConnected,
    ClientDisconnected,
    CommandFailure,
    Connection,
    ConnectionSink,
    ConnId,
    EndReason,
    FileUploaded,
    Health,
    HealthStatus,
    InputFrame,
    MediaChunk,
    RuntimeConfig,
    RuntimeState,
    ServiceComponent,
    SessionEnded,
    SessionEvent,
    SessionStarted,
    SessionState,
    TrackDirection,
    Transition,
    TransitionEvent,
)
from reactor_runtime.event_stream import EventStream
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.bridge import ModelBridge
from reactor_runtime.interface.internal.reactor_core import MediaOps
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.log import get_logger, set_session_id, set_state
from reactor_runtime.manifest import import_model_class
from reactor_runtime.message_gateway import InboundCommand, MessageGateway
from reactor_runtime.metrics import (
    UNKNOWN_COMMAND,
    CommandMetrics,
    MetricsRecorder,
    ModelMetrics,
    RuntimeMetrics,
)
from reactor_runtime.protocol import Channel, Codec, ProtocolVersion, select
from reactor_runtime.recording import ClipResult, Recorder, RecorderError
from reactor_runtime.runner.connection_manager import ConnectionManager
from reactor_runtime.runner.offer_epochs import OfferEpochs
from reactor_runtime.runner.state_machine import SessionStateMachine
from reactor_runtime.transport.router import (
    SessionNotRunningError,
    SessionTransitionError,
    UnknownSessionError,
)
from reactor_runtime.upload_store import UnknownUploadError, UploadStore

_RUNNING_STATES = frozenset({SessionState.WAITING, SessionState.STREAMING, SessionState.ORPHANED})

# The states a stale wire can land in. An offer is only admitted while a
# session runs, so a connection whose negotiation completes after the session
# moved on arrives in one of these — and must not join the registry.
_STALE_CONNECTION_STATES = frozenset(
    {SessionState.READY, SessionState.CLOSING, SessionState.TERMINATED}
)

# The lifecycle word reported for each session state. Coarser than the session
# machine on purpose: an outside observer cares whether the process is loading,
# free, occupied, or finished — not which serving sub-state the session is in.
_RUNTIME_STATES: dict[SessionState, RuntimeState] = {
    SessionState.CREATED: RuntimeState.LOADING,
    SessionState.READY: RuntimeState.AVAILABLE,
    SessionState.WAITING: RuntimeState.SERVING,
    SessionState.STREAMING: RuntimeState.SERVING,
    SessionState.ORPHANED: RuntimeState.SERVING,
    SessionState.CLOSING: RuntimeState.SERVING,
    SessionState.TERMINATED: RuntimeState.TERMINATED,
}


def _stamp_log_state(state: SessionState) -> None:
    """Bind the log's state context to *state*, at both granularities.

    Records carry the machine's own word and the coarse word the health route
    serves, so a reader can filter by whichever vocabulary the surface they are
    looking at showed them.
    """
    set_state(state.name.lower(), _RUNTIME_STATES[state].value)


def _recording_id_from(params: Mapping[str, Any]) -> str:
    """Resolve a session's recording id from its start parameters.

    A ``session_id`` in *params* is adopted as the recording id, so a caller can
    align both clips and logs with the id it knows the session by. Absent one, a
    fresh id is minted per session so sequential recordings in a reused process
    never overwrite each other.
    """
    return str(params.get("session_id") or uuid.uuid4())


# How long to wait for an upload's bytes to arrive when a command or notification
# references it before they are written. A client references an upload over the
# data channel while its bytes are still being delivered on a separate request,
# so a brief wait resolves the reference instead of dropping the command; it is
# generous enough to cover that delivery yet bounded so a genuinely-missing upload
# fails in reasonable time.
_UPLOAD_RESOLVE_TIMEOUT_SECONDS = 10.0

# A runtime process hosts exactly one session, so its id is fixed rather than
# minted: an all-zero UUID every client addresses. Routes still carry and
# validate the id, but this is the only value the runtime accepts.
SESSION_ID = "00000000-0000-0000-0000-000000000000"

# A standalone runtime has no cluster; it reports a constant so the v0 client's
# schema, which requires the field, is satisfied.
_CLUSTER = "local"
_WEBRTC_TRANSPORT_VERSION = "1.0"
_V0_PROTOCOL = "v0"
# Client track directions are the mirror of the model's: a track the model sends
# out is one the client only receives, and one the model takes in is one the
# client only sends.
_CLIENT_DIRECTION = {"out": "recvonly", "in": "sendonly"}

# The close reason a drain sends to clients. The runtime initiates this stop,
# so the runtime words it; every other close reason arrives from the platform.
# Kept within the 64-character bound the stop route enforces on the platform's.
_DRAIN_CLOSE_REASON = "Session ended: the server is shutting down."

logger = get_logger(__name__)


def _server_version() -> str:
    """Return this runtime's package version, or a zero placeholder if unknown."""
    try:
        return importlib.metadata.version("reactor-runtime")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _no_shutdown() -> None:
    """Default process-shutdown hook — a no-op until the service wires one in."""


class Runner(ServiceComponent, ConnectionSink):
    """Orchestrates one model and its session, transport- and platform-neutral.

    Composes the session state machine, the connection multiplexer, the inbound
    message gateway, and the egress journal, and builds the model bridge in
    :meth:`start`. It is the connection sink every transport reports into, and
    funnels each session transition through one listener.
    """

    name = "runner"
    depends_on: tuple[str, ...] = ()

    def __init__(self, cfg: RuntimeConfig, metrics: RuntimeMetrics | None = None) -> None:
        """Wire the session machinery and local components that need no model.

        The bridge is built later, in :meth:`start`, once the model class is
        known; everything here is model-independent.

        Args:
            cfg: The configuration for this runtime process.
            metrics: The registry the session instruments observe on. A runner
                built without one records on a registry of its own, so the
                instruments are always live and only the assembled process
                serves them.
        """
        self._cfg = cfg
        self._metrics = metrics or RuntimeMetrics(version=_server_version(), model=cfg.model_ref)
        self._sm = SessionStateMachine()
        # The log's state context starts at the machine's starting state, so the
        # model-load window — records written before any transition — is already
        # stamped; every later move re-stamps in _dispatch_transition.
        _stamp_log_state(self._sm.current_state)
        self._sm.on_transition(self._dispatch_transition)
        # The session surface of the metrics is one listener over the same moves
        # the journal carries, so no session code below calls an instrument.
        self._metrics_recorder = MetricsRecorder(self._metrics, state=self._sm.current_state)
        self._sm.on_transition(self._metrics_recorder.observe)
        self._command_metrics = CommandMetrics(self._metrics)
        self._model_metrics = ModelMetrics(self._metrics)
        self._events = EventStream()
        self._uploads = UploadStore()
        self._recorder = Recorder(
            cfg.recording,
            on_clip_ready=self._on_clip_ready,
            on_chunk_ready=self._on_chunk_ready,
        )
        self._connections = ConnectionManager(state_machine=self._sm)
        self._offer_epochs = OfferEpochs()
        # Playout settings the model set through its output handle, remembered
        # so a connection that opens later starts with them.
        self._media_rate: float | None = None
        self._media_depth: int | None = None
        # Bumped by every flush. A media fan-out captures it at entry and
        # abandons the remaining connections when it moves, so a flush that
        # lands mid-broadcast cuts every connection, not just the one whose
        # pacer happened to be blocked.
        self._media_generation = 0
        # One codec per wire version, built on first use. Inbound decode is
        # driven by the version a connection negotiated; outbound encode picks
        # the codec for each target connection, so a mixed-version session is
        # addressed in each client's own version.
        self._codecs: dict[ProtocolVersion, Codec] = {}
        self._gateway = MessageGateway(sink=self, on_command=self._submit_command)
        self._bridge: ModelBridge | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inbound: set[asyncio.Task[None]] = set()
        self._teardown: set[asyncio.Task[None]] = set()
        self._orphan_task: asyncio.Task[None] | None = None
        self._session_id = SESSION_ID
        # The session's own id, resolved per session as the start transition is
        # applied (see _dispatch_transition): the id a recording is stored and
        # addressed under, and the id stamped on the session's log records.
        # Separate from the fixed transport session id so a
        # caller can align both with the id it knows the session by; a session
        # started without one mints a fresh id, so sequential recordings in a
        # reused process never share a directory and the logs of one session are
        # never read as another's. The construction value is an unused placeholder.
        self._recording_id = SESSION_ID
        # Names the log's current session binding, so the release that follows a
        # session retires that binding and not a later session's. Zero until the
        # first session binds one.
        self._log_binding = 0
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

        The model load runs off the event loop (it may block while it reads
        weights), so the HTTP surface — already up by the time this runs — stays
        responsive throughout: a client subscribed to ``/events`` observes the
        ``initializing`` self-loop journalled before the load, then the
        ``initialization_success``/``initialization_fail`` transition when it ends.
        """
        self._loop = asyncio.get_running_loop()
        logger.info("loading model", model=self._cfg.model_ref)
        # Journal the loading phase before the (blocking) load, so a consumer
        # replaying /events sees the runtime is initializing during the load
        # window rather than nothing until READY. A self-loop on CREATED: no
        # state change, no side effect (the bridge is not built yet).
        self._sm.send(SessionEvent.INITIALIZING)
        started_at = time.monotonic()
        try:
            model_cls = import_model_class(self._cfg.model_ref)
            contract = ModelContract.of(model_cls)
            model = model_cls()
            await asyncio.to_thread(model.load, self._cfg.config_path)
            bridge = ModelBridge(model, contract)
            bridge.bind_outbound(
                broadcast=self._broadcast_message,
                addressed=self._send_addressed,
                media=self._emit_media,
                media_ops=MediaOps(
                    flush=self._flush_media,
                    set_rate=self._set_media_rate,
                    set_depth=self._set_media_depth,
                ),
                failure=self._on_model_failure,
            )
            bridge.start()
        except Exception:
            self._model_metrics.load_failed(since=started_at)
            logger.exception("model failed to load; terminating the session")
            self._sm.send(SessionEvent.INITIALIZATION_FAIL)
            return
        self._model_metrics.loaded(since=started_at)
        self._bridge = bridge
        self._command_metrics.declare(contract.commands)
        self._model_metrics.declare(
            name for name, info in contract.tracks.items() if info.direction is TrackDirection.OUT
        )
        self._sm.send(SessionEvent.INITIALIZATION_SUCCESS)
        logger.info(
            "model loaded; session ready",
            model=type(model).__name__,
            tracks=len(contract.tracks),
            commands=len(contract.commands),
        )

    async def drain(self) -> None:
        """Stop accepting new sessions and let an active one end on grace.

        A running session is asked to stop and given the grace period to unwind
        to ready; the model itself stays up until :meth:`stop`. The stop carries
        a close reason authored here — the runtime initiates this stop, so the
        runtime words it — and the clients are told before their connections
        close, the same notice a platform-reasoned stop sends.
        """
        self._accepting = False
        if self._sm.current_state in _RUNNING_STATES:
            self._sm.send(
                SessionEvent.STOP_SESSION,
                reason=EndReason.STOPPED,
                close_reason=_DRAIN_CLOSE_REASON,
            )
            await self._await_ready(self._cfg.grace_period)

    async def stop(self) -> None:
        """Release the model, bringing its thread down last.

        Any session teardown still in flight is awaited first, so connection
        closes finish before the model thread goes down — the model is the last
        thing released even when ``stop`` races a session that is still closing.
        """
        self._cancel_orphan_timeout()
        await self._drain_teardown()
        await asyncio.to_thread(self._recorder.close)
        if self._bridge is not None:
            await self._bridge.stop()

    async def _await_ready(self, grace: float) -> None:
        """Wait up to *grace* seconds for the session to unwind to ready.

        The loop yields to the event loop before testing the deadline, so even a
        zero grace period gives an already-scheduled cleanup a turn to run rather
        than returning while the session is still closing.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace
        while self._sm.current_state is not SessionState.READY:
            await asyncio.sleep(0)
            if loop.time() >= deadline:
                return
            await asyncio.sleep(0.01)

    def health(self) -> Health:
        """Report unhealthy only for a terminated session.

        Every other state is healthy: a model still loading is working as
        intended, just not yet available — :meth:`state` carries that word.
        """
        if self._sm.current_state is SessionState.TERMINATED:
            return Health(HealthStatus.UNHEALTHY, "session terminated")
        return Health.healthy()

    def state(self) -> RuntimeState:
        """Report the lifecycle word for the session's current state."""
        return _RUNTIME_STATES[self._sm.current_state]

    # -- inbound (ConnectionSink) ---------------------------------------------

    def connection_opened(self, conn: Connection) -> None:
        """Register a connection whose wire has reached its connected state.

        The model's playout settings (rate, queue depth) apply to every
        connection, so one that opens after they were set receives them here.

        A wire that connects after its session moved on — its negotiation
        finishing once teardown began, after the session unwound to ready, or
        with a later session already running — is closed instead of registered.
        Registered, it would sit outside its own session's teardown snapshot
        and receive another session's traffic. The cross-session case is caught
        by the epoch stamped on the offer at admission; a connection with no
        stamp (a transport that does not stamp, or a directly driven test) is
        gated on state alone.
        """
        stale_epoch = self._offer_epochs.consume(conn.id)
        if self._sm.current_state in _STALE_CONNECTION_STATES or stale_epoch:
            logger.warning(
                "refusing a connection that does not belong to the live session",
                conn_id=conn.id,
                state=self._sm.current_state.name.lower(),
            )
            if self._loop is not None:
                self._spawn_teardown(conn.close())
            return
        self._connections.register(conn)
        if self._media_depth is not None:
            conn.set_media_depth(self._media_depth)
        if self._media_rate is not None:
            conn.set_media_rate(self._media_rate)

    def connection_closed(self, conn_id: ConnId) -> None:
        """Drop a previously opened connection that has gone away."""
        self._connections.drop(conn_id)

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        """Record a transport's negotiation answer for a connection.

        The answer is an opaque, transport-agnostic payload (for WebRTC, the SDP
        answer as ``{"type", "sdp"}``); the runner does not parse it. It arrives
        before the connection's wire connects, so it rides a ``CONNECTION_ANSWERED``
        self-loop — a move that changes no state and touches no occupancy count
        — whose detail carries the connection id and the answer for the egress
        journal to hand back to the offering client. An answer for a session
        that is no longer active is rejected by the machine and dropped.
        """
        self._sm.send(SessionEvent.CONNECTION_ANSWERED, conn_id=conn_id, answer=dict(answer))

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        """Hand an inbound frame to the gateway for decoding and dispatch.

        The gateway decodes asynchronously, so the work is scheduled on the
        runtime loop and tracked until it completes. The frame is decoded in the
        codec the connection negotiated (*version*) and as the family its
        physical *channel* carries.

        The arrival is stamped here, the first place in the runtime that sees the
        frame, and rides any command it carries. A frame that arrives while the
        loop is busy waits before the gateway reads it, and the wait belongs to
        the command's ingress rather than being invisible.
        """
        if self._loop is None:
            return
        received_at = time.monotonic()
        task = self._loop.create_task(
            self._gateway.handle(conn_id, payload, channel, version, received_at=received_at)
        )
        self._inbound.add(task)
        task.add_done_callback(self._inbound.discard)

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        """Route an inbound media frame to its track on the model."""
        if self._bridge is not None:
            self._bridge.push_media(track, frame)

    def keepalive(self, conn_id: ConnId) -> None:
        """Note liveness for a connection."""
        self._connections.note_keepalive(conn_id)

    def resume_track(self, conn_id: ConnId, name: str) -> None:
        """Resume an outbound track for one connection, at the client's request."""
        self._connections.resume_track(conn_id, name)

    def pause_track(self, conn_id: ConnId, name: str) -> None:
        """Pause an outbound track for one connection, at the client's request."""
        self._connections.pause_track(conn_id, name)

    def publish_requested(self, conn_id: ConnId, name: str, request_id: str) -> None:
        """Arbitrate a publish-track claim first-come-first-served, then reply.

        The claim is granted only when the track is unheld; either way the
        outcome is sent back to the requesting connection on whichever channel
        its codec places the reply, correlated by *request_id*, so the client's
        pending request resolves instead of timing out. Routing through
        ``send_response`` keeps channel selection the codec's decision, the same
        path the schema reply takes.
        """
        granted = self._connections.publish_track(conn_id, name)
        self._connections.send_response(
            conn_id,
            lambda version: self._codec_for(version).encode_publish_response(
                request_id, granted=granted
            ),
        )

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        """Release an inbound track a connection had claimed."""
        self._connections.unpublish_track(conn_id, name)

    def file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        """Resolve a client's out-of-band upload and cross it into the model.

        Only fetched and dispatched when the model declares a ``@file_uploaded``
        hook, so an upload no model reads costs no byte copy. The fetch is
        asynchronous, so it is scheduled on the runtime loop and tracked until it
        completes.
        """
        if self._loop is None or self._bridge is None:
            return
        if self._bridge.contract.lifecycle.file_uploaded is None:
            return
        task = self._loop.create_task(self._dispatch_file_uploaded(conn_id, upload_id))
        self._inbound.add(task)
        task.add_done_callback(self._inbound.discard)

    def schema_requested(self, conn_id: ConnId, request_id: str) -> None:
        """Answer a client's schema request, correlated by *request_id*.

        The reply carries the model's rendered OpenAPI contract and is sent to
        the requesting connection on whichever channel its wire version places
        it. A request that arrives before the model is loaded is dropped.
        """
        if self._bridge is None:
            return
        openapi = self._render_schema()
        self._connections.send_response(
            conn_id,
            lambda version: self._codec_for(version).encode_schema_response(request_id, openapi),
        )

    def clip_requested(self, conn_id: ConnId, duration_seconds: float, request_id: str) -> None:
        """Resolve a snap-clip request and reply, correlated by *request_id*.

        The recorder returns the clip's marker range and playlist URL at once;
        the reply rides whichever channel the connection's codec places it on. A
        request the recorder cannot serve is answered with a clip-failed reply.
        """
        self._reply_clip(conn_id, request_id, lambda: self._recorder.request_clip(duration_seconds))

    def recording_requested(self, conn_id: ConnId, request_id: str) -> None:
        """Resolve a full-session recording request and reply, correlated by *request_id*."""
        self._reply_clip(conn_id, request_id, lambda: self._recorder.request_recording())

    def _reply_clip(
        self, conn_id: ConnId, request_id: str, resolve: Callable[[], ClipResult]
    ) -> None:
        """Resolve a clip request and send the clip-ready or clip-failed reply."""
        try:
            clip = resolve().to_dict()
        except (RecorderError, ValueError) as error:
            reason = str(error) or "clip request failed"
            self._connections.send_response(
                conn_id,
                lambda version: self._codec_for(version).encode_clip_failed(request_id, reason),
            )
            return
        self._connections.send_response(
            conn_id,
            lambda version: self._codec_for(version).encode_clip_ready(request_id, clip),
        )

    def _on_clip_ready(self, clip: ClipResult) -> None:
        """Journal a clip once its boundary segment lands, hopping onto the loop.

        The recorder fires this from its own watcher thread, so the emit is
        scheduled on the runtime loop where the egress journal is single-writer.
        """
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._emit_clip_ready, clip)

    def _emit_clip_ready(self, clip: ClipResult) -> None:
        """Journal a clip-ready fact as a self-loop move on the session machine."""
        self._sm.send(
            SessionEvent.CLIP_READY,
            session_id=clip.session_id,
            kind=clip.kind,
            start_marker=clip.start_marker,
            end_marker=clip.end_marker,
            now_marker=clip.now_marker,
            predicted_ready_at_ms=clip.predicted_ready_at_ms,
            playlist_url=clip.playlist_url,
        )

    def _on_model_failure(self, error: BaseException) -> None:
        """End the session for a model that crashed, hopping onto the loop.

        The model reports its run-loop crash from its own thread, so the
        terminal move is scheduled on the runtime loop, where the state machine
        and the egress journal are single-writer.
        """
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._evict_on_failure, str(error) or repr(error))

    def _evict_on_failure(self, error: str) -> None:
        """Evict the session to terminated, carrying the crash as its reason."""
        self._sm.send(SessionEvent.EVICTION, reason=EndReason.ERROR, error=error)

    def _on_chunk_ready(self, recording_id: str, idx: int) -> None:
        """Journal a recording segment once it closes, hopping onto the loop.

        The recorder fires this from its own watcher thread, so the emit is
        scheduled on the runtime loop where the egress journal is single-writer.
        """
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._emit_chunk_ready, recording_id, idx)

    def _emit_chunk_ready(self, recording_id: str, idx: int) -> None:
        """Journal a chunk-ready fact as a self-loop move on the session machine."""
        self._sm.send(SessionEvent.CHUNK_READY, recording_id=recording_id, idx=idx)

    # -- session control (driven by the HTTP routes) --------------------------

    def start_session(self, params: Mapping[str, Any]) -> None:
        """Open a session, moving it from ready to waiting for a client.

        The session id is fixed (:data:`SESSION_ID`), so this only drives the
        state machine. The start is **not** idempotent: it is legal only from
        ``READY``, and a request from any other state — the model still loading
        (``CREATED``), a session already open, a previous one still ``CLOSING``,
        or a ``TERMINATED`` process — is rejected without touching the session.
        The rejection surfaces the current state so the caller can report the
        precise reason. The parameters seed the session's initial state.

        A ``session_id`` in *params* is adopted as this session's own id: the id
        its recording is stored and addressed under, and the id stamped on every
        log record the session writes. A caller can therefore align both clips and
        logs with the id it knows the session by. Absent one, a fresh id is minted
        per session so sequential recordings never overwrite each other. The id is
        resolved as the machine accepts the start, so a rejected request leaves a
        live session's id untouched. The transport session id is unaffected: it is
        always :data:`SESSION_ID`.

        Args:
            params: The initial session parameters supplied by the caller.

        Raises:
            SessionTransitionError: If the session is not in a startable state.
        """
        if not self._sm.send(SessionEvent.START_SESSION, params=dict(params)):
            raise SessionTransitionError("start", self._sm.current_state)
        self._offer_epochs.session_started()
        self._model_metrics.session_started()

    def stop_session(self, *, moderated: bool = False, reason: str = "") -> None:
        """Close the active session, leaving the model loaded and ready again.

        Not idempotent, like :meth:`start_session`: a stop is legal only from a
        running state, so one with no running session (``READY``/``CREATED``),
        a session already ``CLOSING``, or a ``TERMINATED`` process is rejected
        and surfaces the current state.

        A moderated stop is a content-moderation verdict against the session:
        it ends with :attr:`~reactor_runtime.core.model.EndReason.MODERATED`
        and the clients are told why before their connections close.

        *reason* is the platform's human-readable description of why the
        session is ending (for example ``"Session ended: the model was
        updated."``). When set, the clients receive a session-ended notice
        carrying it verbatim before their connections close. A moderated stop
        outranks it: a stop carrying both sends only the moderation notice.
        Delivery is best-effort: a session with no live client, or a send that
        fails, is logged and the stop runs regardless.

        Args:
            moderated: Whether the stop enforces a moderation verdict.
            reason: The platform's close reason, empty for a plain stop.

        Raises:
            SessionTransitionError: If there is no running session to stop.
        """
        end_reason = EndReason.MODERATED if moderated else EndReason.STOPPED
        detail: dict[str, Any] = {"reason": end_reason}
        if reason:
            detail["close_reason"] = reason
        if not self._sm.send(SessionEvent.STOP_SESSION, **detail):
            raise SessionTransitionError("stop", self._sm.current_state)

    def new_conn_id(self) -> ConnId:
        """Mint a fresh connection id, delegating to the manager that owns the namespace.

        The connection manager is the single owner of the id namespace, so the
        runner forwards rather than keeping a second counter that could diverge.
        """
        return self._connections.new_conn_id()

    def offer_admitted(self, conn_id: ConnId) -> None:
        """Stamp an admitted offer with the session it was admitted into.

        A transport calls this as it accepts a connection offer. The stamp is
        compared when the wire connects: negotiation is asynchronous, so a wire
        can reach its connected state after its session ended, and if the next
        session is already running by then the state alone looks valid. A
        re-offer on the same id restamps it.
        """
        self._offer_epochs.stamp(conn_id)

    def require_session_running(self, sid: str) -> None:
        """Admit a request only against the live, correctly-addressed session.

        Args:
            sid: The session id the request addressed.

        Raises:
            SessionNotRunningError: If no session is currently live.
            UnknownSessionError: If *sid* is not this runtime's session id.
        """
        if self._sm.current_state not in _RUNNING_STATES:
            raise SessionNotRunningError
        if sid != self._session_id:
            raise UnknownSessionError

    def track_map(self) -> Mapping[str, Any]:
        """Return the model's declared track manifest for connection setup.

        Empty until the model is loaded; a transport reads it to negotiate the
        media a connection carries.
        """
        if self._bridge is None:
            return {}
        return {
            name: {
                "kind": info.kind.value,
                "direction": info.direction.value,
                "rate": info.rate,
            }
            for name, info in self._bridge.contract.tracks.items()
        }

    # -- read-only surface (read by the HTTP route groups) --------------------

    @property
    def events(self) -> EventStream:
        """The egress journal an HTTP egress route streams out."""
        return self._events

    @property
    def uploads(self) -> UploadStore:
        """The upload store the HTTP upload routes write into and the runner reads."""
        return self._uploads

    @property
    def recorder(self) -> Recorder:
        """The recorder the HTTP clip routes read and the runner drives."""
        return self._recorder

    def descriptor(self) -> dict[str, Any]:
        """Describe the session in the shape the client validates against.

        The client requires a fixed session shape — ``cluster``, ``model``,
        ``server_info``, and a ``capabilities`` block it builds its transceivers
        from — and names track directions from the client's perspective, the
        mirror of the model's. This renders that shape from the model's
        contract; ``model`` and ``capabilities`` fill in once the model is
        loaded. The ``commands`` list is intentionally empty here — a client
        reads the model's command contract from the ``/schema`` endpoint, not
        from the session descriptor.
        """
        descriptor: dict[str, Any] = {
            "session_id": self._session_id,
            "state": self._sm.current_state.name.lower(),
            "cluster": _CLUSTER,
            "model": {"name": ""},
            "server_info": {"server_version": _server_version()},
            "selected_transport": {
                "protocol": "webrtc",
                "version": _WEBRTC_TRANSPORT_VERSION,
            },
            "recording": {
                "enabled": self._cfg.recording.enabled,
                "chunk_seconds": self._cfg.recording.chunk_seconds,
            },
        }
        if self._bridge is None:
            return descriptor
        contract = self._bridge.contract
        descriptor["model"] = {"name": self._cfg.model_name or contract.model}
        descriptor["capabilities"] = {
            "protocol_version": _V0_PROTOCOL,
            "tracks": [
                {
                    "name": name,
                    "kind": info.kind.value,
                    "direction": _CLIENT_DIRECTION[info.direction.value],
                }
                for name, info in contract.tracks.items()
            ],
            "commands": [],
        }
        return descriptor

    def schema(self) -> dict[str, Any]:
        """Return the model's command contract as an OpenAPI document.

        The full per-model contract a client uses to drive the model, served by
        the ``/schema`` route. Empty until the model is loaded.
        """
        if self._bridge is None:
            return {}
        return self._render_schema()

    # -- internals ------------------------------------------------------------

    def _render_schema(self) -> dict[str, Any]:
        """Render the loaded model's contract as an OpenAPI document.

        Titled with the name the manifest publishes the model under, so the
        document a client reads over the wire is the one the schema command
        renders from the same directory.
        """
        assert self._bridge is not None
        contract = self._bridge.contract
        return contract.render_schema(name=self._cfg.model_name).to_openapi()

    async def _submit_command(self, command: InboundCommand) -> None:
        """Submit a decoded client command to the model through the bridge.

        Each upload the command references is resolved to its bytes through the
        store and merged into the arguments before validation, so the model
        receives a file rather than a reference. A command that references an
        upload the store cannot produce is journalled as an error and dropped
        rather than submitted half-resolved. An accepted command is journalled on
        the egress stream so a consumer can audit or moderate it; a command the
        contract rejects is journalled as an error instead and never reaches the
        model. The journalled argument record carries the scalar arguments, never
        the resolved file bytes.

        This is also where the command instruments are recorded, because the
        branches below are the outcomes a command that reached the runtime can
        have. The guard above is not one of them: a command arrives over a
        connection, a connection needs a running session, and a session needs a
        loaded model, so a client cannot reach this method before the bridge
        exists.
        """
        if self._bridge is None:
            return
        label = self._command_label(command.name)
        args = dict(command.args)
        resolve_started = time.monotonic()
        try:
            for param, upload_id in command.uploads.items():
                args[param] = await self._uploads.fetch(
                    upload_id, wait_seconds=_UPLOAD_RESOLVE_TIMEOUT_SECONDS
                )
        except UnknownUploadError:
            self._command_metrics.unresolved_upload(label)
            self._sm.send(
                SessionEvent.ERROR,
                message=f"command {command.name!r} references an unresolved upload",
            )
            if command.conn_id is not None:
                self._reject_command(
                    command.conn_id,
                    command.request_id,
                    UNRESOLVED_UPLOAD,
                    f"command {command.name!r} references an unresolved upload",
                )
            return
        # The wait for a client's bytes is the client's latency, so ingress starts
        # again where that wait ended. Counted whole, one command with a file
        # parameter reports the upload and hides the runtime's own cost.
        started_at = command.received_at
        if command.uploads:
            started_at += time.monotonic() - resolve_started
        outcome = await self._bridge.submit_command(
            command.name,
            args,
            conn_id=command.conn_id,
            request_id=command.request_id,
        )
        if outcome.accepted:
            self._command_metrics.accepted(label, since=started_at)
            self._sm.send(
                SessionEvent.COMMAND,
                name=command.name,
                args=dict(command.args),
                conn_id=command.conn_id,
            )
        else:
            self._command_metrics.rejected(label, since=started_at)
            self._sm.send(
                SessionEvent.ERROR,
                message=f"command {command.name!r} rejected ({outcome.field}: {outcome.reason})",
            )
            if command.conn_id is not None:
                self._reject_command(
                    command.conn_id,
                    command.request_id,
                    INVALID_COMMAND,
                    outcome.reason or "command rejected",
                )

    def _command_label(self, name: str) -> str:
        """Return a command name that is safe to label a metric with.

        A client puts any string in a command frame, and a label value that a
        client chooses is a label value without a bound. The names the model
        declares are bounded, so a declared name labels itself and every other
        name shares one series.
        """
        if self._bridge is not None and name in self._bridge.contract.commands:
            return name
        return UNKNOWN_COMMAND

    async def _dispatch_file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        """Fetch an out-of-band upload and hand it to the model as a reactor event.

        An upload the store cannot produce is journalled as an error and dropped;
        a resolved one crosses into the model as a trusted :class:`FileUploaded`.
        """
        if self._bridge is None:
            return
        try:
            file = await self._uploads.fetch(
                upload_id, wait_seconds=_UPLOAD_RESOLVE_TIMEOUT_SECONDS
            )
        except UnknownUploadError:
            self._sm.send(
                SessionEvent.ERROR, message=f"file upload {upload_id!r} could not be resolved"
            )
            return
        self._bridge.dispatch_reactor_event(FileUploaded(file=file, conn_id=conn_id))

    def _codec_for(self, version: ProtocolVersion) -> Codec:
        """Return the codec for *version*, building and caching it on first use."""
        codec = self._codecs.get(version)
        if codec is None:
            codec = select(version)
            self._codecs[version] = codec
        return codec

    def _emit_media(self, chunk: MediaChunk) -> None:
        """Fan one emitted media chunk out to the recorder and the connections.

        Called off the model loop (emit dispatches to a worker thread). Both
        consumers bound their queue the same way — never below the emission
        being handed over — so a whole chunk fits each of them and the fan-out
        costs the producer nothing while they keep up. A consumer that falls
        behind honours ``chunk.wait``, and a chunk emitted with ``drop=True``
        leaves every consumer non-blocking.

        The connections are served first so the archive is never in front of
        the session. A pacer that makes the producer wait is throttling it to
        the playout rate it asked for, and drains on its own thread meanwhile;
        the recorder's wait is bounded instead, because an encoder can stall
        outright. Feeding the recorder second keeps that bounded stall off the
        live path, and leaves its queue the whole broadcast to drain into.
        """
        for track in chunk.bundle.tracks:
            self._model_metrics.emitted(track, chunk.n_frames)
        generation = self._media_generation
        self._connections.broadcast_media(chunk, abort=lambda: self._media_generation != generation)
        # The archive takes the whole chunk even when a flush cut the broadcast
        # short: a playout cut is not an archive boundary.
        self._recorder.on_chunk(chunk)

    def _flush_media(self) -> None:
        """Drop queued media in every connection and cut playout to black.

        The model's ``output.flush()`` lands here. The recorder is not
        flushed: its stream is the session's archive, and a playout cut is
        not an archive boundary.
        """
        self._media_generation += 1
        self._connections.flush_media()

    def _set_media_rate(self, fps: float) -> None:
        """Re-pace every connection now and remember the rate for new ones."""
        self._media_rate = fps
        self._connections.set_media_rate(fps)

    def _set_media_depth(self, depth: int) -> None:
        """Bound every connection's queue now and remember it for new ones."""
        self._media_depth = depth
        self._connections.set_media_depth(depth)

    def _broadcast_message(self, message: ModelMessage) -> None:
        """Broadcast a model message, encoded for each connection's codec."""
        data = message.to_wire_format()["data"]
        self._connections.broadcast(
            lambda version: self._codec_for(version).encode_model_message(message.name, data)[1]
        )

    def _send_addressed(
        self,
        conn_id: ConnId,
        message: ModelMessage | CommandFailure | None,
        request_id: str | None,
    ) -> None:
        """Send a model's reply to one connection, in its codec, correlated.

        A :class:`CommandFailure` is a handler's reported failure and travels as
        an error frame. A ``None`` message is the bodyless acknowledgement of a
        command whose handler returned nothing. Both are sent only when there is
        a request id to correlate, so every response the client receives can be
        matched to the command that caused it, and both are withheld from legacy
        clients, whose commands are fire-and-forget. A failure is journalled
        either way, so a reply the client cannot be told about is still auditable.
        """
        if isinstance(message, CommandFailure):
            self._on_handler_failure(message, conn_id, request_id)
            if request_id is not None:
                self._reject_command(conn_id, request_id, message.code, message.message)
            return
        if message is None:
            if request_id is None:
                return
            self._connections.send_command_ack(
                conn_id,
                lambda version: self._codec_for(version).encode_command_ack(request_id)[1],
            )
            return
        data = message.to_wire_format()["data"]
        self._connections.send(
            conn_id,
            lambda version: self._codec_for(version).encode_model_message(
                message.name, data, request_id=request_id
            )[1],
        )

    def _on_handler_failure(
        self, failure: CommandFailure, conn_id: ConnId, request_id: str | None
    ) -> None:
        """Journal a handler's failure, hopping onto the loop.

        The model reports the failure from its own thread, so the journal move is
        scheduled on the runtime loop, where the state machine and the egress
        journal are single-writer.
        """
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._emit_handler_failure, failure, conn_id, request_id)

    def _emit_handler_failure(
        self, failure: CommandFailure, conn_id: ConnId, request_id: str | None
    ) -> None:
        """Journal a handler failure as a self-loop move on the session machine.

        Carries the connection and the request id so the entry lines up with the
        :attr:`SessionEvent.COMMAND` move that admitted the command, rather than
        leaving an operator to match the two by timestamp.
        """
        self._sm.send(
            SessionEvent.ERROR,
            message=f"command handler failed ({failure.code}: {failure.message})",
            conn_id=conn_id,
            request_id=request_id,
        )

    def _reject_command(self, conn_id: ConnId, request_id: str, code: str, detail: str) -> None:
        """Answer a command with an error, correlated by *request_id*.

        Covers a payload the contract rejects, an upload that cannot be
        resolved, and a handler that raised. The reply is withheld from legacy
        clients, whose commands are fire-and-forget.
        """
        self._connections.send_command_ack(
            conn_id,
            lambda version: self._codec_for(version).encode_command_error(request_id, code, detail)[
                1
            ],
        )

    def _broadcast_moderation_notice(self) -> None:
        """Tell every client the session is ending on a moderation verdict.

        Broadcast synchronously as the session enters ``CLOSING``, before the
        connection teardown is spawned, so the frame is queued on each ordered
        channel ahead of its close and the client sees the verdict rather than a
        bare disconnect.
        """
        self._connections.broadcast_response(
            lambda version: self._codec_for(version).encode_moderation(
                action="terminate",
                message="Session terminated due to policy violation.",
            )
        )

    def _broadcast_session_ended(self, reason: str) -> None:
        """Tell every client why the platform is ending the session.

        *reason* is the platform-authored, human-readable description and is
        delivered verbatim. Broadcast synchronously as the session enters
        ``CLOSING``, before the connection teardown is spawned, so the frame is
        queued on each ordered channel ahead of its close and the client sees
        the reason rather than a bare disconnect. Best-effort by contract: a
        session with no live client or a broadcast that raises logs a warning,
        and the stop proceeds either way.
        """
        if self._connections.count == 0:
            logger.warning("no live client to notify of session end", reason=reason)
            return
        try:
            self._connections.broadcast_response(
                lambda version: self._codec_for(version).encode_session_ended(reason=reason)
            )
        except Exception:
            logger.warning(
                "failed to send session-ended notice; stopping anyway",
                reason=reason,
                exc_info=True,
            )

    def _dispatch_transition(self, transition: Transition) -> None:
        """Run every side effect a session transition drives, in one place.

        Each move is journalled on the egress stream as a single transition fact
        whose detail carries whatever the move recorded — a connection id, the
        negotiation answer, an end reason, or a journal signal's payload.
        Session-boundary and per-connection moves cross into the model as
        authoritative reactor events. State-entry side effects run only when the
        state actually changes, so a self-loop — a negotiation answer or a
        journal fact riding out during teardown — can never re-run them: a move
        that changes the state re-arms the orphan timer for the state just
        entered, so a session left without a client closes itself; entering
        ``CLOSING`` clears the session's uploaded files, tears the connections
        down, and, once they have closed, unwinds the session to ready, carrying
        the end reason through; and entering ``TERMINATED`` asks the service to
        bring the process down. A crash evicts straight to ``TERMINATED``
        without passing through ``CLOSING``, so that move runs the same teardown
        on its way out — minus the ``CLEANUP_COMPLETE`` unwind, which would
        declare a dead model ready again. Real moves log at info; journal
        self-loops log at debug so a per-segment ``chunk_ready`` does not flood
        the log.

        The session boundary is where the session's recording id resolves, off
        the start parameters, so a rejected start cannot touch it; both the log's
        session context and the recorder's directory read it from there. Binding
        the log context is also part of this boundary, so every record written
        while a session is live names it, the opening move included. The release
        travels differently: it rides the ``SessionEnded``
        event into the model, whose dispatch retires the binding once the
        ``@session_ended`` hook has returned. A terminal move dispatches no
        ``SessionEnded`` and releases nothing — the process is exiting, and its
        last records belong to the session that brought it down. The log's state
        context re-stamps here too, before the move's own line, so a record
        reads the state the process was in when it was written.
        """
        if transition.is_session_start:
            self._recording_id = _recording_id_from(transition.detail.get("params", {}))
            self._log_binding = set_session_id(self._recording_id)
        if transition.from_state is not transition.to_state:
            _stamp_log_state(transition.to_state)
        log = logger.debug if transition.event in JOURNAL_EVENTS else logger.info
        # The fixed transport id (SESSION_ID) is deliberately not a field here:
        # one constant value per process carries nothing, and squatting on
        # session_id would mask the id the session is known by.
        log(
            "session transition",
            event=transition.event.name.lower(),
            from_state=transition.from_state.name.lower(),
            to_state=transition.to_state.name.lower(),
        )
        self._events.emit(TransitionEvent(transition))
        if self._bridge is not None:
            self._dispatch_reactor_events(transition, self._bridge)
        if transition.is_session_start and self._bridge is not None:
            self._start_recorder()
        entered = transition.from_state is not transition.to_state
        if entered:
            self._reset_orphan_timeout(transition.to_state)
        if entered and transition.to_state is SessionState.CLOSING and self._loop is not None:
            reason = transition.detail.get("reason", EndReason.STOPPED)
            close_reason = transition.detail.get("close_reason", "")
            # One stop, one notice: a moderation verdict outranks a close-reason
            # token, so a stop carrying both explains itself once.
            if reason is EndReason.MODERATED:
                self._broadcast_moderation_notice()
            elif close_reason:
                self._broadcast_session_ended(close_reason)
            self._uploads.clear()
            self._spawn_teardown(asyncio.to_thread(self._recorder.stop))
            self._spawn_teardown(self._close_session(reason))
        if entered and transition.to_state is SessionState.TERMINATED:
            if transition.event is SessionEvent.EVICTION and self._loop is not None:
                self._uploads.clear()
                self._spawn_teardown(asyncio.to_thread(self._recorder.stop))
                self._spawn_teardown(self._connections.close_all())
            self.request_shutdown()

    def _start_recorder(self) -> None:
        """Arm the recorder for the session, best-effort.

        Recording reads the model's output through the media fan-out
        (:meth:`_emit_media`), so starting it only opens its directory and
        workers. Recording must never break the session: a recorder that fails to
        start logs and is left disabled rather than raising into the transition
        path.
        """
        try:
            self._recorder.start(self._recording_id)
        except Exception:
            logger.exception("failed to start the recorder; continuing without recording")

    def _dispatch_reactor_events(self, transition: Transition, bridge: ModelBridge) -> None:
        """Cross session and connection facts into the model as reactor events.

        Lifecycle facts are authoritative — authored by the runtime, never the
        client — so they pass to the model unvalidated.
        """
        if transition.is_session_start:
            bridge.dispatch_reactor_event(SessionStarted(self._session_id))
        if transition.is_session_end:
            reason = transition.detail.get("reason", EndReason.STOPPED)
            bridge.dispatch_reactor_event(
                # The event carries the log binding so its dispatch — the point
                # where the @session_ended hook has provably returned — is what
                # retires it, on the model thread.
                SessionEnded(self._session_id, reason, self._log_binding)
            )
        if transition.event is SessionEvent.CONNECTION_OPENED:
            bridge.dispatch_reactor_event(
                ClientConnected(transition.detail["conn_id"], self._connections.count)
            )
        if transition.event is SessionEvent.CONNECTION_CLOSED:
            bridge.dispatch_reactor_event(
                ClientDisconnected(transition.detail["conn_id"], self._connections.count)
            )

    # -- orphan timeout + teardown --------------------------------------------

    def _reset_orphan_timeout(self, state: SessionState) -> None:
        """Re-arm the orphan timer for the state just entered.

        A session waiting for its first client or left without one (``WAITING`` /
        ``ORPHANED``) is given ``orphan_timeout`` seconds to gain a client before
        it times out and closes; entering any other state cancels the timer.
        """
        self._cancel_orphan_timeout()
        if state in (SessionState.WAITING, SessionState.ORPHANED):
            self._arm_orphan_timeout()

    def _arm_orphan_timeout(self) -> None:
        """Start the orphan timer, unless it is disabled or the loop is absent."""
        if self._loop is None or self._cfg.orphan_timeout <= 0:
            return
        self._orphan_task = self._loop.create_task(self._orphan_timeout())

    def _cancel_orphan_timeout(self) -> None:
        """Cancel a pending orphan timer, if any."""
        if self._orphan_task is not None:
            self._orphan_task.cancel()
            self._orphan_task = None

    async def _orphan_timeout(self) -> None:
        """Close a session that has stayed client-less past the orphan timeout."""
        try:
            await asyncio.sleep(self._cfg.orphan_timeout)
        except asyncio.CancelledError:
            return
        self._sm.send(SessionEvent.TIMEOUT, reason=EndReason.TIMED_OUT)

    async def _close_session(self, reason: EndReason) -> None:
        """Close the session's connections, then mark cleanup complete.

        Closing the wires and signalling ``CLEANUP_COMPLETE`` are sequenced, not
        raced: the session only unwinds to ready once its connections have
        actually closed, so the model thread :meth:`stop` releases last cannot go
        while connection-close coroutines are still in flight. A wire that fails
        to close is logged rather than left to abort the teardown, so one bad
        connection cannot strand the session in ``CLOSING``.
        """
        try:
            await self._connections.close_all()
        except Exception:
            logger.exception("error closing session connections during teardown")
        self._sm.send(SessionEvent.CLEANUP_COMPLETE, reason=reason)

    def _spawn_teardown(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run a session-teardown coroutine in the background, tracked to completion."""
        if self._loop is None:
            return
        task = self._loop.create_task(coro)
        self._teardown.add(task)
        task.add_done_callback(self._teardown.discard)

    async def _drain_teardown(self) -> None:
        """Await every outstanding teardown task, surfacing none of their errors."""
        if self._teardown:
            await asyncio.gather(*tuple(self._teardown), return_exceptions=True)
