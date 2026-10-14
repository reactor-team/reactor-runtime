"""The batched application base, :class:`BatchedReactorApp`.

A :class:`ReactorApp` serves one session. A batched app serves up to
:attr:`BatchedReactorApp.max_sessions` sessions from one process and one
model, so the model can run a single forward pass for all of them. Everything
an author writes for a :class:`ReactorApp` keeps its meaning, per session:

* ``self.state`` is the state of the session the code runs for. A command
  handler, a lifecycle hook, ``process_input()``, and ``process_output()`` each
  run for one session, and :attr:`BatchedReactorApp.session_id` names it.
* The declared :class:`MediaInput` holder reads that session's own buffers.
* ``emit()``, ``send()``, ``flush()``, and a client handle's ``send()`` reach
  that session's clients only.

The one new hook is :meth:`BatchedReactorApp.generate_batch`, the model half of
a step for every session that takes one. Each turn of the step loop calls
``process_input()`` once per live session, hands the inputs to
``generate_batch()`` in one call, and calls ``process_output()`` once per
session with that session's result.

A failure is scoped to what failed. An exception out of one session's
``process_input()`` or ``process_output()``, or out of emitting what that
session's ``process_output()`` returned, ends that session with an error and
the rest keep stepping; ``generate_batch()`` reports one session's failure by
returning the exception as that session's result, which reaches its
``process_output()`` as ``outcome.error``. An exception raised by
``generate_batch()`` itself is the model's failure, not a session's: every
session's ``process_output()`` receives it, and one that re-raises it crashes
the model loop and ends every session, as a crash of a :class:`ReactorApp` does.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    Command,
    FileUploaded,
    ReactorEvent,
    SessionEnded,
    SessionStarted,
    StartingInputApplied,
)
from reactor_runtime.core.values import CommandFailure, ConnId, InputFrame
from reactor_runtime.interface.app.input_state import InputState
from reactor_runtime.interface.app.outcome import StepOutcome
from reactor_runtime.interface.app.reactor_app import (
    _REFUSED_SLEEP,
    ReactorApp,
    _fps_is_author_pinned,
    _step_report,
)
from reactor_runtime.interface.app.step_completed import StepCompleted
from reactor_runtime.interface.client import ClientInfo
from reactor_runtime.interface.events.errors import ApplicationError
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.internal.input_buffer import InputBuffer
from reactor_runtime.interface.internal.reactor_core import (
    STEP_SESSION,
    CommandEnvelope,
    RequestId,
)
from reactor_runtime.interface.tracks import MediaInput
from reactor_runtime.log import get_logger, release_session_id, session_scope

logger = get_logger(__name__)

SessionKey = int
"""The runtime's handle for one hosted session, unique for the life of the process."""

SessionFailureSink = Callable[[SessionKey, str], None]
"""Receives a session the model could not serve, and why, on the model thread."""


class JoinPolicy(StrEnum):
    """When a session that has become live joins the batch.

    Attributes:
        ANY: At the next step.
        BOUNDARY: At the next step for which
            :meth:`BatchedReactorApp.at_step_boundary` returns true, for a model
            whose batch must stay aligned, such as one that generates in
            fixed-length blocks.
    """

    ANY = "any"
    BOUNDARY = "boundary"


@dataclass(eq=False)
class _Session:
    """The model-side record of one hosted session."""

    key: SessionKey
    session_id: str
    buffers: dict[str, InputBuffer]
    media: MediaInput | None
    state: InputState | None = None
    clients: dict[ConnId, ClientInfo] = field(default_factory=dict)
    active: bool = False
    starting_pending: bool = False
    connected: int = 0
    joined: bool = False
    failed: bool = False
    ended: bool = False
    generation: int = 0
    last_refusal: str | None = None

    @property
    def live(self) -> bool:
        """Whether the session takes steps: started, its starting input applied, watched."""
        return self.active and not self.starting_pending and self.connected > 0 and not self.failed


@dataclass(frozen=True)
class _Scoped:
    """One inbound item addressed to one session.

    It holds the session the item was posted for, resolved when it was
    posted, so an item that waits in the queue behind that session's end and
    the next session's start under the same key still reaches its own
    session, or none.
    """

    session: _Session
    item: CommandEnvelope | ReactorEvent


# The key of the one session a single-session runner hosts. A runner drives the
# model through the unaddressed entrypoints, which address this key.
_SOLE: SessionKey = 0

_CURRENT: contextvars.ContextVar[_Session | None] = contextvars.ContextVar(
    "reactor_batched_session", default=None
)


def current_session_key() -> SessionKey | None:
    """Return the key of the session the running model code serves, if any.

    Read by the runtime's outbound sinks to route what the model emits, sends,
    or reports to the session it belongs to.
    """
    session = _CURRENT.get()
    return session.key if session is not None else None


class _SessionMedia:
    """The declared :class:`MediaInput` holder, resolved to the current session's buffers."""

    def __init__(self, name: str) -> None:
        self._slot = f"_default_{name}"

    def __get__(self, obj: object, owner: type | None = None) -> Any:
        if obj is None:
            return self
        session = _CURRENT.get()
        if session is not None and session.media is not None:
            return session.media
        return obj.__dict__.get(self._slot)

    def __set__(self, obj: object, value: Any) -> None:
        obj.__dict__[self._slot] = value


class BatchedReactorApp(ReactorApp):
    """Base class for an application that serves several sessions with one model.

    Subclass it as you would :class:`ReactorApp`, declare how many sessions the
    model serves at once, and write :meth:`generate_batch` instead of
    ``generate()``::

        class Painter(BatchedReactorApp):
            state: PainterState
            max_sessions = 4

            @session_started
            def begin(self) -> None:
                self.latents[self.session_id] = self.seed_latent(self.state.seed)

            @session_ended
            def end(self) -> None:
                self.latents.pop(self.session_id, None)

            def generate_batch(self, inputs: dict[str, PainterState]) -> dict[str, PainterOutput]:
                ids = list(inputs)
                frames = self.net(torch.stack([self.latents[sid] for sid in ids]))
                return {sid: PainterOutput(main_video=frame) for sid, frame in zip(ids, frames)}

    Handlers, lifecycle hooks, ``process_input()``, and ``process_output()`` run
    for one session at a time, under the step lock, with ``self.state`` and
    :attr:`session_id` bound to that session.

    A batched app also runs where one session is hosted, as when the
    deployment caps the process at one with ``REACTOR_MAX_SESSIONS=1``: each
    step is then a batch of one.

    Class attributes:
        max_sessions: How many sessions the process hosts at once. The runtime
            reads it before the model loads, admits a session while fewer are
            running, and refuses one when all are.
        join_policy: When a session that became live joins the batch; see
            :class:`JoinPolicy`.
    """

    max_sessions: ClassVar[int] = 1
    join_policy: ClassVar[JoinPolicy] = JoinPolicy.ANY

    _sessions: dict[SessionKey, _Session]
    _roster_changed: asyncio.Event
    _on_session_failure: SessionFailureSink | None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(cls.max_sessions, int) or cls.max_sessions < 1:
            raise TypeError(f"{cls.__qualname__}.max_sessions must be an int of at least 1")
        cls.join_policy = JoinPolicy(cls.join_policy)
        holder = cls._find_holder(MediaInput)
        if holder is not None and not isinstance(cls.__dict__.get(holder[0]), _SessionMedia):
            setattr(cls, holder[0], _SessionMedia(holder[0]))

    def __init__(self) -> None:
        self._sessions = {}
        self._session_state_default: InputState | None = None
        self._clients_default: dict[ConnId, ClientInfo] = {}
        self._on_session_failure = None
        super().__init__()

    # -- what a handler sees ---------------------------------------------------

    @property
    def session_id(self) -> str | None:
        """The id of the session the running code serves, or ``None`` outside one."""
        session = _CURRENT.get()
        return session.session_id if session is not None else None

    @property
    def state(self) -> Any:
        """The live state of the session the running code serves."""
        session = _CURRENT.get()
        return session.state if session is not None else self._session_state_default

    @state.setter
    def state(self, value: Any) -> None:
        session = _CURRENT.get()
        if session is not None:
            session.state = value
        else:
            self.__dict__["_session_state_default"] = value

    @property
    def _clients(self) -> dict[ConnId, ClientInfo]:
        session = _CURRENT.get()
        return session.clients if session is not None else self._clients_default

    @_clients.setter
    def _clients(self, value: dict[ConnId, ClientInfo]) -> None:
        session = _CURRENT.get()
        if session is not None:
            session.clients = value
        else:
            self.__dict__["_clients_default"] = value

    # -- the step, batched -----------------------------------------------------

    def generate_batch(self, inputs: dict[str, Any], /) -> Mapping[str, Any]:
        """Run one step of inference for every session in the batch.

        The model half of a batched step. Synchronous; blocking GPU work is
        expected. Reads its argument and its own attributes, never
        ``self.state`` or the media tracks.

        Args:
            inputs: What :meth:`process_input` returned for each session that
                takes this step, by session id, in the order they joined.

        Returns:
            The step result for each session, by session id. A session's value
            may be an exception, which fails that session alone: its
            ``process_output()`` receives it as ``outcome.error``. A session
            missing from the mapping is failed the same way. The default calls
            ``generate()`` once per session, so a model can adopt batching one
            step at a time.
        """
        return {sid: self.generate(value) for sid, value in inputs.items()}

    def at_step_boundary(self) -> bool:
        """Return whether a session waiting to join may join at the next step.

        Read before each step under :attr:`JoinPolicy.BOUNDARY`. The default
        is always true.
        """
        return True

    async def run(self) -> None:
        """Step every live session together, one ``generate_batch()`` per turn.

        Each turn takes the step lock, calls ``process_input()`` per session,
        ``generate_batch()`` once, and ``process_output()`` per session, then
        releases the lock and emits and reports each session's step, all
        sessions at once, so one slow client holds back only the turn it is
        in. A turn in which every session refused waits a few milliseconds.

        Raises:
            Exception: The error ``generate_batch()`` raised, when a session's
                ``process_output()`` re-raised it. It ends the model loop and
                every session with it.
        """
        fps_pinned = _fps_is_author_pinned(type(self))
        while True:
            live = [session for session in self._sessions.values() if session.live]
            if not live:
                self._roster_changed.clear()
                await self._roster_changed.wait()
                continue
            if self.join_policy is JoinPolicy.ANY or self.at_step_boundary():
                for session in live:
                    session.joined = True
            batch = [session for session in live if session.joined]
            reports: list[tuple[_Session, StepCompleted, StepOutcome]] = []
            async with self._step_lock:
                inputs = await self._collect_inputs(batch)
                if inputs:
                    outcomes, batch_error = self._generate(inputs)
                    for session, outcome in outcomes:
                        returned = await self._collect_output(session, outcome, batch_error)
                        if returned is not _FAILED:
                            reports.append((session, _step_report(returned, outcome), outcome))
            if reports:
                await asyncio.gather(
                    *(
                        self._deliver(session, report, outcome, fps_pinned)
                        for session, report, outcome in reports
                    )
                )
            await asyncio.sleep(_REFUSED_SLEEP if not inputs else 0)

    async def _collect_inputs(self, batch: list[_Session]) -> dict[_Session, Any]:
        """Call ``process_input()`` for each session; leave out a refusal or a failure."""
        inputs: dict[_Session, Any] = {}
        for session in batch:
            if not session.live:
                continue
            with self._serving(session):
                try:
                    inputs[session] = await self.process_input()
                except ApplicationError as refused:
                    reason = f"{type(refused).__name__}: {refused}"
                    if reason != session.last_refusal:
                        logger.debug("step refused", reason=str(refused))
                        session.last_refusal = reason
                except Exception as error:
                    self._fail_session(session, error, "process_input")
                else:
                    session.last_refusal = None
        return inputs

    def _generate(
        self, inputs: dict[_Session, Any]
    ) -> tuple[list[tuple[_Session, StepOutcome]], Exception | None]:
        """Run ``generate_batch()`` once and split what it did into one outcome per session.

        Returns:
            Each session's outcome, and the error ``generate_batch()`` raised,
            if it raised one.
        """
        by_id = {session.session_id: value for session, value in inputs.items()}
        started = time.perf_counter()
        try:
            results = self.generate_batch(by_id)
        except Exception as error:
            elapsed = time.perf_counter() - started
            return [
                (session, StepOutcome(error=error, elapsed=elapsed)) for session in inputs
            ], error
        elapsed = time.perf_counter() - started
        outcomes: list[tuple[_Session, StepOutcome]] = []
        for session in inputs:
            if session.session_id not in results:
                missing = KeyError(f"generate_batch() returned no result for {session.session_id}")
                outcomes.append((session, StepOutcome(error=missing, elapsed=elapsed)))
                continue
            value = results[session.session_id]
            if isinstance(value, Exception):
                outcomes.append((session, StepOutcome(error=value, elapsed=elapsed)))
            else:
                outcomes.append((session, StepOutcome(result=value, elapsed=elapsed)))
        return outcomes, None

    async def _collect_output(
        self, session: _Session, outcome: StepOutcome, batch_error: Exception | None
    ) -> Any:
        """Call ``process_output()`` for one session, failing it alone on its own error.

        A re-raise of an error that ``generate_batch()`` itself raised is the
        model's failure and propagates out of the loop.
        """
        with self._serving(session):
            try:
                return await self.process_output(outcome)
            except Exception as error:
                if batch_error is not None and error is batch_error:
                    raise
                self._fail_session(session, error, "process_output")
                return _FAILED

    async def _deliver(
        self, session: _Session, report: StepCompleted, outcome: StepOutcome, fps_pinned: bool
    ) -> None:
        """Emit one session's step media and report its step, failing it alone on an error."""
        with self._serving(session):
            try:
                rate = None
                if report.output is not None:
                    pace = None if fps_pinned else outcome.elapsed
                    token = STEP_SESSION.set(session.generation)
                    try:
                        await self.emit(report.output, compute_time=pace)
                    finally:
                        STEP_SESSION.reset(token)
                    frames = self._to_bundle(report.output).frame_count
                    rate = self._playout_rate(frames, pace)
                self._report_step(report, session.generation, rate)
            except Exception as error:
                self._fail_session(session, error, "emit")

    def _fail_session(self, session: _Session, error: BaseException, where: str) -> None:
        """Stop stepping *session* and ask the runtime to end it with *error*."""
        logger.error(
            "session failed; ending it",
            where=where,
            error=f"{type(error).__name__}: {error}",
            exc_info=error,
        )
        session.failed = True
        if self._on_session_failure is not None:
            self._on_session_failure(session.key, str(error) or type(error).__name__)

    # -- binding and ingress (called by the runtime) ---------------------------

    def bind_session_failure(self, callback: SessionFailureSink) -> None:
        """Bind the sink a failed session is reported to. Called once before the loop starts."""
        self._on_session_failure = callback

    def submit_session_command(
        self,
        key: SessionKey,
        command: Command,
        conn_id: ConnId | None,
        request_id: RequestId | None,
    ) -> None:
        """Enqueue a validated command for the session *key*."""
        self._enqueue_scoped(key, CommandEnvelope(command, conn_id, request_id))

    def post_session_event(self, key: SessionKey, event: ReactorEvent) -> None:
        """Enqueue a reactor event for the session *key*.

        A :class:`SessionStarted` registers the session here, on the caller's
        thread, so media that arrives before the model loop handles the start
        lands in the session's own buffers.
        """
        if isinstance(event, SessionStarted):
            self._sessions[key] = self._new_session(key, event.session_id)
        self._enqueue_scoped(key, event)

    def push_session_media(self, key: SessionKey, track: str, frame: InputFrame) -> None:
        """Route an inbound frame into the session *key*'s buffer for *track*."""
        session = self._sessions.get(key)
        if session is None:
            return
        buffer = session.buffers.get(track)
        if buffer is not None:
            buffer.push(frame)

    def submit_command(
        self, command: Command, conn_id: ConnId | None, request_id: RequestId | None
    ) -> None:
        """Enqueue a command for the one session a single-session runner hosts."""
        self.submit_session_command(_SOLE, command, conn_id, request_id)

    def post_reactor_event(self, event: ReactorEvent) -> None:
        """Enqueue a reactor event for the one session a single-session runner hosts."""
        self.post_session_event(_SOLE, event)

    def push_media(self, track: str, frame: InputFrame) -> None:
        """Route an inbound frame to the one session a single-session runner hosts."""
        self.push_session_media(_SOLE, track, frame)

    def _enqueue_scoped(self, key: SessionKey, item: CommandEnvelope | ReactorEvent) -> None:
        """Thread-safe put of an item for the session *key* now holds onto the inbound queue."""
        queue = self._inbound_q
        session = self._sessions.get(key)
        if queue is None or session is None or self._loop.is_closed():
            return
        scoped = _Scoped(session, item)
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(queue.put_nowait, scoped)

    def _new_session(self, key: SessionKey, session_id: str) -> _Session:
        """Build the record of a session, with input buffers of its own."""
        holder = self._find_holder(MediaInput)
        buffers: dict[str, InputBuffer] = {}
        media: MediaInput | None = None
        if holder is not None:
            buffers = {name: InputBuffer() for name in holder[1].__tracks__}
            media = holder[1](**buffers)
        return _Session(key=key, session_id=session_id, buffers=buffers, media=media)

    # -- engine hooks ----------------------------------------------------------

    def _on_loop_ready(self) -> None:
        """Create the loop-bound state, and the event that wakes an idle step loop."""
        super()._on_loop_ready()
        self._roster_changed = asyncio.Event()

    def _background_coros(self) -> list[Coroutine[Any, Any, None]]:
        """Run the session-addressed drain loop alongside ``run()``."""
        return [self._scoped_inbound_loop()]

    async def _scoped_inbound_loop(self) -> None:
        """Drain the inbound queue, dispatching each item as the session it addresses."""
        queue = self._inbound_q
        assert queue is not None
        while True:
            item: Any = await queue.get()
            if not isinstance(item, _Scoped):
                continue
            session = item.session
            if session.ended:
                continue
            with self._serving(session):
                if isinstance(item.item, CommandEnvelope):
                    await self._dispatch_command(item.item)
                else:
                    await self._dispatch_session_event(session, item.item)
            self._roster_changed.set()

    async def _dispatch_session_event(self, session: _Session, event: ReactorEvent) -> None:
        """Run the lifecycle hook for one reactor event of *session* and track its liveness."""
        hooks = self.__reactor_contract__.lifecycle
        if isinstance(event, ClientConnected):
            session.clients[event.conn_id] = self._make_client(
                event.conn_id, time.monotonic(), system=event.system
            )
            session.connected = event.total
            self._sync_connected()
            await self._invoke_hook(hooks.connected, event.conn_id)
        elif isinstance(event, ClientDisconnected):
            session.connected = event.total
            self._sync_connected()
            await self._invoke_hook(hooks.disconnected, event.conn_id)
            session.clients.pop(event.conn_id, None)
        elif isinstance(event, SessionStarted):
            self._sessions_started += 1
            session.generation = self._sessions_started
            session.active = True
            session.starting_pending = event.starting_input
            async with self._step_lock:
                if self.__app_state__ is not None:
                    session.state = self.__app_state__()
                await self._call_hook(hooks.session_started, None)
        elif isinstance(event, StartingInputApplied):
            session.starting_pending = False
        elif isinstance(event, SessionEnded):
            session.active = False
            session.connected = 0
            self._sync_connected()
            async with self._step_lock:
                await self._call_hook(hooks.session_ended, None)
                session.clients.clear()
                session.state = None
            for buffer in session.buffers.values():
                buffer.reset()
            session.ended = True
            if self._sessions.get(session.key) is session:
                del self._sessions[session.key]
            release_session_id(event._log_binding)
        elif isinstance(event, FileUploaded):
            await self._invoke_hook(hooks.file_uploaded, event.conn_id, uploaded_file=event.file)

    # -- internals ---------------------------------------------------------------

    @contextlib.contextmanager
    def _serving(self, session: _Session) -> Iterator[None]:
        """Bind *session* as the one the enclosed code serves, logs included."""
        token = _CURRENT.set(session)
        # A single-session runner stamps the session's own id process-wide;
        # the key it drives carries only the transport id, so it scopes none.
        scope = (
            session_scope(session.session_id) if session.key != _SOLE else contextlib.nullcontext()
        )
        try:
            with scope:
                yield
        finally:
            _CURRENT.reset(token)

    def _sync_connected(self) -> None:
        """Hold :attr:`connected` set while any session has a client."""
        if any(session.connected > 0 for session in self._sessions.values()):
            self.connected.set()
        else:
            self.connected.clear()

    def _make_client(
        self, conn_id: ConnId, joined_at: float, *, system: bool = False
    ) -> ClientInfo:
        """Build a client handle whose ``send()`` reaches its own session, wherever it is called."""
        session = _CURRENT.get()
        if session is None:
            return super()._make_client(conn_id, joined_at, system=system)

        def send(message: ModelMessage) -> None:
            with self._serving(session):
                self._reply(conn_id, message, None)

        return ClientInfo(id=conn_id, joined_at=joined_at, system=system, _send=send)

    def _reply(
        self,
        conn_id: ConnId,
        message: ModelMessage | CommandFailure | None,
        request_id: RequestId | None,
    ) -> None:
        """Send a reply to one connection of the current session."""
        if self._out_addressed is not None:
            self._out_addressed(conn_id, message, request_id)


_FAILED = object()
"""What :meth:`BatchedReactorApp._collect_output` returns for a session it failed."""
