"""The engine-backed authoring base — :class:`EnginePipeline`.

An application binds one engine and serves it. With no overrides that is the
whole application: the engine's declarations become the command surface, the
input tracks, and the initialization, and the base drives the rollout::

    class ArcadeLingBot(EnginePipeline):
        engine = LingBotPipeline

Three layers sit above the engine's defaults, each reachable without touching
the one below. ``@override_input`` replaces one generated event. Implementing
``map_inputs`` replaces the fold from window to conditioning, leaving
``generate`` and ``finalize`` alone. Implementing ``run`` replaces the loop and
drives :meth:`step` directly.

:meth:`step` is the unit and :meth:`run` is one composition of it: a step
returns its chunk rather than emitting it, so a caller that is not a loop — a
single triggered advance, a benchmark, a replay — gets the same result the loop
would have emitted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

from reactor_runtime.core.model import ReactorEvent, SessionEnded, SessionStarted
from reactor_runtime.core.values import InputFrame
from reactor_runtime.engine_contract.inputs import Init, UserInput
from reactor_runtime.engine_contract.pipeline import Cache, StreamingPipeline, VideoChunk
from reactor_runtime.interface.engine.frames import DEFAULT_VALUE_RANGE, to_video_frames
from reactor_runtime.interface.engine.overrides import OVERRIDE_ATTR
from reactor_runtime.interface.engine.reflection import (
    VIDEO_TRACK,
    EngineInputs,
    command_for,
    default_init,
    discover_inputs,
    init_values,
    missing_init_fields,
    output_holder,
    track_holder,
    wire_name,
)
from reactor_runtime.interface.engine.store import InputStore
from reactor_runtime.interface.events.decorators import (
    EVENT_ATTR,
    EventHandler,
    handler_from_signature,
    make_command,
)
from reactor_runtime.interface.internal.reactor_core import ReactorCore
from reactor_runtime.interface.model.reactor_model import ReactorModel
from reactor_runtime.interface.tracks import Output
from reactor_runtime.interface.tracks.input import Input
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

Stepping = Literal["automatic", "triggered"]
"""How the model advances: continuously, or once per ``step`` command."""

STEPPING_MODES: frozenset[str] = frozenset({"automatic", "triggered"})
"""The values ``stepping`` accepts, from a class attribute or a deployment."""

STEP_COMMAND = "step"
"""The built-in command that advances a triggered model by one step."""

_GENERATED_PREFIX = "_engine_event_"

# A skipped step and a parked loop both yield control so the dispatch loops keep
# delivering commands, without busy-spinning.
_IDLE_SLEEP = 0.005
_PARK_SLEEP = 0.05


class InitRequiredError(Exception):
    """No rollout can exist until the client sends the initialization it owns.

    Attributes:
        fields: The initialization fields that declare no default, and are
            therefore the client's to supply.
    """

    def __init__(self, fields: list[str]) -> None:
        self.fields = fields
        super().__init__(f"init is missing required field(s) {fields}")


class EnginePipeline(ReactorModel):
    """Serves one engine, driving its rollout from the client's input window.

    Binding an engine is the whole of it: subclass and set ``engine``. The
    outbound track the engine's frames land on is declared for you, so an
    engine that says nothing about where its video goes still serves.
    Everything :class:`ReactorModel` offers — ``@event``, the lifecycle hooks,
    ``send`` — still applies, and a hand-written ``@event`` of the same wire
    name as a generated one wins.

    Class attributes:
        engine: The engine pipeline class this application binds. :meth:`load`
            instantiates it.
        stepping: ``"automatic"`` (the default) runs the loop continuously;
            ``"triggered"`` parks it and advances one step per ``step`` command.
            A deployment overrides it through ``runtime.stepping`` in
            ``reactor.yaml``.
        output_range: The value range a floating-point chunk from ``generate``
            spans, defaulting to the ``[-1, 1]`` these decoders emit. An engine
            that decodes to ``[0, 1]`` declares it; an engine that returns
            ``uint8`` is unaffected.
        fps: Pins the playback rate of an emitted chunk. Left undeclared, the
            rate follows the measured time each step took.

    Attributes:
        inputs: The session's input store. Push into it from a custom
            ``@event`` — ``self.inputs.push(Move(direction="left"))`` — and the
            input reaches the mapping in the next window, or in a later one
            with ``at_step=``.
    """

    engine: ClassVar[type]
    stepping: Stepping = "automatic"
    output_range: ClassVar[tuple[float, float]] = DEFAULT_VALUE_RANGE

    __engine_inputs__: ClassVar[EngineInputs]
    __engine_tracks__: ClassVar[type[Input] | None] = None
    __engine_output__: ClassVar[type[Output]]
    __engine_overrides__: ClassVar[dict[type[UserInput], Callable[..., Any]]] = {}

    inputs: InputStore
    _step_lock: asyncio.Lock
    _session_active: bool
    _runnable: asyncio.Event

    def __init_subclass__(cls, **kwargs: object) -> None:
        # The generated commands are stamped before ReactorModel resolves the
        # contract, so they are discovered as ordinary handler-backed commands.
        # An abstract intermediate that binds no engine is left alone; the
        # requirement is enforced at instantiation.
        engine_cls = getattr(cls, "engine", None)
        if isinstance(engine_cls, type):
            _bind_engine(cls, engine_cls)
        super().__init_subclass__(**kwargs)

    def __init__(self) -> None:
        super().__init__()
        if not isinstance(getattr(type(self), "engine", None), type):
            raise TypeError(
                f"{type(self).__name__} must bind an engine: 'engine = MyPipeline', where "
                "MyPipeline satisfies the StreamingPipeline protocol."
            )
        self._dynamic_fps = not _fps_is_author_pinned(type(self))
        self.inputs = InputStore(type(self).__engine_inputs__.media)
        self._engine: StreamingPipeline | None = None
        self._cache: Cache = None
        self._has_rollout = False
        self._index = 0

    # -- author hooks ---------------------------------------------------------

    def load(self, config_path: Path | None) -> None:
        """Instantiate the bound engine.

        An engine that needs constructor arguments — a config, a device, a set
        of weights — is built by overriding this and assigning ``self._engine``.

        Args:
            config_path: Path to the model's config file, or ``None``.
        """
        self._engine = type(self).engine()

    def map_inputs(self, autoregressive_index: int, cache: Cache, inputs: list[UserInput]) -> Any:
        """Fold one window into the next step's conditioning.

        Delegates to the engine. Implement this on the application to condition
        on something the engine did not anticipate; ``generate`` and
        ``finalize`` are untouched either way.

        Args:
            autoregressive_index: The step this conditioning is for.
            cache: The rollout's memory.
            inputs: The window, in arrival order.

        Returns:
            The engine's ``ModelInput`` for this step, or ``None`` to skip it.
        """
        assert self._engine is not None
        return self._engine.map_inputs(
            autoregressive_index=autoregressive_index, cache=cache, inputs=inputs
        )

    # -- the step -------------------------------------------------------------

    async def step(self) -> VideoChunk | None:
        """Advance the rollout by one step and return what it produced.

        Drains the window, folds it, and runs the engine's generate/finalize
        pair, opening the rollout first when there is none yet. Emission belongs
        to the caller: nothing is sent on a track from here, and the chunk comes
        back exactly as the engine decoded it.

        Returns:
            The step's decoded video, or ``None`` when the step was skipped
            because the mapping asked for it.

        Raises:
            InitRequiredError: The engine declares an initialization field with no
                default and the client has not sent one, so no rollout exists.
        """
        async with self._step_lock:
            return self._advance()

    def _advance(self) -> VideoChunk | None:
        """Run one step on the model loop. Serialised by :meth:`step`."""
        assert self._engine is not None
        window = self.inputs.drain(self._index)
        if not self._has_rollout:
            window = self._open_rollout(window)
        elif any(isinstance(item, Init) for item in window):
            # The mapping owns re-initialization — engines differ in what
            # resetting a cache means — so the runtime keeps only its own books:
            # a new sequence starts at step zero, and inputs scheduled against
            # the old sequence's steps no longer address anything.
            self.inputs.clear_deferred()
            self._index = 0
        model_input = self.map_inputs(
            autoregressive_index=self._index, cache=self._cache, inputs=window
        )
        if model_input is None:
            return None
        chunk = self._engine.generate(
            autoregressive_index=self._index, cache=self._cache, input=model_input
        )
        self._engine.finalize(autoregressive_index=self._index, cache=self._cache)
        self._index += 1
        return chunk

    def _open_rollout(self, window: list[UserInput]) -> list[UserInput]:
        """Open the rollout this window's step runs against, and return the rest of it.

        A window that begins with an ``Init`` is the client asking for its own
        initial conditions; anything else takes the initialization the engine
        declared as defaults. The leading ``Init`` is consumed here rather than
        passed on — there is no cache yet for a mapping to receive.

        Raises:
            InitRequiredError: The declared initialization has a field with no
                default and this window carries no ``init`` supplying it.
        """
        assert self._engine is not None
        init_cls = type(self).__engine_inputs__.init
        if window and isinstance(window[0], Init):
            init: Init | None = window[0]
            rest = window[1:]
        else:
            init = default_init(init_cls)
            rest = window
            if init is None and init_cls is not None:
                raise InitRequiredError(missing_init_fields(init_cls))
        self._cache = self._engine.initialize_cache(**(init_values(init) if init else {}))
        self._has_rollout = True
        self._index = 0
        return rest

    def _end_rollout(self) -> None:
        """Discard the rollout and everything queued against it."""
        self._cache = None
        self._has_rollout = False
        self._index = 0
        self.inputs.reset()

    # -- the loop -------------------------------------------------------------

    async def run(self) -> None:
        """Drive :meth:`step` for as long as the session has an audience.

        In the default automatic mode each step's frames are emitted as they are
        produced, paced by the time the step took unless ``fps`` is pinned. In
        triggered mode the loop parks and the ``step`` command advances the
        model instead. Either way the rollout is discarded when the session ends,
        and a fresh one opens for the next.
        """
        while True:
            await self._runnable.wait()
            try:
                while self._runnable.is_set():
                    if self.stepping == "triggered":
                        await asyncio.sleep(_PARK_SLEEP)
                    else:
                        await self._run_one()
            finally:
                self._end_rollout()

    async def _run_one(self) -> None:
        """Take one automatic turn: step, then emit whatever it produced."""
        started = time.perf_counter()
        try:
            chunk = await self.step()
        except InitRequiredError:
            # A required initialization field is the client's to send. Wait for
            # it rather than failing the session.
            await asyncio.sleep(_IDLE_SLEEP)
            return
        if chunk is None:
            await asyncio.sleep(_IDLE_SLEEP)
            return
        await self.emit_chunk(chunk, time.perf_counter() - started)

    async def emit_chunk(self, chunk: VideoChunk, compute_time: float | None = None) -> None:
        """Send one step's decoded video on the outbound track.

        Normalizes whatever the engine returned — device tensor, channels
        first, half precision, its own value range — into the frames a
        transport carries.

        Args:
            chunk: What a step produced.
            compute_time: Seconds the step took, used to pace playback when the
                model does not pin ``fps``.
        """
        frames = to_video_frames(chunk, type(self).output_range)
        output = type(self).__engine_output__(**{VIDEO_TRACK: frames})
        if self._dynamic_fps and compute_time is not None:
            await self.emit(output, compute_time=compute_time)
        else:
            await self.emit(output)

    async def _triggered_step(self) -> None:
        """Advance one step on request — what the ``step`` command runs."""
        if self.stepping != "triggered":
            logger.warning("step command ignored: this model drives its own loop")
            return
        started = time.perf_counter()
        try:
            chunk = await self.step()
        except InitRequiredError as required:
            logger.warning("step before initialization", missing=required.fields)
            return
        if chunk is not None:
            await self.emit_chunk(chunk, time.perf_counter() - started)

    # -- engine hooks ---------------------------------------------------------

    def _on_loop_ready(self) -> None:
        super()._on_loop_ready()
        self._step_lock = asyncio.Lock()
        self._session_active = False
        self._runnable = asyncio.Event()

    def push_media(self, track: str, frame: InputFrame) -> None:
        """Route an inbound frame into the window rather than a track buffer.

        Media reaches an engine through the same ordered window its events do,
        so the store accumulates frames and materializes them against the
        declared ``MediaInput`` when the window is assembled.
        """
        self.inputs.push_frame(track, frame)

    async def _dispatch_reactor_event(self, event: ReactorEvent) -> None:
        """Gate the loop on the session having started and a client being present."""
        if isinstance(event, SessionStarted):
            self._session_active = True
        elif isinstance(event, SessionEnded):
            # Forbid stepping before the hook runs, so the loop breaks at the
            # next turn rather than after the handler has finished. The rollout
            # goes with the session here rather than in the loop's teardown,
            # because a triggered model's loop is parked and never runs one.
            self._session_active = False
            self._update_runnable()
            self._end_rollout()
        await super()._dispatch_reactor_event(event)
        self._update_runnable()

    def _update_runnable(self) -> None:
        """Reconcile the run gate from session liveness and the client count."""
        if self.connected.is_set() and self._session_active:
            self._runnable.set()
        else:
            self._runnable.clear()


def _bind_engine(cls: type[EnginePipeline], engine_cls: type) -> None:
    """Stamp an engine's serving surface onto the application class."""
    inputs = discover_inputs(engine_cls)
    cls.__engine_inputs__ = inputs
    cls.__engine_tracks__ = track_holder(f"{engine_cls.__name__}Input", inputs.media)
    cls.__engine_output__ = output_holder(f"{engine_cls.__name__}Output")

    overrides = _collect_overrides(cls)
    cls.__engine_overrides__ = overrides
    claimed = _existing_command_names(cls)

    declared: dict[str, type[UserInput]] = dict(inputs.events)
    if inputs.init is not None:
        declared[wire_name(inputs.init)] = inputs.init

    for name, input_cls in declared.items():
        override = overrides.get(input_cls)
        if override is not None:
            _stamp_override(cls, name, override)
        elif name not in claimed:
            _stamp(
                cls,
                _push_handler(input_cls),
                EventHandler(
                    name=name,
                    description=inspect.cleandoc(input_cls.__doc__ or ""),
                    command=command_for(name, input_cls),
                    is_async=False,
                    reserved=(),
                ),
            )

    if STEP_COMMAND not in claimed:
        _stamp(
            cls,
            _step_handler(),
            EventHandler(
                name=STEP_COMMAND,
                description="Advance the model by one step and deliver that step's frames.",
                command=make_command(STEP_COMMAND, []),
                is_async=True,
                reserved=(),
            ),
        )


def _stamp(cls: type, handler: Callable[..., Any], declared: EventHandler) -> None:
    """Bind a generated handler to the class under a name of the runtime's own.

    A generated handler is reached only through the contract, never by an
    author, so it is bound under a prefixed attribute. That keeps a command
    whose wire name happens to match part of the authoring surface — ``step``
    most obviously — from shadowing the method it is named after.
    """
    setattr(handler, EVENT_ATTR, declared)
    setattr(cls, f"{_GENERATED_PREFIX}{declared.name}", handler)


def _stamp_override(cls: type[EnginePipeline], name: str, method: Callable[..., Any]) -> None:
    """Bind an overriding handler, whose return value is what reaches the window.

    The command is read off the author's method, so the payload is its signature
    and the schema treats it like any other declared event. What runs is a
    wrapper that calls the method and queues whatever it hands back, leaving the
    author's own method callable and unchanged.
    """
    declared = handler_from_signature(method, name)
    is_coroutine = declared.is_async

    async def handler(self: EnginePipeline, **kwargs: Any) -> None:
        result = method(self, **kwargs)
        _queue(self, await result if is_coroutine else result)

    _stamp(cls, handler, dataclasses.replace(declared, is_async=True))


def _queue(pipeline: EnginePipeline, result: Any) -> None:
    """Queue an overriding handler's return value, dropping a ``None``."""
    if isinstance(result, UserInput):
        pipeline.inputs.push(result)


def _push_handler(input_cls: type[UserInput]) -> Callable[..., None]:
    """Build the handler that turns a validated payload into a queued input."""

    def handler(self: EnginePipeline, **kwargs: Any) -> None:
        self.inputs.push(input_cls(**kwargs))

    return handler


def _step_handler() -> Callable[..., Any]:
    """Build this class's own ``step`` handler, so the stamp is not shared."""

    async def handler(self: EnginePipeline) -> None:
        await self._triggered_step()

    return handler


def _collect_overrides(cls: type) -> dict[type[UserInput], Callable[..., Any]]:
    """Find the handlers marked with ``@override_input``, most derived winning.

    A base's overrides are inherited: binding an engine replaces the marked
    method on the class with its wrapper, so the mapping the base resolved is
    what a subclass reads rather than the marker, which is gone.
    """
    found: dict[type[UserInput], Callable[..., Any]] = {}
    for klass in cls.__mro__:
        for attr in vars(klass).values():
            target = getattr(attr, OVERRIDE_ATTR, None)
            if isinstance(target, type) and issubclass(target, UserInput):
                found.setdefault(target, attr)
    for target, method in getattr(cls, "__engine_overrides__", {}).items():
        found.setdefault(target, method)
    return found


def _existing_command_names(cls: type) -> set[str]:
    """Collect the command names already claimed by hand-written ``@event`` handlers."""
    names: set[str] = set()
    for klass in cls.__mro__:
        for attr in vars(klass).values():
            handler = getattr(attr, EVENT_ATTR, None)
            if isinstance(handler, EventHandler):
                names.add(handler.name)
    return names


def _fps_is_author_pinned(cls: type) -> bool:
    """Return whether the model, or an intermediate base, pins ``fps`` itself.

    The walk stops at :class:`ReactorCore`, whose ``fps`` is the framework
    default rather than an author's choice.
    """
    for klass in cls.__mro__:
        if klass is ReactorCore:
            return False
        if "fps" in vars(klass):
            return True
    return False
