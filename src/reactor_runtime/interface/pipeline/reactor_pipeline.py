"""Generator-driven authoring base — :class:`ReactorPipeline`.

A higher-level model base built on :class:`ReactorModel`. Instead of writing a
manual ``run()`` loop, an author implements an ``inference()`` generator and
declares a typed :class:`InputState`; the base drives the generator across
connection cycles, manages the per-connection state, and adapts the emission
rate to the model's own pace.

It owns three things on top of :class:`ReactorModel`:

- The ``run()`` driver: a fresh ``self.state`` per connection, gate on a client
  being present, advance the generator one ``yield`` at a time, emit each
  :class:`Output`, skip a turn on :data:`Idle`, and tear the session down
  cleanly when the client leaves.
- Auto-generated commands: every public :class:`InputState` field becomes a
  ``set_<field>`` command, stamped onto the subclass so the same eager
  :class:`ModelContract` that powers ``@event`` handlers discovers, validates,
  and documents it. A hand-written ``@event`` of the same name wins.
- A generator lock: ``@event`` and lifecycle handlers run only between the
  generator's ``yield`` points, so ``self.state`` is consistent within a single
  inference turn rather than mutating mid-computation.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any, ClassVar, get_type_hints

from reactor_runtime.core.model import ReactorEvent, SessionEnded, SessionStarted
from reactor_runtime.core.values import ConnId
from reactor_runtime.interface.events.decorators import EVENT_ATTR, EventHandler, make_command
from reactor_runtime.interface.internal.input_buffer import BufferClosed
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope, ReactorCore
from reactor_runtime.interface.model.reactor_model import ReactorModel
from reactor_runtime.interface.pipeline.idle import Idle
from reactor_runtime.interface.pipeline.input_state import InputState
from reactor_runtime.interface.tracks import Output
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

# A short pause on an idle turn or a generator restart yields the loop so the
# dispatch loops deliver commands, without busy-spinning.
_IDLE_SLEEP = 0.005
_RESTART_SLEEP = 0.005


class _GeneratorEnded(Exception):  # noqa: N818 — a control-flow signal, not an error a caller handles
    """Internal signal that the inference generator finished and should restart.

    ``StopIteration`` cannot cross an ``async def`` boundary (PEP 479 turns it
    into ``RuntimeError``), so a finished generator is reported with this
    instead and the driver restarts it.
    """


class ReactorPipeline(ReactorModel):
    """Generator-driven model with typed, client-mutable state.

    Subclass and provide:

    - ``state: MyState`` — a class annotation naming an :class:`InputState`
      subclass. ``self.state`` holds the live instance during a connection.
    - ``inference()`` — a generator (``def`` or ``async def``) that reads
      ``self.state``, optionally consumes input tracks, and yields an
      :class:`Output` per produced frame. Yield :data:`Idle` (or ``None``) to
      skip a turn.

    A fresh ``self.state`` is built each time generation starts and cleared when
    it stops, so a public state field is reset between runs. Public state fields
    become ``set_<field>`` commands automatically; everything
    :class:`ReactorModel` offers — ``@event``, the lifecycle hooks, ``emit``,
    ``send`` — still applies.

    Generation runs only while the session is live and a client is connected.
    Ending the session (or the last client leaving) stops the generator at the
    next turn boundary; a fresh session with a connected client starts it again.

    When the subclass declares no ``fps``, the emission rate adapts to the
    measured inference time; declaring ``fps`` pins it to a fixed rate.
    """

    __pipeline_state__: ClassVar[type[InputState]]

    state: Any
    """The live :class:`InputState` instance, or ``None`` while stopped."""

    _gen_lock: asyncio.Lock | None
    _session_active: bool
    _runnable: asyncio.Event

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Resolve the state class and stamp its auto-setters onto the subclass
        # before ReactorModel builds the contract, so they are discovered as
        # ordinary commands. An abstract intermediate without a state annotation
        # is left alone; the requirement is enforced at instantiation.
        state_cls = _resolve_state_class(cls)
        if state_cls is not None:
            cls.__pipeline_state__ = state_cls
            _stamp_auto_setters(cls, state_cls)
        super().__init_subclass__(**kwargs)

    def __init__(self) -> None:
        super().__init__()
        if getattr(type(self), "__pipeline_state__", None) is None:
            raise TypeError(
                f"{type(self).__name__} must declare 'state: MyState' where MyState "
                "is an InputState subclass."
            )
        self.state = None
        self._gen_lock = None

    # -- engine hooks ---------------------------------------------------------

    def _on_loop_ready(self) -> None:
        super()._on_loop_ready()
        self._gen_lock = asyncio.Lock()
        self._session_active = False
        self._runnable = asyncio.Event()

    # -- session-aware gating -------------------------------------------------

    async def _dispatch_reactor_event(self, event: ReactorEvent) -> None:
        """Track session liveness so the driver stops when the session ends.

        The session-boundary facts are authoritative: a session start permits
        generation, a session end forbids it. Combined with the connection
        count the base maintains, this is what gates :meth:`run` — so a
        ``stop_session`` halts the generator even though its connections are
        torn down without a per-client disconnect.
        """
        if isinstance(event, SessionStarted):
            self._session_active = True
        elif isinstance(event, SessionEnded):
            # Forbid generation before the hook runs so the driver, which checks
            # between turns without the lock, breaks at the next boundary rather
            # than waiting on the @session_ended handler to acquire it.
            self._session_active = False
            self._update_runnable()
        await super()._dispatch_reactor_event(event)
        self._update_runnable()

    def _update_runnable(self) -> None:
        """Reconcile the run gate from session liveness and the client count."""
        if self.connected.is_set() and self._session_active:
            self._runnable.set()
        else:
            self._runnable.clear()

    # -- handler serialisation ------------------------------------------------

    async def _dispatch_command(self, envelope: CommandEnvelope) -> None:
        """Dispatch a command under the generator lock, so it lands between yields."""
        lock = self._gen_lock
        if lock is None:
            await super()._dispatch_command(envelope)
            return
        async with lock:
            await super()._dispatch_command(envelope)

    async def _invoke_hook(
        self, hook: Callable[..., Any] | None, conn_id: ConnId | None, **extra: Any
    ) -> None:
        """Run a lifecycle hook under the generator lock, like a command handler."""
        lock = self._gen_lock
        if lock is None:
            await super()._invoke_hook(hook, conn_id, **extra)
            return
        async with lock:
            await super()._invoke_hook(hook, conn_id, **extra)

    # -- the driver -----------------------------------------------------------

    async def run(self) -> None:
        """Drive ``inference()`` across session cycles.

        Each run gets a fresh ``self.state`` and a new generator, and lasts while
        the session is live and a client is connected. When that gate drops — the
        session ends or the last client leaves — the generator is closed, the
        state and buffers reset, and the driver waits for the next live session.
        The inference callable is resolved from ``self.inference`` once, so
        ``load`` may replace it with an instance-bound generator; whether it is
        async is inferred from that callable.

        An uncaught exception in ``inference()`` — including a yield that is not
        an :class:`Output`, :data:`Idle`, or ``None`` — is fatal: it propagates
        out of this driver and permanently ends the model loop, not just the
        current session. The generator is still closed on the way out, so the
        teardown stays clean.
        """
        lock = self._gen_lock
        if lock is None:
            raise RuntimeError("run() started before the model loop was ready")

        inference_fn = self.inference
        is_async = inspect.isasyncgenfunction(inference_fn)
        dynamic_fps = not _fps_is_author_pinned(type(self))

        while True:
            self.state = self.__pipeline_state__()
            await self._runnable.wait()

            gen = inference_fn()
            try:
                while self._runnable.is_set():
                    try:
                        async with lock:
                            output, compute_time = await self._advance(gen, is_async)
                    except _GeneratorEnded:
                        gen = inference_fn()
                        await asyncio.sleep(_RESTART_SLEEP)
                        continue
                    except BufferClosed:
                        break

                    if output is None:
                        await asyncio.sleep(_IDLE_SLEEP)
                        continue

                    if dynamic_fps:
                        await self.emit(output, compute_time=compute_time)
                    else:
                        await self.emit(output)
            finally:
                await self._close_generator(gen, is_async)
                self.state = None
                for buffer in self._input_buffers.values():
                    buffer.reset()
                self.output_buffer.flush()

    async def _advance(self, gen: Any, is_async: bool) -> tuple[Output | None, float]:
        """Advance the generator by one yield, timing the work it took.

        Returns the yielded :class:`Output` and the seconds spent producing it,
        or ``None`` with a zero time for a skipped turn (an :data:`Idle` or
        ``None`` yield).

        Raises:
            _GeneratorEnded: The generator finished and should be restarted.
            BufferClosed: An input read found its track closed mid-session.
            TypeError: The generator yielded something other than an
                :class:`Output`, :data:`Idle`, or ``None``.
        """
        start = time.perf_counter()
        try:
            result = await gen.__anext__() if is_async else next(gen)
        except (StopIteration, StopAsyncIteration) as end:
            raise _GeneratorEnded from end
        compute_time = time.perf_counter() - start
        if result is None or result is Idle:
            return None, 0.0
        if not isinstance(result, Output):
            raise TypeError(
                f"{type(self).__name__}.inference() yielded "
                f"{type(result).__name__}; yield an Output instance or Idle."
            )
        return result, compute_time

    async def _close_generator(self, gen: Any, is_async: bool) -> None:
        """Close the generator on teardown, isolating an error in its cleanup."""
        try:
            if is_async:
                await gen.aclose()
            else:
                gen.close()
        except Exception:
            logger.exception("inference generator raised while closing")

    # -- author hook ----------------------------------------------------------

    def inference(self) -> Any:
        """Produce frames — override this as a generator.

        Implement as a sync ``def`` or ``async def`` generator. Each turn reads
        ``self.state`` for the current parameters, optionally reads input
        tracks, runs a forward pass, and yields an :class:`Output`. Yield
        :data:`Idle` or ``None`` to skip a turn. The generator starts when a
        client connects and is restarted if it finishes while the client stays.

        Example::

            def inference(self):
                while not self.state._started:
                    yield Idle
                while True:
                    frame = self.forward(prompt=self.state.prompt)
                    yield MyOutput(main_video=frame)
        """
        raise NotImplementedError(f"{type(self).__name__} must implement inference()")


def _fps_is_author_pinned(cls: type) -> bool:
    """Return whether the model (or an intermediate base) pins ``fps`` itself.

    The emission rate is adaptive unless the author declares ``fps``. The walk
    covers the model's own classes but stops at :class:`ReactorCore`, whose
    ``fps`` is the framework default rather than an author's choice — so a
    subclass that inherits a pinned rate from an intermediate
    :class:`ReactorPipeline` base counts as pinned even without redeclaring it.
    """
    for klass in cls.__mro__:
        if klass is ReactorCore:
            break
        if "fps" in vars(klass):
            return True
    return False


def _resolve_state_class(cls: type) -> type[InputState] | None:
    """Return the :class:`InputState` subclass named by the ``state`` annotation."""
    try:
        hints = get_type_hints(cls)
    except Exception:
        return None
    hint = hints.get("state")
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
