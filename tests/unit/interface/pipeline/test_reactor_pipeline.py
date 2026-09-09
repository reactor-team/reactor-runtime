import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import numpy as np
import pytest

from reactor_runtime import (
    EVENT_REGISTRY,
    Idle,
    InputField,
    InputState,
    Output,
    ReactorPipeline,
    Video,
    event,
    session_started,
)
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import ConnId
from reactor_runtime.interface.internal.input_buffer import BufferClosed
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.interface.pipeline.reactor_pipeline import _GeneratorEnded


class Frame(Output):
    main_video: Video


class State(InputState):
    speed: float = InputField(default=1.0, ge=0.0, le=10.0)
    seed: int = InputField(default=0)
    _started: bool = False


class Pipe(ReactorPipeline):
    state: State
    fps = 12

    def inference(self) -> Iterator[Frame]:
        while True:
            yield Frame(main_video=np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Pipe)


def _frame() -> Frame:
    return Frame(main_video=np.zeros((2, 2, 3), dtype=np.uint8))


def _ready(pipe: ReactorPipeline) -> None:
    """Bring the loop-bound state (including the generator lock) up."""
    pipe._on_loop_ready()
    pipe.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=lambda chunk: None
    )


# -- contract wiring ----------------------------------------------------------


def test_public_state_fields_become_commands() -> None:
    commands = ModelContract.of(Pipe).commands
    assert "set_speed" in commands
    assert "set_seed" in commands


def test_private_state_fields_are_not_exposed() -> None:
    assert "set__started" not in ModelContract.of(Pipe).commands


def test_auto_setter_command_carries_constraints() -> None:
    spec = ModelContract.of(Pipe).commands["set_speed"]
    info = spec.command.__command_fields__["speed"].info
    assert info.ge == 0.0
    assert info.le == 10.0


def test_auto_setter_is_rendered_in_the_schema() -> None:
    schema = ModelContract.of(Pipe).render_schema()
    assert "set_speed" in schema.commands
    assert "set_seed" in schema.commands


def test_auto_setters_render_though_they_never_enter_the_event_registry() -> None:
    # A pipeline stamps its set_<field> handlers directly, bypassing the @event
    # decorator that fills EVENT_REGISTRY — so the registry is not a complete
    # command set. The schema must render them from the resolved contract instead.
    class LocalState(InputState):
        gain: float = InputField(default=1.0, ge=0.0, le=2.0)

    class LocalPipe(ReactorPipeline):
        state: LocalState

        def inference(self) -> Iterator[Frame]:
            while True:
                yield _frame()

    assert "set_gain" not in EVENT_REGISTRY  # the gap this guards against
    doc = ModelContract.of(LocalPipe).render_schema().to_openapi()
    assert "/events/set_gain" in doc["paths"]


def test_a_custom_event_shadows_the_generated_setter() -> None:
    class Custom(ReactorPipeline):
        state: State

        @event(name="set_speed", description="hand written")
        def set_speed(self, speed: float = InputField(default=2.0)) -> None:
            self.state.speed = speed

        def inference(self) -> Iterator[Frame]:
            yield _frame()

    spec = ModelContract.of(Custom).commands["set_speed"]
    assert spec.description == "hand written"
    # The generated setter for the untouched field is still present.
    assert "set_seed" in ModelContract.of(Custom).commands


def test_missing_state_annotation_raises_on_instantiation() -> None:
    class NoState(ReactorPipeline):
        def inference(self) -> Iterator[Frame]:
            yield _frame()

    with pytest.raises(TypeError):
        NoState()


# -- command dispatch into state ---------------------------------------------


async def test_set_field_command_updates_the_live_state() -> None:
    pipe = Pipe()
    _ready(pipe)
    pipe.state = State()
    command = ModelContract.of(Pipe).validate("set_speed", {"speed": 3.5})
    await pipe._dispatch_command(CommandEnvelope(command, ConnId(1001), None))
    assert pipe.state.speed == 3.5


async def test_a_setter_with_no_live_state_is_a_no_op() -> None:
    pipe = Pipe()
    _ready(pipe)
    # A fresh pipeline has no live state until a connection opens.
    assert pipe.state is None
    command = ModelContract.of(Pipe).validate("set_seed", {"seed": 5})
    await pipe._dispatch_command(CommandEnvelope(command, ConnId(1001), None))  # no raise


async def test_handlers_wait_for_the_generator_lock() -> None:
    pipe = Pipe()
    _ready(pipe)
    pipe.state = State()
    command = ModelContract.of(Pipe).validate("set_speed", {"speed": 9.0})
    assert pipe._gen_lock is not None
    async with pipe._gen_lock:
        task = asyncio.create_task(
            pipe._dispatch_command(CommandEnvelope(command, ConnId(1001), None))
        )
        await asyncio.sleep(0.01)
        # The handler is parked on the lock the generator would hold.
        assert pipe.state.speed == 1.0
    await task
    assert pipe.state.speed == 9.0


# -- generator advancement ----------------------------------------------------


async def test_advance_returns_a_sync_yield() -> None:
    pipe = Pipe()
    _ready(pipe)

    def gen() -> Iterator[Frame]:
        yield _frame()

    output, compute_time = await pipe._advance(gen(), is_async=False)
    assert isinstance(output, Frame)
    assert compute_time >= 0.0


async def test_advance_returns_an_async_yield() -> None:
    pipe = Pipe()
    _ready(pipe)

    async def gen() -> AsyncIterator[Frame]:
        yield _frame()

    output, _ = await pipe._advance(gen(), is_async=True)
    assert isinstance(output, Frame)


async def test_advance_treats_idle_and_none_as_a_skip() -> None:
    pipe = Pipe()
    _ready(pipe)

    def gen() -> Iterator[object]:
        yield Idle
        yield None

    generator = gen()
    first, _ = await pipe._advance(generator, is_async=False)
    second, _ = await pipe._advance(generator, is_async=False)
    assert first is None
    assert second is None


async def test_advance_signals_generator_end() -> None:
    pipe = Pipe()
    _ready(pipe)

    def gen() -> Iterator[Frame]:
        return
        yield  # unreachable; makes this a generator

    with pytest.raises(_GeneratorEnded):
        await pipe._advance(gen(), is_async=False)


async def test_advance_rejects_a_non_output_yield() -> None:
    pipe = Pipe()
    _ready(pipe)

    def gen() -> Iterator[int]:
        yield 123

    with pytest.raises(TypeError):
        await pipe._advance(gen(), is_async=False)


# -- the run() driver ---------------------------------------------------------


class FixedRecorder(ReactorPipeline):
    state: State
    fps = 12

    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[tuple[Output, float | None]] = []

    def inference(self) -> Iterator[Frame]:
        while True:
            yield _frame()

    async def emit(
        self, output: Output, *, compute_time: float | None = None, drop: bool = False
    ) -> None:
        self.emitted.append((output, compute_time))
        if len(self.emitted) >= 3:
            self._runnable.clear()


class DynamicRecorder(ReactorPipeline):
    state: State

    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[tuple[Output, float | None]] = []

    def inference(self) -> Iterator[Frame]:
        while True:
            yield _frame()

    async def emit(
        self, output: Output, *, compute_time: float | None = None, drop: bool = False
    ) -> None:
        self.emitted.append((output, compute_time))
        if len(self.emitted) >= 3:
            self._runnable.clear()


def _open_session(pipe: ReactorPipeline) -> None:
    """Make a readied pipeline runnable, as a live session with a client would."""
    pipe._session_active = True
    pipe.connected.set()
    pipe._runnable.set()


async def _drive(pipe: ReactorPipeline) -> None:
    _ready(pipe)
    _open_session(pipe)
    task = asyncio.create_task(pipe.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_run_emits_each_yielded_output() -> None:
    pipe = FixedRecorder()
    await _drive(pipe)
    assert len(pipe.emitted) >= 3
    assert all(isinstance(output, Frame) for output, _ in pipe.emitted)


async def test_fixed_fps_emits_without_a_compute_time() -> None:
    pipe = FixedRecorder()
    await _drive(pipe)
    assert all(compute_time is None for _, compute_time in pipe.emitted)


async def test_dynamic_fps_emits_with_a_compute_time() -> None:
    pipe = DynamicRecorder()
    await _drive(pipe)
    assert pipe.emitted
    assert all(compute_time is not None for _, compute_time in pipe.emitted)


class _PinnedBase(ReactorPipeline):
    """An intermediate base that pins fps without declaring state itself."""

    fps = 7


class InheritedFpsRecorder(_PinnedBase):
    state: State

    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[tuple[Output, float | None]] = []

    def inference(self) -> Iterator[Frame]:
        while True:
            yield _frame()

    async def emit(
        self, output: Output, *, compute_time: float | None = None, drop: bool = False
    ) -> None:
        self.emitted.append((output, compute_time))
        if len(self.emitted) >= 3:
            self._runnable.clear()


async def test_fps_pinned_on_an_intermediate_base_is_treated_as_fixed() -> None:
    # The leaf does not redeclare fps, but inherits a pinned rate from its base,
    # so emission stays fixed rather than adapting to measured compute.
    pipe = InheritedFpsRecorder()
    await _drive(pipe)
    assert pipe.emitted
    assert all(compute_time is None for _, compute_time in pipe.emitted)


class FatalInferencePipe(ReactorPipeline):
    state: State
    fps = 12

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def inference(self) -> Iterator[object]:
        try:
            yield 123  # not an Output, Idle, or None
        finally:
            self.closed = True


async def test_an_inference_error_is_fatal_and_closes_the_generator() -> None:
    pipe = FatalInferencePipe()
    _ready(pipe)
    _open_session(pipe)
    # A bad yield ends the whole driver rather than just the turn, and the
    # generator is closed on the way out so teardown does not hang.
    with pytest.raises(TypeError):
        await pipe.run()
    assert pipe.closed is True


# -- teardown failures --------------------------------------------------------


class FailingCleanupPipe(ReactorPipeline):
    state: State
    fps = 12

    def inference(self) -> Iterator[Frame]:
        try:
            while True:
                yield _frame()
        finally:
            raise RuntimeError("world reset failed")


class UnwindingCleanupPipe(ReactorPipeline):
    state: State
    fps = 12

    def inference(self) -> Iterator[object]:
        try:
            yield 123  # not an Output, Idle, or None
        finally:
            raise RuntimeError("world reset failed")


class BufferClosedCleanupPipe(FailingCleanupPipe):
    def __init__(self) -> None:
        super().__init__()
        self.advances = 0

    async def _advance(self, gen: Any, is_async: bool) -> tuple[Output | None, float]:
        # Produce one frame so the generator is running, then report the buffer as
        # closed. A generator that never started skips its own cleanup on close.
        self.advances += 1
        if self.advances == 1:
            return await super()._advance(gen, is_async)
        raise BufferClosed


async def test_a_cleanup_failure_after_a_clean_session_end_ends_the_model_loop() -> None:
    pipe = FailingCleanupPipe()
    _ready(pipe)
    _open_session(pipe)
    task = asyncio.create_task(pipe.run())
    await asyncio.sleep(0.05)
    # The client leaves, so the session loop finishes without an exception of its
    # own and the cleanup failure is the only one to report.
    pipe._runnable.clear()
    with pytest.raises(RuntimeError, match="world reset failed"):
        await task
    assert pipe.state is None


async def test_a_cleanup_failure_while_unwinding_keeps_the_original_exception() -> None:
    pipe = UnwindingCleanupPipe()
    _ready(pipe)
    _open_session(pipe)
    # The bad yield is the fault worth reporting; the cleanup failure on the way
    # out must not replace it.
    with pytest.raises(TypeError):
        await pipe.run()
    assert pipe.state is None


async def test_a_cleanup_failure_after_a_closed_buffer_ends_the_model_loop() -> None:
    pipe = BufferClosedCleanupPipe()
    _ready(pipe)
    _open_session(pipe)
    # A closed input buffer breaks the loop without an exception, so it counts as
    # a clean end and the cleanup failure propagates.
    with pytest.raises(RuntimeError, match="world reset failed"):
        await pipe.run()


async def test_cancelling_the_loop_with_a_failing_cleanup_stays_cancelled() -> None:
    pipe = FailingCleanupPipe()
    _ready(pipe)
    _open_session(pipe)
    task = asyncio.create_task(pipe.run())
    await asyncio.sleep(0.05)
    # A shutdown cancels the loop. The cleanup failure is logged and dropped so a
    # graceful stop does not look like a crash.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pipe.state is None


# -- session-aware gating -----------------------------------------------------


async def test_a_started_session_with_a_client_is_runnable() -> None:
    pipe = Pipe()
    _ready(pipe)
    await pipe._dispatch_reactor_event(SessionStarted("s"))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    assert pipe._runnable.is_set()


async def test_session_ended_clears_the_run_gate() -> None:
    pipe = Pipe()
    _ready(pipe)
    await pipe._dispatch_reactor_event(SessionStarted("s"))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    await pipe._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert not pipe._runnable.is_set()


async def test_the_last_client_leaving_clears_the_run_gate() -> None:
    pipe = Pipe()
    _ready(pipe)
    await pipe._dispatch_reactor_event(SessionStarted("s"))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    await pipe._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
    assert not pipe._runnable.is_set()


async def test_state_is_built_at_session_start_and_cleared_at_session_end() -> None:
    pipe = Pipe()
    _ready(pipe)
    assert pipe.state is None
    await pipe._dispatch_reactor_event(SessionStarted("s"))
    assert isinstance(pipe.state, State)
    await pipe._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert pipe.state is None


async def test_state_survives_a_client_leaving_and_rejoining_mid_session() -> None:
    pipe = Pipe()
    _ready(pipe)
    await pipe._dispatch_reactor_event(SessionStarted("s"))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    pipe.state.speed = 7.0
    await pipe._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1002), 1))
    assert pipe.state.speed == 7.0


class _ScheduleState(InputState):
    speed: float = InputField(default=1.0)
    _schedule: Any = None


class _SessionInitPipe(ReactorPipeline):
    state: _ScheduleState
    fps = 12

    def inference(self) -> Iterator[Frame]:
        while True:
            yield _frame()

    @session_started
    async def _init_schedule(self) -> None:
        self.state._schedule = {}


async def test_session_started_hook_runs_with_the_fresh_state_in_place() -> None:
    pipe = _SessionInitPipe()
    _ready(pipe)
    await pipe._dispatch_reactor_event(SessionStarted("s"))
    assert pipe.state._schedule == {}

    # A mutation survives for the whole session and does not leak into the next.
    pipe.state._schedule[3] = "prompt"
    await pipe._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    await pipe._dispatch_reactor_event(SessionStarted("s2"))
    assert pipe.state._schedule == {}


class Streamer(ReactorPipeline):
    state: State
    fps = 12

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def inference(self) -> Iterator[Frame]:
        while True:
            yield _frame()

    async def emit(
        self, output: Output, *, compute_time: float | None = None, drop: bool = False
    ) -> None:
        self.count += 1
        await asyncio.sleep(0)  # yield so other tasks (and a stop) get scheduled


async def test_session_end_stops_the_running_driver() -> None:
    pipe = Streamer()
    _ready(pipe)
    await pipe._dispatch_reactor_event(SessionStarted("s"))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    task = asyncio.create_task(pipe.run())
    try:
        await asyncio.sleep(0.02)
        assert pipe.count > 0  # generating while the session is live
        await pipe._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
        await asyncio.sleep(0.01)
        settled = pipe.count
        await asyncio.sleep(0.03)
        # No further frames are produced once the session has ended.
        assert pipe.count == settled
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
