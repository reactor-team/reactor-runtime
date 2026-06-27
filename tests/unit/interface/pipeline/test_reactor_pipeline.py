import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest

from reactor_runtime import (
    Idle,
    InputField,
    InputState,
    Output,
    ReactorPipeline,
    Video,
    event,
)
from reactor_runtime.core.values import ConnId
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
    output: Frame
    fps = 12

    def inference(self) -> Iterator[Frame]:
        while True:
            yield Frame(main_video=np.zeros((2, 2, 3), dtype=np.uint8))


def _frame() -> Frame:
    return Frame(main_video=np.zeros((2, 2, 3), dtype=np.uint8))


def _ready(pipe: ReactorPipeline) -> None:
    """Bring the loop-bound state (including the generator lock) up."""
    pipe._on_loop_ready()
    pipe.bind_output(broadcast=lambda message: None, addressed=lambda *args: None)


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


def test_a_custom_event_shadows_the_generated_setter() -> None:
    class Custom(ReactorPipeline):
        state: State
        output: Frame

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
        output: Frame

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
    output: Frame
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
            self.connected.clear()


class DynamicRecorder(ReactorPipeline):
    state: State
    output: Frame

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
            self.connected.clear()


async def _drive(pipe: ReactorPipeline) -> None:
    _ready(pipe)
    pipe.connected.set()
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
