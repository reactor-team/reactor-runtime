import asyncio
import contextlib
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from reactor_runtime import (
    InputField,
    InputState,
    ModelMessage,
    NotReady,
    Output,
    ReactorModel,
    StepStats,
    Video,
)
from reactor_runtime.core import MediaChunk
from reactor_runtime.interface.internal.input_buffer import BufferClosed
from reactor_runtime.interface.internal.step_driver import StepDriver


class Frame(Output):
    main_video: Video


class Note(ModelMessage):
    text: str


class State(InputState):
    prompt: str = InputField(default="hi")
    _scratch: int = 0


def _frame() -> np.ndarray:
    return np.zeros((2, 2, 3), dtype=np.uint8)


class Gen:
    """A model that hands back whatever the test told it to."""

    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **inputs: Any) -> Any:
        self.calls.append(inputs)
        if not self.results:
            return _frame()
        return self.results.pop(0)


class Stepper(ReactorModel):
    """The plain shape: state fields in, one frame out."""

    state: State
    model: Gen


@pytest.fixture(autouse=True)
def _seed_registries(isolate_interface_registries: None, register: Callable[..., None]) -> None:
    register(Frame, Note)


def _ready(app: ReactorModel) -> list[MediaChunk]:
    """Bring the loop-bound state up and capture every emitted chunk.

    The gate is set because the driver only steps for a connected client.
    """
    chunks: list[MediaChunk] = []
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None,
        addressed=lambda conn, message, request: None,
        media=chunks.append,
    )
    app.connected.set()
    return chunks


def _build(cls: type[ReactorModel], *results: Any) -> tuple[Any, list[MediaChunk]]:
    app = cls()
    app.model = Gen(*results)
    chunks = _ready(app)
    app.state = State() if getattr(type(app), "__state_type__", None) is not None else None
    return app, chunks


# -- binding ------------------------------------------------------------------


def test_the_driver_needs_a_model_to_drive() -> None:
    class NoModel(ReactorModel):
        state: State

    app = NoModel()
    _ready(app)
    with pytest.raises(TypeError, match="no model to drive"):
        StepDriver(app)


# -- one step -----------------------------------------------------------------


async def test_a_step_maps_generates_and_emits() -> None:
    app, chunks = _build(Stepper)
    await StepDriver(app).advance()
    # The default mapping passes every public state field, by name.
    assert app.model.calls == [{"prompt": "hi"}]
    assert len(chunks) == 1


async def test_a_declined_mapping_emits_nothing() -> None:
    class Declines(Stepper):
        def map_step(self, state: Any, input: Any) -> dict[str, Any]:
            raise NotReady("waiting for frames")

    app, chunks = _build(Declines)
    with pytest.raises(NotReady, match="waiting for frames"):
        await StepDriver(app).advance()
    assert chunks == []
    assert app.model.calls == []


async def test_a_model_producing_nothing_declines_the_step() -> None:
    app, chunks = _build(Stepper, None)
    with pytest.raises(NotReady, match="produced nothing"):
        await StepDriver(app).advance()
    assert chunks == []


# -- a session the step outlives ----------------------------------------------


class Counts(Stepper):
    """A model that keeps its own tally of the steps that reached the wire."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = 0

    def on_step(self, stats: StepStats) -> None:
        self.steps += 1
        # A model is entitled to write to its state here; a session that ended
        # has none, and this must never be reached in that case.
        self.state._scratch += 1


async def test_no_step_is_taken_without_a_client() -> None:
    app, chunks = _build(Counts)
    app.connected.clear()
    with pytest.raises(NotReady, match="no client is connected"):
        await StepDriver(app).advance()
    assert app.model.calls == []
    assert chunks == []
    assert app.steps == 0


async def test_a_step_the_session_outlives_is_abandoned() -> None:
    """A session ending while a chunk goes out must not reach the model again.

    The gate is cleared from inside the media sink, which is where a real session
    end lands: the emit is holding for downstream room when the session goes.
    """
    app = Counts()
    app.model = Gen()
    chunks: list[MediaChunk] = []
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None,
        addressed=lambda conn, message, request: None,
        media=lambda chunk: (chunks.append(chunk), app.connected.clear())[0],
    )
    app.connected.set()
    app.state = State()

    await StepDriver(app).advance()

    # The chunk went out, and nothing was asked of the model afterwards.
    assert len(chunks) == 1
    assert app.steps == 0
    assert app.state._scratch == 0


async def test_a_step_the_session_survives_reaches_on_step() -> None:
    app, chunks = _build(Counts)
    await StepDriver(app).advance()
    assert len(chunks) == 1
    assert app.steps == 1
    assert app.state._scratch == 1


# -- how products reach to_output ---------------------------------------------


async def test_a_mapping_spreads_into_keyword_arguments() -> None:
    class Spread(Stepper):
        def to_output(self, produced: Any = None, **products: Any) -> Frame:
            assert produced is None
            assert set(products) == {"frames", "depth"}
            return Frame(main_video=products["frames"])

    app, chunks = _build(Spread, {"frames": _frame(), "depth": _frame()})
    await StepDriver(app).advance()
    assert len(chunks) == 1


async def test_anything_else_arrives_positionally() -> None:
    seen: list[Any] = []

    class Positional(Stepper):
        def to_output(self, produced: Any = None, **products: Any) -> Frame:
            seen.append(produced)
            return Frame(main_video=produced)

    frame = _frame()
    app, _ = _build(Positional, frame)
    await StepDriver(app).advance()
    assert seen[0] is frame


async def test_the_default_places_one_product_on_the_single_declared_track() -> None:
    app, chunks = _build(Stepper)
    await StepDriver(app).advance()
    assert set(chunks[0].bundle.tracks) == {"main_video"}


async def test_the_default_cannot_choose_between_two_output_classes() -> None:
    class Second(Output):
        other_video: Video

    app, _ = _build(Stepper)
    with pytest.raises(TypeError, match="several Output classes"):
        await StepDriver(app).advance()
    assert Second.__tracks__  # declared for the registry, used by the error above


# -- messages as a step's product ---------------------------------------------


class Policy(ReactorModel):
    """A model whose product is a message, with no media track at all."""

    state: State
    model: Gen

    def to_output(self, produced: Any = None, **products: Any) -> Note:
        return Note(text=str(produced))


async def test_a_message_only_step_sends_and_does_not_emit() -> None:
    sent: list[Note] = []
    app, chunks = _build(Policy, "act")
    app._out_broadcast = lambda message: sent.append(cast(Note, message))
    await StepDriver(app).advance()
    assert [message.text for message in sent] == ["act"]
    assert chunks == []


async def test_a_message_is_sent_before_the_media_is_emitted() -> None:
    order: list[str] = []

    class Both(Stepper):
        def to_output(self, produced: Any = None, **products: Any) -> Any:
            # Returned output-first on purpose: the driver still sends first.
            return (Frame(main_video=produced), Note(text="tick"))

    app, _ = _build(Both)
    app._out_broadcast = lambda message: order.append("send")
    app._out_media = lambda chunk: order.append("emit")
    await StepDriver(app).advance()
    assert order == ["send", "emit"]


async def test_publishing_nothing_is_allowed() -> None:
    class Quiet(Stepper):
        def to_output(self, produced: Any = None, **products: Any) -> None:
            return None

    app, chunks = _build(Quiet)
    await StepDriver(app).advance()
    assert chunks == []


async def test_a_step_emits_at_most_one_output() -> None:
    class Twice(Stepper):
        def to_output(self, produced: Any = None, **products: Any) -> Any:
            return [Frame(main_video=produced), Frame(main_video=produced)]

    app, _ = _build(Twice)
    with pytest.raises(TypeError, match="one step emits at most one"):
        await StepDriver(app).advance()


async def test_an_unpublishable_return_is_rejected() -> None:
    class Wrong(Stepper):
        def to_output(self, produced: Any = None, **products: Any) -> Any:
            return 123

    app, _ = _build(Wrong)
    with pytest.raises(TypeError, match="return an Output"):
        await StepDriver(app).advance()


# -- stats --------------------------------------------------------------------


class WithStats(Stepper):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[StepStats] = []
        self.stepped: list[StepStats] = []

    def to_output(self, produced: Any = None, stats: StepStats | None = None, **rest: Any) -> Frame:
        assert stats is not None
        self.seen.append(stats)
        return Frame(main_video=produced)

    def on_step(self, stats: StepStats) -> None:
        self.stepped.append(stats)


async def test_stats_are_injected_when_the_signature_asks_for_them() -> None:
    app, _ = _build(WithStats)
    driver = StepDriver(app)
    await driver.advance()
    await driver.advance()
    assert [stats.step for stats in app.seen] == [0, 1]
    assert all(stats.compute_time >= 0.0 for stats in app.seen)


async def test_on_step_always_receives_the_stats() -> None:
    app, _ = _build(WithStats)
    await StepDriver(app).advance()
    assert [stats.step for stats in app.stepped] == [0]


async def test_a_declined_step_does_not_advance_the_step_index() -> None:
    app, _ = _build(WithStats, None, _frame())
    driver = StepDriver(app)
    with pytest.raises(NotReady):
        await driver.advance()
    await driver.advance()
    assert [stats.step for stats in app.seen] == [0]


async def test_a_to_output_without_stats_is_called_without_them() -> None:
    app, chunks = _build(Stepper)
    await StepDriver(app).advance()  # the default to_output declares no stats
    assert len(chunks) == 1


# -- state changes ------------------------------------------------------------


class Mirror(Stepper):
    """Sends a message whenever the state changes."""

    async def on_state_changed(self, state: Any) -> None:
        await self.send(Note(text=state.prompt))


def _sent(app: ReactorModel) -> list[Note]:
    """Capture broadcast messages, replacing the bound sink."""
    messages: list[Note] = []
    app._out_broadcast = lambda message: messages.append(cast(Note, message))
    return messages


async def test_the_first_look_at_the_state_is_a_baseline() -> None:
    app, _ = _build(Mirror)
    messages = _sent(app)
    await StepDriver(app).advance()
    # A client that just connected was told the state already.
    assert messages == []


async def test_a_command_between_steps_is_reported_before_the_next_step() -> None:
    app, _ = _build(Mirror)
    messages = _sent(app)
    driver = StepDriver(app)
    await driver.advance()
    app.state.prompt = "a forest"  # as a set_prompt command would
    await driver.advance()
    assert [message.text for message in messages] == ["a forest"]


async def test_an_unchanged_state_reports_nothing() -> None:
    app, _ = _build(Mirror)
    messages = _sent(app)
    driver = StepDriver(app)
    await driver.advance()
    app.state.prompt = "x"
    await driver.advance()
    await driver.advance()
    await driver.advance()
    assert len(messages) == 1


async def test_a_state_written_during_the_step_is_reported_after_it() -> None:
    class WritesDuringStep(Mirror):
        def to_output(self, produced: Any = None, **products: Any) -> Frame:
            self.state.prompt = "written by the step"
            return Frame(main_video=produced)

    app, _ = _build(WritesDuringStep)
    messages = _sent(app)
    await StepDriver(app).advance()
    assert [message.text for message in messages] == ["written by the step"]


async def test_replacing_the_state_wholesale_is_reported() -> None:
    app, _ = _build(Mirror)
    messages = _sent(app)
    driver = StepDriver(app)
    app.state.prompt = "before"
    await driver.advance()  # baseline: the prompt is "before"
    app.state = State()  # a factory reset, as a command would do
    await driver.advance()
    # The report carries the field defaults the fresh state came with.
    assert [message.text for message in messages] == ["hi"]


async def test_a_declined_step_still_reports_the_change() -> None:
    class Declines(Mirror):
        def map_step(self, state: Any, input: Any) -> dict[str, Any]:
            raise NotReady("paused")

    app, _ = _build(Declines)
    messages = _sent(app)
    driver = StepDriver(app)
    with pytest.raises(NotReady):
        await driver.advance()
    app.state.prompt = "set while declining"
    with pytest.raises(NotReady):
        await driver.advance()
    assert [message.text for message in messages] == ["set while declining"]


async def test_a_field_holding_an_array_does_not_break_the_comparison() -> None:
    class ArrayState(InputState):
        prompt: str = InputField(default="")
        _frames: Any = None

    class Arrays(ReactorModel):
        state: ArrayState
        model: Gen

        async def on_state_changed(self, state: Any) -> None:
            await self.send(Note(text=state.prompt))

    app = Arrays()
    app.model = Gen()
    _ready(app)
    app.state = ArrayState()
    messages = _sent(app)
    driver = StepDriver(app)

    app.state._frames = np.zeros((4, 4, 3), dtype=np.uint8)
    await driver.advance()  # baseline, with an array in a field
    # The same array, untouched: identity settles it without comparing values.
    await driver.advance()
    app.state.prompt = "changed"
    await driver.advance()
    assert [message.text for message in messages] == ["changed"]


async def test_the_default_hook_says_nothing() -> None:
    app, _ = _build(Stepper)  # the default is a no-op
    messages = _sent(app)
    driver = StepDriver(app)
    await driver.advance()
    app.state.prompt = "x"
    await driver.advance()
    assert messages == []


async def test_the_hook_sees_the_state_as_it_now_stands() -> None:
    seen: list[str] = []

    class Watcher(Stepper):
        async def on_state_changed(self, state: Any) -> None:
            seen.append(state.prompt)

    app, _ = _build(Watcher)
    driver = StepDriver(app)
    await driver.advance()
    app.state.prompt = "a forest"
    await driver.advance()
    assert seen == ["a forest"]


# -- pacing -------------------------------------------------------------------


async def test_playout_is_paced_from_the_measured_step_by_default() -> None:
    class Slow(Stepper):
        pass

    app, chunks = _build(Slow)
    await StepDriver(app).advance()
    # One frame produced in a measured interval: the rate is frames over seconds,
    # never the class default of 30.
    assert chunks[0].fps != 30.0


async def test_a_pinned_class_fps_is_used_as_declared() -> None:
    class Pinned(Stepper):
        fps = 12

    app, chunks = _build(Pinned)
    await StepDriver(app).advance()
    assert chunks[0].fps == 12.0


async def test_a_rate_on_the_output_wins() -> None:
    class Declared(Stepper):
        fps = 12

        def to_output(self, produced: Any = None, **products: Any) -> Frame:
            return Frame(main_video=produced, fps=24)

    app, chunks = _build(Declared)
    await StepDriver(app).advance()
    assert chunks[0].fps == 24.0


# -- the loop -----------------------------------------------------------------


async def _drive(app: ReactorModel, seconds: float = 0.05) -> None:
    task = asyncio.create_task(StepDriver(app).run())
    await asyncio.sleep(seconds)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_the_loop_parks_until_a_client_is_connected() -> None:
    app, chunks = _build(Stepper)
    app.connected.clear()
    await _drive(app)
    assert chunks == []

    app.connected.set()
    await _drive(app)
    assert chunks


async def test_a_declined_step_holds_the_stream_and_the_loop_continues() -> None:
    class Alternates(Stepper):
        def __init__(self) -> None:
            super().__init__()
            self.turns = 0

        def map_step(self, state: Any, input: Any) -> dict[str, Any]:
            self.turns += 1
            if self.turns % 2:
                raise NotReady("not yet")
            return {}

    app, chunks = _build(Alternates)
    app.connected.set()
    await _drive(app)
    assert app.turns > 2
    assert chunks


async def test_a_closed_track_ends_the_cycle_rather_than_the_loop() -> None:
    class Closes(Stepper):
        def __init__(self) -> None:
            super().__init__()
            self.turns = 0

        def map_step(self, state: Any, input: Any) -> dict[str, Any]:
            self.turns += 1
            if self.turns > 2:
                raise BufferClosed("track closed")
            return {}

    app, chunks = _build(Closes)
    app.connected.set()
    driver = StepDriver(app)
    task = asyncio.create_task(driver.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    # Two frames went out and the loop survived to wait for the next cycle.
    assert len(chunks) == 2
    assert task.cancelled()


async def test_the_step_tally_counts_on_across_cycles() -> None:
    app, _ = _build(WithStats)
    driver = StepDriver(app)
    await driver.advance()

    # A cycle ends — a client leaves, a track closes — and the next one picks up
    # the tally where it left off, because it counts what the runtime drove.
    for buffer in app._input_buffers.values():
        buffer.reset()
    await driver.advance()

    assert [stats.step for stats in app.seen] == [0, 1]
