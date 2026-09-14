"""The step loop: the default ``run()`` of a ReactorApp.

Each test drives a small application through ``run()`` on the current event
loop, with ``emit`` overridden to record what reached the wire and at what pace.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from reactor_runtime import (
    ApplicationError,
    InputField,
    InputState,
    MediaInput,
    MessageField,
    ModelMessage,
    Output,
    ReactorApp,
    StepOutcome,
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
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope
from reactor_runtime.interface.model.contract import ModelContract


class Frame(Output):
    main_video: Video


class Camera(MediaInput):
    webcam: Video


class State(InputState):
    prompt: str = InputField(default="a forest")
    paused: bool = InputField(default=False)


class Restarted(ModelMessage):
    reason: str = MessageField(description="Why.")


def _frame() -> Frame:
    return Frame(main_video=np.zeros((2, 2, 3), dtype=np.uint8))


class Recording(ReactorApp):
    """Records every emit and every broadcast, in order, instead of sending them."""

    state: State

    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[tuple[Output, float | None]] = []
        self.wire: list[str] = []
        self.generated = 0

    async def emit(
        self, output: Output, *, compute_time: float | None = None, drop: bool = False
    ) -> None:
        self.emitted.append((output, compute_time))
        self.wire.append("media")
        await asyncio.sleep(0)


class OnlyGenerate(Recording):
    def generate(self, step: State) -> Frame:
        self.generated += 1
        return _frame()


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(OnlyGenerate)


def _ready(app: Recording) -> None:
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: app.wire.append(type(message).__name__),
        addressed=lambda *args: None,
        media=lambda chunk: None,
    )


async def _go_live(app: Recording) -> None:
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))


async def _run_for(app: Recording, seconds: float = 0.02) -> asyncio.Task[None]:
    task = asyncio.create_task(app.run())
    await asyncio.sleep(seconds)
    return task


async def _stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# -- the simplest model -------------------------------------------------------


async def test_a_model_that_writes_only_generate_runs_and_emits() -> None:
    app = OnlyGenerate()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert app.generated > 0
    assert len(app.emitted) == app.generated
    assert all(isinstance(output, Frame) for output, _ in app.emitted)
    await _stop(task)


async def test_playout_paces_from_the_measured_generate_time() -> None:
    app = OnlyGenerate()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    _, compute_time = app.emitted[0]
    assert compute_time is not None
    assert compute_time >= 0.0
    await _stop(task)


async def test_a_pinned_fps_ignores_the_measured_time() -> None:
    class Pinned(OnlyGenerate):
        fps = 12

    app = Pinned()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert app.emitted[0][1] is None
    await _stop(task)


async def test_the_default_generate_names_the_class_and_the_two_ways_out() -> None:
    class Empty(Recording):
        pass

    with pytest.raises(NotImplementedError, match="Empty must define generate"):
        Empty().generate(None)


# -- prepare_step -------------------------------------------------------------


async def test_prepare_step_receives_the_state_and_the_media_holder() -> None:
    seen: list[tuple[Any, Any]] = []

    class WithCamera(Recording):
        media: Camera

        async def prepare_step(self, state: State, media: Camera) -> State:
            seen.append((state, media))
            return state

        def generate(self, step: State) -> Frame:
            return _frame()

    app = WithCamera()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    state, media = seen[0]
    assert state is app.state
    assert media is app.media
    await _stop(task)


async def test_the_media_holder_is_none_when_no_tracks_are_declared() -> None:
    seen: list[Any] = []

    class NoCamera(Recording):
        async def prepare_step(self, state: State, media: Any) -> State:
            seen.append(media)
            return state

        def generate(self, step: State) -> Frame:
            return _frame()

    app = NoCamera()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert seen[0] is None
    await _stop(task)


async def test_a_refused_step_never_reaches_generate_and_the_loop_asks_again() -> None:
    class Gated(OnlyGenerate):
        async def prepare_step(self, state: State, media: Any) -> State:
            if state.paused:
                raise ApplicationError("paused")
            return state

    app = Gated()
    _ready(app)
    await _go_live(app)
    app.state.paused = True
    task = await _run_for(app)
    assert app.generated == 0
    assert app.emitted == []

    app.state.paused = False
    await asyncio.sleep(0.02)
    assert app.generated > 0
    await _stop(task)


async def test_prepare_step_shapes_what_generate_gets() -> None:
    inputs: list[Any] = []

    class Mapped(Recording):
        async def prepare_step(self, state: State, media: Any) -> str:
            return state.prompt.upper()

        def generate(self, step: str) -> Frame:
            inputs.append(step)
            return _frame()

    app = Mapped()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert inputs[0] == "A FOREST"
    await _stop(task)


# -- collect_step -------------------------------------------------------------


async def test_none_from_generate_runs_the_step_and_emits_nothing() -> None:
    class Quiet(Recording):
        def generate(self, step: State) -> None:
            self.generated += 1
            return

    app = Quiet()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert app.generated > 0
    assert app.emitted == []
    await _stop(task)


async def test_an_error_from_generate_ends_the_loop_by_default() -> None:
    class Broken(Recording):
        def generate(self, step: State) -> Frame:
            raise RuntimeError("cannot step from here")

    app = Broken()
    _ready(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    with pytest.raises(RuntimeError, match="cannot step from here"):
        await asyncio.wait_for(task, timeout=1.0)


async def test_a_result_that_is_not_an_output_ends_the_loop_by_name() -> None:
    class BareArray(Recording):
        def generate(self, step: State) -> np.ndarray:
            return np.zeros((2, 2, 3), dtype=np.uint8)

    app = BareArray()
    _ready(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    with pytest.raises(NotImplementedError, match="ndarray"):
        await asyncio.wait_for(task, timeout=1.0)


async def test_collect_step_recovers_from_a_model_error_and_the_loop_goes_on() -> None:
    class RolloutExhausted(Exception):  # noqa: N818 (the model's own error, named for the state)
        pass

    class Recovering(Recording):
        def __init__(self) -> None:
            super().__init__()
            self.index = 0
            self.recoveries = 0

        def generate(self, step: State) -> Frame:
            if self.index >= 3:
                raise RolloutExhausted(self.index)
            self.index += 1
            return _frame()

        async def collect_step(self, outcome: StepOutcome) -> Output | None:
            if isinstance(outcome.error, RolloutExhausted):
                self.index = 0
                self.recoveries += 1
                await self.send(Restarted(reason="window"))
                return None
            if outcome.error is not None:
                raise outcome.error
            return outcome.to_output()

    app = Recovering()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert app.recoveries > 0
    assert len(app.emitted) > 3
    await _stop(task)


async def test_a_message_sent_in_collect_step_precedes_the_step_media() -> None:
    class Announcing(Recording):
        def generate(self, step: State) -> Frame:
            return _frame()

        async def collect_step(self, outcome: StepOutcome) -> Output | None:
            await self.send(Restarted(reason="every step"))
            return outcome.to_output()

    app = Announcing()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert app.wire[:2] == ["Restarted", "media"]
    assert all(app.wire[i] == "Restarted" for i in range(0, len(app.wire) - 1, 2))
    await _stop(task)


async def test_collect_step_receives_the_elapsed_time_of_the_step() -> None:
    seen: list[StepOutcome] = []

    class Timed(Recording):
        def generate(self, step: State) -> Frame:
            return _frame()

        async def collect_step(self, outcome: StepOutcome) -> Output | None:
            seen.append(outcome)
            return outcome.to_output()

    app = Timed()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert seen[0].error is None
    assert isinstance(seen[0].result, Frame)
    assert seen[0].elapsed >= 0.0
    await _stop(task)


# -- the step lock ------------------------------------------------------------


async def test_a_handler_waits_for_the_step_lock() -> None:
    app = OnlyGenerate()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    command = ModelContract.of(OnlyGenerate).validate("set_prompt", {"prompt": "a lake"})
    async with app._step_lock:
        handler = asyncio.create_task(
            app._dispatch_command(CommandEnvelope(command, ConnId(1001), None))
        )
        await asyncio.sleep(0.01)
        assert app.state.prompt == "a forest"
    await handler
    assert app.state.prompt == "a lake"


async def test_a_lifecycle_hook_waits_for_the_step_lock() -> None:
    ran: list[str] = []

    class Hooked(OnlyGenerate):
        @session_started
        def start(self) -> None:
            ran.append("started")

    app = Hooked()
    _ready(app)
    async with app._step_lock:
        dispatch = asyncio.create_task(app._dispatch_reactor_event(SessionStarted("s")))
        await asyncio.sleep(0.01)
        assert ran == []
    await dispatch
    assert ran == ["started"]


async def test_a_handler_cannot_land_inside_a_step() -> None:
    """A handler that arrives during an await inside prepare_step runs after collect_step."""
    trace: list[str] = []
    inside_prepare = asyncio.Event()
    release_prepare = asyncio.Event()

    class Slow(Recording):
        async def prepare_step(self, state: State, media: Any) -> State:
            trace.append("prepare")
            inside_prepare.set()
            await release_prepare.wait()
            return state

        def generate(self, step: State) -> Frame:
            trace.append("generate")
            return _frame()

        async def collect_step(self, outcome: StepOutcome) -> Output | None:
            trace.append("collect")
            return outcome.to_output()

        @event(name="poke", description="A handler that records when it ran.")
        def poke(self) -> None:
            trace.append("handler")

    app = Slow()
    _ready(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    await inside_prepare.wait()  # the loop is parked at the await inside prepare_step
    command = ModelContract.of(Slow).validate("poke", {})
    handler = asyncio.create_task(
        app._dispatch_command(CommandEnvelope(command, ConnId(1001), None))
    )
    await asyncio.sleep(0.005)
    assert "handler" not in trace  # parked on the lock while the step is open
    release_prepare.set()
    await handler
    await _stop(task)

    assert trace[:3] == ["prepare", "generate", "collect"]
    assert trace.index("handler") > trace.index("collect")


# -- the live gate ------------------------------------------------------------


async def test_the_loop_waits_for_a_session_with_a_client() -> None:
    app = OnlyGenerate()
    _ready(app)
    task = await _run_for(app)
    assert app.generated == 0
    await app._dispatch_reactor_event(SessionStarted("s"))
    await asyncio.sleep(0.01)
    assert app.generated == 0
    await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    await asyncio.sleep(0.01)
    assert app.generated > 0
    await _stop(task)


async def test_the_loop_stops_when_the_last_client_leaves_and_resumes_on_rejoin() -> None:
    app = OnlyGenerate()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    await app._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
    await asyncio.sleep(0.01)
    paused_at = app.generated
    await asyncio.sleep(0.01)
    assert app.generated == paused_at

    await app._dispatch_reactor_event(ClientConnected(ConnId(1002), 1))
    await asyncio.sleep(0.01)
    assert app.generated > paused_at
    await _stop(task)


async def test_a_session_end_stops_the_loop() -> None:
    app = OnlyGenerate()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    await asyncio.sleep(0.01)
    stopped_at = app.generated
    await asyncio.sleep(0.01)
    assert app.generated == stopped_at
    assert app.state is None
    await _stop(task)


async def test_state_is_in_place_before_the_first_step() -> None:
    seen: list[Any] = []

    class Peeking(OnlyGenerate):
        async def prepare_step(self, state: State, media: Any) -> State:
            seen.append(state)
            return state

    app = Peeking()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert isinstance(seen[0], State)
    await _stop(task)


async def test_a_session_end_resets_the_input_buffers() -> None:
    class WithCamera(OnlyGenerate):
        media: Camera

    app = WithCamera()
    _ready(app)
    await _go_live(app)
    buffer = app._input_buffers["webcam"]
    buffer.close()
    assert buffer.closed
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert not buffer.closed


# -- refusal cost and pacing --------------------------------------------------


async def test_a_refused_step_waits_before_the_loop_asks_again() -> None:
    refusals = 0

    class Refusing(OnlyGenerate):
        async def prepare_step(self, state: State, media: Any) -> State:
            nonlocal refusals
            refusals += 1
            raise ApplicationError("paused")

    app = Refusing()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app, seconds=0.05)
    # Without the wait a refused turn is one call and one yield: thousands in 50 ms.
    assert 1 <= refusals <= 40
    await _stop(task)


async def test_a_refusal_is_logged_once_per_change_of_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from reactor_runtime.interface.app import reactor_app

    class Refusing(OnlyGenerate):
        async def prepare_step(self, state: State, media: Any) -> State:
            raise ApplicationError("paused" if state.paused else "no prompt")

    app = Refusing()
    _ready(app)
    await _go_live(app)
    app.state.paused = True
    with caplog.at_level(logging.DEBUG, logger=reactor_app.__name__):
        task = await _run_for(app, seconds=0.03)
        app.state.paused = False
        await asyncio.sleep(0.03)
    await _stop(task)
    reasons = [
        getattr(record, "reactor_fields", {}).get("reason")
        for record in caplog.records
        if "step refused" in record.getMessage()
    ]
    assert reasons == ["paused", "no prompt"]


async def test_playout_paces_from_the_whole_step_not_only_generate() -> None:
    class SlowPrepare(OnlyGenerate):
        async def prepare_step(self, state: State, media: Any) -> State:
            await asyncio.sleep(0.02)
            return state

    app = SlowPrepare()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app, seconds=0.05)
    _, compute_time = app.emitted[0]
    assert compute_time is not None
    assert compute_time >= 0.02
    await _stop(task)


async def test_fps_pinned_in_load_is_honoured() -> None:
    class PinsInLoad(OnlyGenerate):
        def load(self, config_path: Any) -> None:
            type(self).fps = 12

    app = PinsInLoad()
    app.load(None)
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert app.emitted[0][1] is None
    await _stop(task)
