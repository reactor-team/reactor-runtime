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
    StepCompleted,
    StepOutcome,
    Video,
    event,
)
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    EndReason,
    SessionEnded,
    SessionStarted,
    StartingInputApplied,
)
from reactor_runtime.core.values import CompletedStep, ConnId
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
    def generate(self, input: State) -> Frame:
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


async def _go_live(app: ReactorApp) -> None:
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))


async def _run_for(app: ReactorApp, seconds: float = 0.02) -> asyncio.Task[None]:
    task = asyncio.create_task(app.run())
    await asyncio.sleep(seconds)
    return task


async def _stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# -- the simplest model -------------------------------------------------------


async def test_no_step_runs_until_the_starting_input_has_been_applied() -> None:
    app = OnlyGenerate()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s", starting_input=True))
    await app._dispatch_reactor_event(ClientConnected(ConnId(0), 1, system=True))
    task = await _run_for(app)
    # The system client is connected, but the starting commands have not landed.
    assert app.connected.is_set()
    assert app.generated == 0

    await app._dispatch_reactor_event(StartingInputApplied())
    await asyncio.sleep(0.02)
    await _stop(task)
    assert app.generated > 0


async def test_each_session_with_a_starting_input_waits_for_its_own() -> None:
    app = OnlyGenerate()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s", starting_input=True))
    await app._dispatch_reactor_event(ClientConnected(ConnId(0), 1, system=True))
    await app._dispatch_reactor_event(StartingInputApplied())
    assert app._live.is_set()
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))

    await app._dispatch_reactor_event(SessionStarted("t", starting_input=True))
    await app._dispatch_reactor_event(ClientConnected(ConnId(0), 1, system=True))

    assert not app._live.is_set()


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


# -- process_input -------------------------------------------------------------


async def test_process_input_reads_the_state_and_the_media_holder_off_self() -> None:
    seen: list[tuple[Any, Any]] = []

    class WithCamera(Recording):
        media: Camera

        async def process_input(self) -> State:
            seen.append((self.state, self.media))
            return self.state

        def generate(self, input: State) -> Frame:
            return _frame()

    app = WithCamera()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    state, media = seen[0]
    assert state is app.state
    assert media is app.media
    await _stop(task)


async def test_the_default_process_input_hands_generate_none_without_a_state() -> None:
    inputs: list[Any] = []

    class Stateless(ReactorApp):
        def generate(self, input: Any) -> Frame:
            inputs.append(input)
            return _frame()

    app = Stateless()
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=lambda chunk: None
    )
    await _go_live(app)
    task = await _run_for(app)
    assert inputs
    assert inputs[0] is None
    await _stop(task)


async def test_a_refused_step_never_reaches_generate_and_the_loop_asks_again() -> None:
    class Gated(OnlyGenerate):
        async def process_input(self) -> State:
            if self.state.paused:
                raise ApplicationError("paused")
            return self.state

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


async def test_process_input_shapes_what_generate_gets() -> None:
    inputs: list[Any] = []

    class Mapped(Recording):
        async def process_input(self) -> str:
            return self.state.prompt.upper()

        def generate(self, input: str) -> Frame:
            inputs.append(input)
            return _frame()

    app = Mapped()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert inputs[0] == "A FOREST"
    await _stop(task)


# -- process_output -------------------------------------------------------------


async def test_none_from_generate_runs_the_step_and_emits_nothing() -> None:
    class Quiet(Recording):
        def generate(self, input: State) -> None:
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
        def generate(self, input: State) -> Frame:
            raise RuntimeError("cannot step from here")

    app = Broken()
    _ready(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    with pytest.raises(RuntimeError, match="cannot step from here"):
        await asyncio.wait_for(task, timeout=1.0)


async def test_a_result_that_is_not_an_output_ends_the_loop_by_name() -> None:
    class BareArray(Recording):
        def generate(self, input: State) -> np.ndarray:
            return np.zeros((2, 2, 3), dtype=np.uint8)

    app = BareArray()
    _ready(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    with pytest.raises(NotImplementedError, match="ndarray"):
        await asyncio.wait_for(task, timeout=1.0)


async def test_process_output_recovers_from_a_model_error_and_the_loop_goes_on() -> None:
    class RolloutExhausted(Exception):  # noqa: N818 (the model's own error, named for the state)
        pass

    class Recovering(Recording):
        def __init__(self) -> None:
            super().__init__()
            self.index = 0
            self.recoveries = 0

        def generate(self, input: State) -> Frame:
            if self.index >= 3:
                raise RolloutExhausted(self.index)
            self.index += 1
            return _frame()

        async def process_output(self, outcome: StepOutcome) -> Output | None:
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


async def test_a_message_sent_in_process_output_precedes_the_step_media() -> None:
    class Announcing(Recording):
        def generate(self, input: State) -> Frame:
            return _frame()

        async def process_output(self, outcome: StepOutcome) -> Output | None:
            await self.send(Restarted(reason="every step"))
            return outcome.to_output()

    app = Announcing()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert app.wire[:2] == ["Restarted", "media"]
    assert all(app.wire[i] == "Restarted" for i in range(0, len(app.wire) - 1, 2))
    await _stop(task)


async def test_process_output_receives_the_elapsed_time_of_the_step() -> None:
    seen: list[StepOutcome] = []

    class Timed(Recording):
        def generate(self, input: State) -> Frame:
            return _frame()

        async def process_output(self, outcome: StepOutcome) -> Output | None:
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


async def test_a_handler_cannot_land_inside_a_step() -> None:
    """A handler that arrives during an await inside process_input runs after process_output."""
    trace: list[str] = []
    inside_prepare = asyncio.Event()
    release_prepare = asyncio.Event()

    class Slow(Recording):
        async def process_input(self) -> State:
            trace.append("prepare")
            inside_prepare.set()
            await release_prepare.wait()
            return self.state

        def generate(self, input: State) -> Frame:
            trace.append("generate")
            return _frame()

        async def process_output(self, outcome: StepOutcome) -> Output | None:
            trace.append("collect")
            return outcome.to_output()

        @event(name="poke", description="A handler that records when it ran.")
        def poke(self) -> None:
            trace.append("handler")

    app = Slow()
    _ready(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    await inside_prepare.wait()  # the loop is parked at the await inside process_input
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
        async def process_input(self) -> State:
            seen.append(self.state)
            return self.state

    app = Peeking()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert isinstance(seen[0], State)
    await _stop(task)


async def test_the_input_buffers_reset_when_the_gate_drops() -> None:
    class WithCamera(OnlyGenerate):
        media: Camera

    app = WithCamera()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    buffer = app._input_buffers["webcam"]
    buffer.close()
    assert buffer.closed
    # The last client leaving drops the gate; the loop's finally re-opens the track.
    await app._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
    await asyncio.sleep(0.01)
    assert not buffer.closed

    await app._dispatch_reactor_event(ClientConnected(ConnId(1002), 1))
    await asyncio.sleep(0.01)
    buffer.close()
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    await asyncio.sleep(0.01)
    assert not buffer.closed
    await _stop(task)


async def test_a_gate_drop_during_emit_still_resets_the_input_buffers() -> None:
    """The last client leaves and another joins while emit() is blocked on the wire."""

    class WithCamera(OnlyGenerate):
        media: Camera

        def __init__(self) -> None:
            super().__init__()
            self.swapped = False

        async def emit(
            self, output: Output, *, compute_time: float | None = None, drop: bool = False
        ) -> None:
            await super().emit(output, compute_time=compute_time, drop=drop)
            if not self.swapped:
                self.swapped = True
                # Both events land before the loop reads the gate again.
                await self._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
                await self._dispatch_reactor_event(ClientConnected(ConnId(1002), 1))

    app = WithCamera()
    _ready(app)
    await _go_live(app)
    buffer = app._input_buffers["webcam"]
    buffer.close()
    task = await _run_for(app)
    assert app.swapped
    assert not buffer.closed  # the boundary was seen, the finally ran
    assert app.generated > 1  # and the loop went on with the new client
    await _stop(task)


async def test_a_hand_written_run_gets_no_buffer_reset_from_the_base() -> None:
    """The reset belongs to the default loop; a 3.3.2 loop owns its buffers."""

    class OwnLoop(ReactorApp):
        media: Camera

        async def run(self) -> None:
            await asyncio.Event().wait()

    app = OwnLoop()
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=lambda chunk: None
    )
    await app._dispatch_reactor_event(SessionStarted("s"))
    buffer = app._input_buffers["webcam"]
    buffer.close()
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert buffer.closed


# -- refusal cost and pacing --------------------------------------------------


async def test_a_refused_step_waits_before_the_loop_asks_again() -> None:
    refusals = 0

    class Refusing(OnlyGenerate):
        async def process_input(self) -> State:
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
        async def process_input(self) -> State:
            raise ApplicationError("paused" if self.state.paused else "no prompt")

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


async def test_playout_paces_from_the_generate_time_not_the_whole_step() -> None:
    class SlowPrepare(OnlyGenerate):
        async def process_input(self) -> State:
            await asyncio.sleep(0.02)
            return self.state

    app = SlowPrepare()
    _ready(app)
    await _go_live(app)
    task = await _run_for(app, seconds=0.05)
    _, compute_time = app.emitted[0]
    assert compute_time is not None
    # The 20 ms spent in process_input is application time, not model time.
    assert compute_time < 0.02
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


async def test_fps_assigned_on_the_output_in_load_does_not_pin() -> None:
    # `self.output.fps` is the between-emits rate control, not the pin: the
    # step loop still paces from the measured generate() time.
    class SetsRateInLoad(OnlyGenerate):
        def load(self, config_path: Any) -> None:
            self.output.fps = 12

    app = SetsRateInLoad()
    app.load(None)
    _ready(app)
    await _go_live(app)
    task = await _run_for(app)
    assert app.emitted[0][1] is not None
    await _stop(task)


# -- step reports ---------------------------------------------------------------


def _ready_reporting(app: Recording) -> list[CompletedStep]:
    """Bring the app up with a step sink that records each report in wire order."""
    steps: list[CompletedStep] = []

    def record(step: CompletedStep) -> None:
        steps.append(step)
        app.wire.append("step")

    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: app.wire.append(type(message).__name__),
        addressed=lambda *args: None,
        media=lambda chunk: None,
        step=record,
    )
    return steps


async def test_each_step_is_reported_after_its_media() -> None:
    app = OnlyGenerate()
    steps = _ready_reporting(app)
    await _go_live(app)
    task = await _run_for(app)
    await _stop(task)

    # The cancel can land between a step's generate() and its report.
    assert app.generated - 1 <= len(steps) <= app.generated
    assert len(steps) > 0
    assert app.wire[:4] == ["media", "step", "media", "step"]
    first = steps[0]
    assert first.bundle is not None
    assert set(first.bundle.tracks) == {"main_video"}
    assert first.error is None
    assert first.files == {}
    assert first.elapsed is not None


async def test_a_refused_step_is_not_reported() -> None:
    class Gated(OnlyGenerate):
        async def process_input(self) -> State:
            if self.state.paused:
                raise ApplicationError("paused")
            return self.state

    app = Gated()
    steps = _ready_reporting(app)
    await _go_live(app)
    app.state.paused = True
    task = await _run_for(app)
    await _stop(task)

    assert steps == []


async def test_a_step_that_emits_nothing_is_reported_without_media() -> None:
    class Quiet(Recording):
        def generate(self, input: State) -> None:
            self.generated += 1

    app = Quiet()
    steps = _ready_reporting(app)
    await _go_live(app)
    task = await _run_for(app)
    await _stop(task)

    assert app.generated - 1 <= len(steps) <= app.generated
    assert len(steps) > 0
    assert all(step.bundle is None and step.error is None for step in steps)


async def test_a_step_process_output_recovered_is_reported_as_a_normal_step() -> None:
    class Flaky(Recording):
        def generate(self, input: State) -> Frame:
            self.generated += 1
            if self.generated == 1:
                raise RuntimeError("one bad step")
            return _frame()

        async def process_output(self, outcome: StepOutcome) -> Output | None:
            if outcome.error is not None:
                return None
            return outcome.to_output()

    app = Flaky()
    steps = _ready_reporting(app)
    await _go_live(app)
    task = await _run_for(app)
    await _stop(task)

    assert steps[0].bundle is None
    assert steps[0].error is None
    assert steps[1].bundle is not None


async def test_a_step_report_process_output_returns_is_emitted_and_reported_as_given() -> None:
    class KeepsFiles(OnlyGenerate):
        async def process_output(self, outcome: StepOutcome) -> StepCompleted:
            return StepCompleted(output=outcome.to_output(), files={"prompt.txt": b"a red door"})

    app = KeepsFiles()
    steps = _ready_reporting(app)
    await _go_live(app)
    task = await _run_for(app)
    await _stop(task)

    assert app.wire[:2] == ["media", "step"]
    first = steps[0]
    assert first.bundle is not None
    assert set(first.bundle.tracks) == {"main_video"}
    assert first.files == {"prompt.txt": b"a red door"}
    assert first.elapsed is not None


async def test_a_recovered_step_can_report_its_error_without_media() -> None:
    class MarksFailures(Recording):
        def generate(self, input: State) -> Frame:
            self.generated += 1
            raise RuntimeError("no reference")

        async def process_output(self, outcome: StepOutcome) -> StepCompleted:
            return StepCompleted(error=f"recovered: {outcome.error}", elapsed=1.5)

    app = MarksFailures()
    steps = _ready_reporting(app)
    await _go_live(app)
    task = await _run_for(app)
    await _stop(task)

    assert "media" not in app.wire
    assert steps[0].bundle is None
    assert steps[0].error == "recovered: no reference"
    assert steps[0].elapsed == 1.5


async def test_a_step_that_crashes_the_loop_is_not_reported() -> None:
    class Broken(Recording):
        def generate(self, input: State) -> Frame:
            raise RuntimeError("cannot step from here")

    app = Broken()
    steps = _ready_reporting(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    with pytest.raises(RuntimeError, match="cannot step from here"):
        await asyncio.wait_for(task, timeout=1.0)

    assert steps == []


async def test_a_hand_written_run_reports_its_own_steps() -> None:
    class BuildsClips(Recording):
        async def run(self) -> None:
            await self.complete_step(
                StepCompleted(
                    output=_frame(),
                    files={"last_frame.png": b"png"},
                    error=None,
                    elapsed=1.5,
                )
            )
            await self.complete_step(StepCompleted(error="RuntimeError: out of memory"))

    app = BuildsClips()
    steps = _ready_reporting(app)
    await app.run()

    assert [step.error for step in steps] == [None, "RuntimeError: out of memory"]
    assert steps[0].bundle is not None
    assert steps[0].files == {"last_frame.png": b"png"}
    assert steps[0].elapsed == 1.5
    assert steps[1].bundle is None


async def test_complete_step_without_a_sink_does_nothing() -> None:
    app = OnlyGenerate()
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=lambda chunk: None
    )
    await app.complete_step(StepCompleted(output=_frame()))


async def test_a_report_is_stamped_with_the_session_the_model_is_in() -> None:
    app = OnlyGenerate()
    steps = _ready_reporting(app)
    await app._dispatch_reactor_event(SessionStarted("s1"))
    await app.complete_step(StepCompleted())
    await app._dispatch_reactor_event(SessionEnded("s1", EndReason.STOPPED))
    await app._dispatch_reactor_event(SessionStarted("s2"))
    await app.complete_step(StepCompleted())

    assert [step.session for step in steps] == [1, 2]


async def test_a_step_that_spans_a_restart_keeps_the_session_it_began_in() -> None:
    release = asyncio.Event()

    class SlowWire(OnlyGenerate):
        async def emit(
            self, output: Output, *, compute_time: float | None = None, drop: bool = False
        ) -> None:
            await release.wait()

    app = SlowWire()
    steps = _ready_reporting(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    await asyncio.sleep(0.01)  # the first step is now blocked in emit

    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    await app._dispatch_reactor_event(SessionStarted("s2"))
    release.set()
    await asyncio.sleep(0.01)
    await _stop(task)

    assert app._sessions_started == 2
    assert steps[0].session == 1
