"""A stepped session: the step loop runs one step per client request.

Each test drives a small application through ``run()`` and its inbound queue on
the current event loop, with the outbound sinks recording what the runtime
would have sent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from reactor_runtime import (
    ApplicationError,
    CommandError,
    InputField,
    InputState,
    MediaInput,
    Output,
    ReactorApp,
    ReadMode,
    Video,
    event,
)
from reactor_runtime.core.model import (
    ClientConnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import CompletedStep, ConnId, InputFrame, StepRefusal
from reactor_runtime.interface.internal.input_buffer import InputBuffer
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope, StepEnvelope
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.interface.pipeline.reactor_pipeline import ReactorPipeline

CLIENT = ConnId(1002)


class Frame(Output):
    main_video: Video


class Camera(MediaInput):
    webcam: Video


class State(InputState):
    prompt: str = InputField(default="a forest")


class Stepper(ReactorApp):
    state: State
    camera: Camera

    def __init__(self) -> None:
        super().__init__()
        self.notes: list[str] = []
        self.inputs: list[tuple[str, list[str], int]] = []
        self.emitted = 0

    @event(name="note")
    def note(self, text: str) -> None:
        self.notes.append(text)

    @event(name="reject")
    def reject(self) -> None:
        raise CommandError("rejected", "not this one")

    async def process_input(self) -> tuple[str, list[str], int]:
        if self.state.prompt == "refuse":
            raise ApplicationError("not now")
        frames = webcam(self).try_read(1, mode=ReadMode.FIFO) or []
        return self.state.prompt, list(self.notes), len(frames)

    def generate(self, input: tuple[str, list[str], int]) -> Frame:
        self.inputs.append(input)
        return Frame(main_video=np.zeros((2, 2, 3), dtype=np.uint8))

    async def emit(
        self, output: Output, *, compute_time: float | None = None, drop: bool = False
    ) -> None:
        self.emitted += 1


def webcam(app: Stepper) -> InputBuffer:
    return cast(InputBuffer, app.camera.webcam)


class Failing(Stepper):
    def generate(self, input: Any) -> Frame:
        raise RuntimeError("model blew up")


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Stepper)


class Harness:
    """An app with its sinks recorded, its inbound queue drained, and its loop running."""

    def __init__(self, app: Stepper) -> None:
        self.app = app
        self.refusals: list[StepRefusal] = []
        self.reports: list[CompletedStep] = []
        self.addressed: list[tuple[Any, ...]] = []
        app._on_loop_ready()
        app.bind_output(
            broadcast=lambda message: None,
            addressed=lambda *args: self.addressed.append(args),
            media=lambda chunk: None,
            step=self.reports.append,
            step_refusal=self.refusals.append,
        )
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        app._inbound_q = self.queue
        self.tasks: list[asyncio.Task[None]] = []

    async def start(self, *, stepped: bool = True) -> None:
        self.tasks = [
            asyncio.create_task(self.app._inbound_loop()),
            asyncio.create_task(self.app.run()),
        ]
        self.post(SessionStarted("s", stepped=stepped))
        self.post(ClientConnected(CLIENT, 1))
        await self.settle()

    def post(self, item: Any) -> None:
        self.queue.put_nowait(item)

    def command(self, name: str, **args: Any) -> None:
        self.post(CommandEnvelope(self._validate(name, args), CLIENT, None))

    def step(
        self,
        request_id: str,
        *commands: tuple[str, dict[str, Any]],
        frames: int = 0,
    ) -> None:
        # The runtime posts a request's commands ahead of it, as a client's own.
        for name, args in commands:
            self.command(name, **args)
        media_input = {
            "webcam": tuple(
                InputFrame(data=np.full((2, 2, 3), i, dtype=np.uint8)) for i in range(frames)
            )
        }
        self.post(
            StepEnvelope(
                media_input=media_input if frames else {},
                conn_id=CLIENT,
                request_id=request_id,
            )
        )

    def _validate(self, name: str, args: dict[str, Any]) -> Any:
        return ModelContract.of(type(self.app)).validate(name, args)

    async def settle(self) -> None:
        for _ in range(20):
            await asyncio.sleep(0.005)
            if self.queue.empty() and self.app._pending_step is None:
                return

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


@pytest.fixture
async def harness() -> Any:
    h = Harness(Stepper())
    try:
        yield h
    finally:
        await h.stop()


async def test_a_stepped_session_runs_no_step_until_one_is_requested(harness: Harness) -> None:
    await harness.start()
    await asyncio.sleep(0.02)
    assert harness.app.connected.is_set()
    assert not harness.app._live.is_set()
    assert harness.app.inputs == []


async def test_a_requested_step_writes_its_frames_then_runs_once_after_its_commands(
    harness: Harness,
) -> None:
    await harness.start()
    harness.step("r1", ("set_prompt", {"prompt": "a lake"}), ("note", {"text": "x"}), frames=1)
    await harness.settle()

    assert harness.app.inputs == [("a lake", ["x"], 1)]
    assert harness.refusals == []
    assert len(harness.reports) == 1
    report = harness.reports[0]
    assert report.request is not None
    assert (report.request.request_id, report.request.index) == ("r1", 1)
    assert report.bundle is not None
    assert harness.app.emitted == 0


async def test_steps_are_numbered_from_one_and_a_refused_step_takes_no_number(
    harness: Harness,
) -> None:
    await harness.start()
    harness.step("r1")
    harness.step("r2", ("set_prompt", {"prompt": "refuse"}))
    harness.step("r3", ("set_prompt", {"prompt": "a lake"}))
    await harness.settle()

    assert harness.refusals == [StepRefusal(CLIENT, "r2", 1, "not now")]
    assert [(r.request.request_id, r.request.index) for r in harness.reports if r.request] == [
        ("r1", 1),
        ("r3", 2),
    ]


async def test_a_step_process_input_refuses_leaves_the_state_and_frames_it_was_given(
    harness: Harness,
) -> None:
    await harness.start()
    harness.step("r1", ("set_prompt", {"prompt": "refuse"}), frames=2)
    await harness.settle()

    assert harness.refusals[0].reason == "not now"
    assert harness.app.state.prompt == "refuse"
    assert webcam(harness.app).available == 2
    assert harness.reports == []
    assert harness.app.inputs == []


async def test_a_failing_command_is_answered_alone_and_the_step_still_runs(
    harness: Harness,
) -> None:
    await harness.start()
    harness.step(
        "r1",
        ("set_prompt", {"prompt": "a lake"}),
        ("reject", {}),
        ("note", {"text": "after"}),
        frames=1,
    )
    await harness.settle()

    assert harness.refusals == []
    [(conn_id, failure, request_id)] = harness.addressed
    assert (conn_id, failure.code, request_id) == (CLIENT, "rejected", None)
    assert harness.app.inputs == [("a lake", ["after"], 1)]


async def test_an_empty_request_runs_a_step_on_the_current_state(harness: Harness) -> None:
    await harness.start()
    harness.command("set_prompt", prompt="a lake")
    harness.step("r1")
    await harness.settle()

    assert harness.app.inputs == [("a lake", [], 0)]
    assert [r.request.index for r in harness.reports if r.request] == [1]


async def test_a_step_runs_after_the_commands_sent_before_it_and_before_those_after(
    harness: Harness,
) -> None:
    await harness.start()
    harness.command("note", text="before")
    harness.step("r1")
    harness.command("note", text="after")
    await harness.settle()

    assert harness.app.inputs == [("a forest", ["before"], 0)]
    assert harness.app.notes == ["before", "after"]


async def test_frames_follow_the_models_read_mode_across_steps(harness: Harness) -> None:
    # The model reads one frame FIFO per step, so the second frame of the first
    # request is still buffered for the next step.
    await harness.start()
    harness.step("r1", frames=2)
    harness.step("r2")
    await harness.settle()

    assert [frames for _, _, frames in harness.app.inputs] == [1, 1]
    assert webcam(harness.app).available == 0


async def test_a_step_requested_in_a_session_that_does_not_step_is_refused(
    harness: Harness,
) -> None:
    await harness.start(stepped=False)
    harness.step("r1")
    await harness.settle()

    assert harness.refusals == [StepRefusal(CLIENT, "r1", 1, "the session is not stepped")]
    assert [r for r in harness.reports if r.request is not None] == []


async def test_a_step_requested_with_no_session_running_is_not_ready(harness: Harness) -> None:
    harness.tasks = [
        asyncio.create_task(harness.app._inbound_loop()),
        asyncio.create_task(harness.app.run()),
    ]
    harness.step("r1")
    await harness.settle()

    assert harness.refusals == [
        StepRefusal(CLIENT, "r1", 0, "no session is running", not_ready=True)
    ]
    assert harness.reports == []


async def test_the_end_of_a_stepped_session_empties_the_input_tracks(harness: Harness) -> None:
    await harness.start()
    harness.step("r1", frames=3)
    await harness.settle()
    assert webcam(harness.app).available == 2

    harness.post(SessionEnded("s", EndReason.STOPPED))
    await harness.settle()
    assert webcam(harness.app).available == 0
    assert not harness.app._stepped


async def test_a_model_failure_in_a_requested_step_ends_the_loop() -> None:
    h = Harness(Failing())
    await h.start()
    h.step("r1")
    await asyncio.sleep(0.05)
    run = h.tasks[1]
    assert run.done()
    with pytest.raises(RuntimeError, match="model blew up"):
        run.result()
    assert h.refusals == []
    assert h.reports == []
    await h.stop()


def test_only_the_default_step_loop_supports_stepped_sessions() -> None:
    class OwnLoop(Stepper):
        async def run(self) -> None:
            await asyncio.sleep(0)

    class Pipeline(ReactorPipeline):
        state: State

        def inference(self) -> Any:
            yield None

    assert Stepper.supports_stepped()
    assert not OwnLoop.supports_stepped()
    assert not Pipeline.supports_stepped()
