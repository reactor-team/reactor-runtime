"""How a ReactorRobotApp delivers each step's result: ``actions`` and ``actions_failed``.

Captures are pushed into the app's camera tracks and its ``observe`` command as
a robot sends them; every message the app broadcasts is recorded.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest

from reactor_runtime import MediaInput, Video
from reactor_runtime.apps.robot import Actions, Observation, ReactorRobotApp
from reactor_runtime.apps.robot.messages import ActionsFailed, ActionsMessage
from reactor_runtime.core.model import ClientConnected, SessionStarted
from reactor_runtime.core.values import CompletedStep, ConnId, InputFrame
from reactor_runtime.interface.app.outcome import StepOutcome
from reactor_runtime.interface.app.step_completed import StepCompleted
from reactor_runtime.interface.events.messages import ModelMessage
from reactor_runtime.interface.tracks import Output

CONN = ConnId(1001)


class OneCamera(MediaInput):
    scene: Video


class ChunkPolicy(ReactorRobotApp):
    """Returns a two-row chunk whose values name the step it came from."""

    media: OneCamera
    delay_s = 0.0

    def generate(self, input: Observation, /) -> Actions:
        if self.delay_s:
            time.sleep(self.delay_s)
        return Actions(
            [[float(input.step), 0.5], [float(input.step), 1.5]],
            dt_ms=20.0,
            names=["shoulder", "elbow"],
            extra={"n": dict(input.state)["n"]},
        )


class SlowChunkPolicy(ChunkPolicy):
    delay_s = 0.02


class FlakyPolicy(ChunkPolicy):
    """Raises on the steps listed in ``failing``."""

    failing = frozenset({0})

    def generate(self, input: Observation, /) -> Actions:
        if input.step in self.failing:
            raise RuntimeError(f"no grip on step {input.step}")
        return super().generate(input)


class OverridingPolicy(ChunkPolicy):
    """A process_output() that tries to drop the actions."""

    async def process_output(self, outcome: StepOutcome, /) -> Output | StepCompleted | None:
        return StepCompleted(data={"kept": True})


class UnserializableExtraPolicy(ChunkPolicy):
    def generate(self, input: Observation, /) -> Actions:
        return Actions([[1.0]], dt_ms=10.0, extra={"handle": object()})


class StrictPolicy(ChunkPolicy):
    """Ends the model loop on a model error, as a ReactorApp does."""

    def generate(self, input: Observation, /) -> Actions:
        raise RuntimeError("broken")

    async def process_output(self, outcome: StepOutcome, /) -> Output | StepCompleted | None:
        if outcome.error is not None:
            raise outcome.error
        return await super().process_output(outcome)


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(ChunkPolicy)


def _ready(app: ChunkPolicy) -> tuple[list[ModelMessage], list[CompletedStep]]:
    sent: list[ModelMessage] = []
    steps: list[CompletedStep] = []
    app._on_loop_ready()
    app.bind_output(
        broadcast=sent.append,
        addressed=lambda *args: None,
        media=lambda chunk: None,
        step=steps.append,
    )
    return sent, steps


async def _go_live(app: ChunkPolicy) -> None:
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(CONN, 1))


@asynccontextmanager
async def _running(app: ChunkPolicy) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(app.run())
    try:
        yield task
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def _stamp(n: int) -> int:
    return 1_000 * (n + 1)


def _capture(app: ChunkPolicy, n: int) -> None:
    """Send capture *n* as a robot does: the camera frame and the state, one stamp."""
    frame = InputFrame(data=np.full((2, 2, 3), n % 256, dtype=np.uint8), capture_time_us=_stamp(n))
    app._input_buffers["scene"].push(frame, CONN, app._sessions_started)
    app._observe(capture_time_us=_stamp(n), state={"n": n})


async def _until(condition: Callable[[], bool], within: float = 2.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the condition")
        await asyncio.sleep(0.002)


def _of(sent: list[ModelMessage], kind: type[Any]) -> list[Any]:
    return [message for message in sent if isinstance(message, kind)]


async def test_each_step_sends_one_actions_message_with_its_observation_stamp() -> None:
    app = SlowChunkPolicy()
    sent, steps = _ready(app)
    await _go_live(app)
    async with _running(app):
        for n in range(3):
            _capture(app, n)
            await _until(lambda n=n: len(steps) == n + 1)

    delivered: list[ActionsMessage] = _of(sent, ActionsMessage)
    assert [(a.step, a.capture_time_us) for a in delivered] == [(n, _stamp(n)) for n in range(3)]
    for n, actions in enumerate(delivered):
        assert actions.chunk == [[float(n), 0.5], [float(n), 1.5]]
        assert actions.dt_ms == 20.0
        assert actions.names == ["shoulder", "elbow"]
        assert actions.extra == {"n": n}
        assert actions.generate_ms >= 20.0
        # The capture arrived before generate() ran, so it is at least that old.
        assert actions.observation_age_ms >= actions.generate_ms
    assert all(step.error is None for step in steps)
    assert _of(sent, ActionsFailed) == []


async def test_the_actions_message_goes_on_the_wire_as_actions() -> None:
    app = ChunkPolicy()
    sent, steps = _ready(app)
    await _go_live(app)
    async with _running(app):
        _capture(app, 0)
        await _until(lambda: len(steps) == 1)

    wire = _of(sent, ActionsMessage)[0].to_wire_format()
    assert wire["type"] == "actions"
    assert set(wire["data"]) == {
        "step",
        "capture_time_us",
        "chunk",
        "dt_ms",
        "names",
        "extra",
        "generate_ms",
        "observation_age_ms",
    }
    assert ActionsFailed.name == "actions_failed"


async def test_a_raising_generate_sends_actions_failed_and_the_next_tick_runs() -> None:
    app = FlakyPolicy()
    sent, steps = _ready(app)
    await _go_live(app)
    async with _running(app) as task:
        _capture(app, 0)
        await _until(lambda: len(steps) == 1)
        _capture(app, 1)
        await _until(lambda: len(steps) == 2)
        assert not task.done()

    failed: list[ActionsFailed] = _of(sent, ActionsFailed)
    assert [(f.step, f.capture_time_us) for f in failed] == [(0, _stamp(0))]
    assert failed[0].reason == "RuntimeError: no grip on step 0"
    delivered: list[ActionsMessage] = _of(sent, ActionsMessage)
    assert [(a.step, a.capture_time_us) for a in delivered] == [(1, _stamp(1))]
    assert steps[0].error == "RuntimeError: no grip on step 0"
    assert steps[1].error is None


async def test_an_override_of_process_output_cannot_suppress_delivery() -> None:
    app = OverridingPolicy()
    sent, steps = _ready(app)
    await _go_live(app)
    async with _running(app):
        _capture(app, 0)
        await _until(lambda: len(steps) == 1)

    delivered: list[ActionsMessage] = _of(sent, ActionsMessage)
    assert [(a.step, a.capture_time_us) for a in delivered] == [(0, _stamp(0))]


async def test_an_extra_that_is_not_json_is_answered_with_actions_failed() -> None:
    app = UnserializableExtraPolicy()
    sent, steps = _ready(app)
    await _go_live(app)
    async with _running(app):
        _capture(app, 0)
        await _until(lambda: len(steps) == 1)

    assert _of(sent, ActionsMessage) == []
    failed: list[ActionsFailed] = _of(sent, ActionsFailed)
    assert [(f.step, f.capture_time_us) for f in failed] == [(0, _stamp(0))]
    assert "not JSON" in failed[0].reason
    assert steps[0].error == failed[0].reason


async def test_a_process_output_that_re_raises_still_ends_the_loop() -> None:
    app = StrictPolicy()
    sent, _ = _ready(app)
    await _go_live(app)
    task = asyncio.create_task(app.run())
    _capture(app, 0)
    with pytest.raises(RuntimeError, match="broken"):
        await asyncio.wait_for(task, timeout=2.0)
    assert _of(sent, ActionsMessage) == []
    assert _of(sent, ActionsFailed) == []


async def test_a_failed_generate_without_a_capture_is_re_raised() -> None:
    app = ChunkPolicy()
    _ready(app)
    error = RuntimeError("stepped failure")
    with pytest.raises(RuntimeError, match="stepped failure"):
        await app.process_output(StepOutcome(error=error))
