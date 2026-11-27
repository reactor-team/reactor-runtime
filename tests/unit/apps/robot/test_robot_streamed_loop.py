"""The streamed loop of a ReactorRobotApp: camera readers, the synchronizer and the policy.

Frames are pushed into the app's camera tracks as the transport pushes them,
and robot states arrive as ``observe`` commands; the policy records what each
``generate()`` call got.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import numpy as np
import pytest

from reactor_runtime import MediaInput, Video
from reactor_runtime.apps.robot import Actions, Observation, ReactorRobotApp
from reactor_runtime.apps.robot.app import NO_OBSERVATION
from reactor_runtime.apps.robot.loop import PolicyPacer
from reactor_runtime.core.model import ClientConnected, ClientDisconnected, SessionStarted
from reactor_runtime.core.values import CompletedStep, ConnId, InputFrame
from reactor_runtime.interface.events.errors import ApplicationError

CONN = ConnId(1001)


class TwoCameras(MediaInput):
    scene: Video
    wrist: Video


class StateCamera(MediaInput):
    state: Video


class RecordingPolicy(ReactorRobotApp):
    """Records every observation and the thread ``generate()`` ran on."""

    media: TwoCameras
    delay_s = 0.0

    def __init__(self) -> None:
        super().__init__()
        self.observations: list[Observation] = []
        self.threads: set[int] = set()
        self.called_at: list[float] = []

    def generate(self, input: Observation, /) -> Actions:
        self.called_at.append(time.monotonic())
        self.threads.add(threading.get_ident())
        if self.delay_s:
            time.sleep(self.delay_s)
        self.observations.append(input)
        return Actions([[float(input.step)]], dt_ms=10.0)


class SlowPolicy(RecordingPolicy):
    delay_s = 0.03


class PacedPolicy(RecordingPolicy):
    fps = 40


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(RecordingPolicy)


def _ready(app: RecordingPolicy) -> list[CompletedStep]:
    steps: list[CompletedStep] = []
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None,
        addressed=lambda *args: None,
        media=lambda chunk: None,
        step=steps.append,
    )
    return steps


async def _go_live(app: RecordingPolicy) -> None:
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(CONN, 1))


@asynccontextmanager
async def _running(app: RecordingPolicy) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(app.run())
    try:
        yield task
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def _frame(value: int, stamp: int | None) -> InputFrame:
    return InputFrame(data=np.full((2, 2, 3), value, dtype=np.uint8), capture_time_us=stamp)


def _capture(app: RecordingPolicy, n: int, *, state: bool = True) -> None:
    """Send capture *n* as a robot does: both camera frames and the state, one stamp."""
    stamp = 1_000 * (n + 1)
    app._input_buffers["scene"].push(_frame(n % 256, stamp), CONN, app._sessions_started)
    app._input_buffers["wrist"].push(_frame(n % 256, stamp), CONN, app._sessions_started)
    if state:
        app._observe(capture_time_us=stamp, state={"n": n})


async def _until(condition: Callable[[], bool], within: float = 2.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the condition")
        await asyncio.sleep(0.002)


async def test_each_step_gets_one_observation_whose_frames_and_state_match() -> None:
    app = RecordingPolicy()
    steps = _ready(app)
    await _go_live(app)
    async with _running(app):
        for n in range(3):
            _capture(app, n)
            await _until(lambda n=n: len(app.observations) == n + 1)
        await _until(lambda: len(steps) == 3)

    assert [obs.step for obs in app.observations] == [0, 1, 2]
    for n, obs in enumerate(app.observations):
        assert obs.capture_time_us == 1_000 * (n + 1)
        assert set(obs.frames) == {"scene", "wrist"}
        assert (obs.frames["scene"] == n).all()
        assert (obs.frames["wrist"] == n).all()
        assert dict(obs.state) == {"n": n}
        assert dict(obs.skew_us) == {"scene": 0, "wrist": 0}
    assert [step.bundle for step in steps] == [None, None, None]


async def test_states_sent_before_the_gate_opens_join_the_first_frames() -> None:
    app = RecordingPolicy()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    app._observe(capture_time_us=1_000, state={"n": 0})
    await app._dispatch_reactor_event(ClientConnected(CONN, 1))
    async with _running(app):
        _capture(app, 0, state=False)
        await _until(lambda: len(app.observations) == 1)

    assert dict(app.observations[0].state) == {"n": 0}


async def test_no_complete_capture_means_no_generate_call() -> None:
    app = RecordingPolicy()
    steps = _ready(app)
    await _go_live(app)
    async with _running(app):
        # Frames without their state, and a state for a capture with no frames.
        _capture(app, 0, state=False)
        app._observe(capture_time_us=999, state={"n": -1})
        await asyncio.sleep(0.05)

    assert app.observations == []
    assert steps == []
    assert app._sync.stats().taken == 0


async def test_the_default_process_input_refuses_with_no_observation() -> None:
    app = RecordingPolicy()
    _ready(app)
    with pytest.raises(ApplicationError, match=NO_OBSERVATION):
        await app.process_input()


async def test_a_frame_without_a_capture_stamp_is_dropped_and_counted() -> None:
    app = RecordingPolicy()
    _ready(app)
    await _go_live(app)
    async with _running(app):
        app._input_buffers["scene"].push(_frame(1, None), CONN, app._sessions_started)
        await _until(lambda: app._unstamped == 1)

    assert app._sync.stats().pushed == 0


async def test_generate_runs_off_the_event_loop() -> None:
    app = RecordingPolicy()
    _ready(app)
    await _go_live(app)
    async with _running(app):
        _capture(app, 0)
        await _until(lambda: len(app.observations) == 1)

    assert app.threads
    assert threading.get_ident() not in app.threads


async def test_a_slow_model_skips_captures_instead_of_queueing_them() -> None:
    app = SlowPolicy()
    _ready(app)
    await _go_live(app)
    sent = 0
    async with _running(app):
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            _capture(app, sent)
            sent += 1
            await asyncio.sleep(0.002)
        # The captures keep coming while the model runs on its thread.
        await asyncio.sleep(0.05)

    stamps = [obs.capture_time_us for obs in app.observations]
    assert stamps == sorted(set(stamps))
    assert 2 <= len(stamps) < sent // 3
    stats = app._sync.stats()
    assert stats.skipped + stats.dropped > 0
    assert stats.taken == len(stamps)


async def test_a_class_fps_spaces_the_steps() -> None:
    app = PacedPolicy()
    _ready(app)
    await _go_live(app)
    sent = 0
    async with _running(app):
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            _capture(app, sent)
            sent += 1
            await asyncio.sleep(0.002)

    gaps = np.diff(app.called_at)
    assert len(app.called_at) >= 3
    # 40 steps a second, so 25 ms apart; the grid never starts one early.
    assert gaps.min() >= 0.02
    assert len(app.called_at) <= 0.3 * 40 + 2


async def test_the_last_client_leaving_drops_what_the_synchronizer_holds() -> None:
    app = RecordingPolicy()
    _ready(app)
    await _go_live(app)
    async with _running(app):
        _capture(app, 0, state=False)
        await _until(lambda: app._sync.stats().buffered["scene"] == 1)
        await app._dispatch_reactor_event(ClientDisconnected(CONN, 0))
        await _until(lambda: not any(app._sync.stats().buffered.values()))

    assert app.observations == []


def test_a_camera_named_state_fails_when_the_class_is_defined() -> None:
    with pytest.raises(TypeError, match="camera named 'state'"):

        class Clashing(ReactorRobotApp):
            media: StateCamera


async def test_the_pacer_keeps_a_grid_and_skips_missed_slots() -> None:
    pacer = PolicyPacer(100)
    assert pacer.period == pytest.approx(0.01)
    started = time.monotonic()
    for _ in range(3):
        await pacer.wait()
        pacer.started()
    assert time.monotonic() - started >= 0.02

    # A step that overruns by more than a slot starts a new grid from its start.
    await asyncio.sleep(0.03)
    await pacer.wait()
    late = time.monotonic()
    pacer.started()
    await pacer.wait()
    assert time.monotonic() - late >= 0.009


async def test_a_pacer_without_a_rate_never_waits() -> None:
    pacer = PolicyPacer(None)
    pacer.started()
    started = time.monotonic()
    await pacer.wait()
    assert time.monotonic() - started < 0.005


def test_a_pacer_refuses_a_rate_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        PolicyPacer(0)
