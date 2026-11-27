"""A ReactorRobotApp subclass builds its contract and runs its steps like a ReactorApp.

The same policy body is declared twice, once on each base, and the two classes
are compared: what a client sees in the contract, and what the default step
loop does with ``generate()``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np
import pytest

from reactor_runtime import (
    ApplicationError,
    InputField,
    InputState,
    MediaInput,
    Output,
    ReactorApp,
    Video,
    event,
    session_started,
)
from reactor_runtime.apps.robot import Observation, ReactorRobotApp
from reactor_runtime.core.model import ClientConnected, SessionStarted
from reactor_runtime.core.values import CompletedStep, ConnId
from reactor_runtime.interface.model.contract import ModelContract


class PolicyView(Output):
    scene_view: Video


class PolicyCameras(MediaInput):
    scene: Video


class PolicyState(InputState):
    task: str = InputField(default="pick up the cube")
    paused: bool = InputField(default=False)


class PolicyBody:
    """The policy both test classes declare, base aside."""

    state: PolicyState
    media: PolicyCameras

    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[Output] = []
        self.observations: list[Observation] = []
        self.sessions = 0

    async def emit(
        self, output: Output, *, compute_time: float | None = None, drop: bool = False
    ) -> None:
        self.emitted.append(output)
        await asyncio.sleep(0)

    @event(name="home", description="Send the arm to its home pose.")
    def home(self) -> None:
        pass

    @session_started
    def on_session(self) -> None:
        self.sessions += 1

    async def process_input(self) -> Observation:
        state: PolicyState = self.state
        if state.paused:
            raise ApplicationError("paused")
        return Observation(
            step=len(self.observations),
            capture_time_us=1_000 * len(self.observations),
            frames={"scene": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={"joints": [0.0, 0.0]},
            task=state.task,
        )

    def generate(self, input: Observation) -> PolicyView:
        self.observations.append(input)
        return PolicyView(scene_view=np.zeros((2, 2, 3), dtype=np.uint8))


class PlainPolicy(PolicyBody, ReactorApp):
    pass


class RobotPolicy(PolicyBody, ReactorRobotApp):
    pass


class BrokenBody(PolicyBody):
    def generate(self, input: Observation) -> PolicyView:
        raise RuntimeError("no policy here")


class BrokenPlainPolicy(BrokenBody, ReactorApp):
    pass


class BrokenRobotPolicy(BrokenBody, ReactorRobotApp):
    pass


type Policy = PlainPolicy | RobotPolicy


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(PlainPolicy)
    register_model(RobotPolicy)


def _ready(app: ReactorApp) -> list[CompletedStep]:
    steps: list[CompletedStep] = []
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None,
        addressed=lambda *args: None,
        media=lambda chunk: None,
        step=steps.append,
    )
    return steps


async def _go_live(app: ReactorApp) -> None:
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))


async def _run_for(app: ReactorApp, seconds: float = 0.02) -> None:
    task = asyncio.create_task(app.run())
    await asyncio.sleep(seconds)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_the_package_exports_the_app_and_its_step_types() -> None:
    import reactor_runtime.apps.robot as robot

    assert robot.__all__ == ["Actions", "Observation", "ReactorRobotApp"]
    assert issubclass(ReactorRobotApp, ReactorApp)


def test_a_subclass_builds_the_same_contract_as_a_reactor_app() -> None:
    plain = ModelContract.of(PlainPolicy).render_schema(name="policy")
    robot = ModelContract.of(RobotPolicy).render_schema(name="policy")

    assert robot == plain
    assert set(robot.commands) == {"home", "set_task", "set_paused"}
    assert set(robot.tracks) == {"scene", "scene_view"}


def test_a_subclass_binds_its_handlers_and_hooks_like_a_reactor_app() -> None:
    contract = ModelContract.of(RobotPolicy)

    assert contract.commands["home"].handler is PolicyBody.home
    assert contract.lifecycle.session_started is PolicyBody.on_session
    assert RobotPolicy.__app_state__ is PolicyState


def test_a_subclass_serves_stepped_sessions_like_a_reactor_app() -> None:
    assert RobotPolicy.supports_stepped()
    assert RobotPolicy.supports_stepped() == PlainPolicy.supports_stepped()


@pytest.mark.parametrize("cls", [PlainPolicy, RobotPolicy])
async def test_the_default_step_loop_runs_generate_and_reports_each_step(
    cls: type[Policy],
) -> None:
    app = cls()
    steps = _ready(app)
    await _go_live(app)
    await _run_for(app)

    assert app.sessions == 1
    assert app.observations
    assert app.observations[0].task == "pick up the cube"
    assert len(app.emitted) == len(app.observations)
    assert len(app.observations) - 1 <= len(steps) <= len(app.observations)
    assert steps[0].bundle is not None
    assert set(steps[0].bundle.tracks) == {"scene_view"}


@pytest.mark.parametrize("cls", [PlainPolicy, RobotPolicy])
async def test_a_refused_step_never_reaches_generate(cls: type[Policy]) -> None:
    app = cls()
    steps = _ready(app)
    await _go_live(app)
    app.state.paused = True
    await _run_for(app)

    assert app.observations == []
    assert steps == []


@pytest.mark.parametrize("cls", [BrokenPlainPolicy, BrokenRobotPolicy])
async def test_an_error_from_generate_ends_the_loop(
    cls: type[BrokenPlainPolicy | BrokenRobotPolicy],
) -> None:
    app = cls()
    _ready(app)
    await _go_live(app)
    with pytest.raises(RuntimeError, match="no policy here"):
        await asyncio.wait_for(app.run(), timeout=1.0)
