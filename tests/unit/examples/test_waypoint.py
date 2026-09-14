"""Surface and behaviour checks for the waypoint example.

The model half is tested with a fake engine in place of ``world_engine``, so
the suite runs without a GPU, torch, or the weights. The application half is
driven through its hooks the way the runtime's loop drives them.
"""

from __future__ import annotations

import io
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from waypoint_model import NotSeeded, WaypointModel, WaypointStepInput, WaypointStepResult

from reactor_runtime import ApplicationError, StepOutcome, UploadedFile
from reactor_runtime.core.model import EndReason, SessionEnded, SessionStarted
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config
from waypoint import Waypoint, WaypointOutput, WaypointState, WaypointStatus

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "waypoint"
_SEED = np.full((720, 1280, 3), 7, dtype=np.uint8)


class FakeEngine:
    """Records what the model half asks of the engine and returns four frames."""

    def __init__(self) -> None:
        self.resets = 0
        self.appended: list[Any] = []
        self.generated: list[Any] = []

    def reset(self) -> None:
        self.resets += 1

    def append_frame(self, frames: Any) -> None:
        self.appended.append(frames)

    def gen_frame(self, ctrl: Any = None) -> Any:
        self.generated.append(ctrl)
        return _FakeTensor(np.zeros((4, 720, 1280, 3), dtype=np.uint8))


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


@pytest.fixture
def fake_world_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the ``torch`` and ``world_engine`` imports inside the model half."""
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda array: array  # type: ignore[ty:unresolved-attribute]
    torch.no_grad = lambda: _NoGrad()  # type: ignore[ty:unresolved-attribute]
    world_engine = types.ModuleType("world_engine")
    world_engine.CtrlInput = _CtrlInput  # type: ignore[ty:unresolved-attribute]
    world_engine.WorldEngine = lambda *args, **kwargs: FakeEngine()  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "world_engine", world_engine)


class _NoGrad:
    def __enter__(self) -> None: ...

    def __exit__(self, *exc: object) -> None: ...


class _CtrlInput:
    def __init__(
        self, button: set[int] | None = None, mouse: Any = (0.0, 0.0), scroll_wheel: int = 0
    ):
        self.button = button or set()
        self.mouse = mouse
        self.scroll_wheel = scroll_wheel


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Waypoint)


def _step(seed_id: int = 1, seed: np.ndarray | None = _SEED) -> WaypointStepInput:
    return WaypointStepInput(
        buttons=frozenset({0x57}), mouse=(0.1, 0.2), scroll_wheel=1, seed=seed, seed_id=seed_id
    )


def _loaded_model() -> tuple[WaypointModel, FakeEngine]:
    model = WaypointModel()
    engine = FakeEngine()
    model.engine = engine
    return model, engine


# -- the client contract ------------------------------------------------------


def test_commands_are_the_state_setters_plus_reset() -> None:
    assert set(ModelContract.of(Waypoint).commands) == {
        "set_image",
        "set_paused",
        "set_action",
        "set_buttons",
        "set_mouse_x",
        "set_mouse_y",
        "set_scroll_wheel",
        "reset",
    }


def test_set_image_is_hand_written_and_replies_with_the_status() -> None:
    spec = ModelContract.of(Waypoint).commands["set_image"]
    assert spec.response is WaypointStatus
    assert "seed frame" in spec.description


def test_the_one_output_track_is_main_video() -> None:
    assert list(WaypointOutput.__tracks__) == ["main_video"]


def test_manifest_resolves_to_the_app_class() -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert import_model_class(cfg.model_ref) is Waypoint
    assert cfg.config_path == _EXAMPLE_DIR / "config.yml"


def test_button_set_merges_the_action_with_the_extra_buttons() -> None:
    state = WaypointState()
    state.action = "forward_left"
    state.buttons = "32, 16, junk, 999"
    assert state.button_set() == frozenset({0x57, 0x41, 32, 16})


# -- the model half -----------------------------------------------------------


def test_load_constructs_the_engine_and_runs_the_warmup(
    fake_world_engine: None, tmp_path: Path
) -> None:
    config = tmp_path / "config.yml"
    config.write_text("model_uri: local/weights\nwarmup_steps: 3\n")
    model = WaypointModel()
    model.load(config)
    assert isinstance(model.engine, FakeEngine)
    assert len(model.engine.generated) == 3
    assert model.seed_id is None
    assert model.index == 0


def test_a_new_seed_id_starts_a_world_from_the_seed(fake_world_engine: None) -> None:
    model, engine = _loaded_model()
    result = model.generate(_step(seed_id=1))
    assert engine.resets == 1
    assert engine.appended[0].shape == (4, 720, 1280, 3)
    assert result.index == 0
    assert result.frames.shape == (4, 720, 1280, 3)


def test_the_same_seed_id_continues_the_world(fake_world_engine: None) -> None:
    model, engine = _loaded_model()
    model.generate(_step(seed_id=1))
    second = model.generate(_step(seed_id=1))
    assert engine.resets == 1
    assert second.index == 1


def test_generate_passes_the_controls_to_the_engine(fake_world_engine: None) -> None:
    model, engine = _loaded_model()
    model.generate(_step())
    ctrl = engine.generated[0]
    assert ctrl.button == {0x57}
    assert ctrl.mouse == (0.1, 0.2)
    assert ctrl.scroll_wheel == 1


def test_a_new_world_without_a_seed_is_the_models_own_error(fake_world_engine: None) -> None:
    model, _ = _loaded_model()
    with pytest.raises(NotSeeded):
        model.generate(_step(seed_id=1, seed=None))


def test_reset_forgets_the_seed_so_the_next_step_reapplies_it(fake_world_engine: None) -> None:
    model, engine = _loaded_model()
    model.generate(_step(seed_id=1))
    model.reset()
    assert model.seed_id is None
    result = model.generate(_step(seed_id=1))
    assert engine.resets == 3
    assert result.index == 0


def test_generate_before_load_is_a_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        WaypointModel().generate(_step())


# -- the application half -----------------------------------------------------


class RecordingModel:
    """A model half that records the steps it was asked for."""

    def __init__(self) -> None:
        self.steps: list[WaypointStepInput] = []
        self.resets = 0

    def generate(self, step: WaypointStepInput) -> WaypointStepResult:
        self.steps.append(step)
        return WaypointStepResult(
            frames=np.zeros((4, 8, 8, 3), dtype=np.uint8), index=len(self.steps) - 1
        )

    def reset(self) -> None:
        self.resets += 1


def _app() -> tuple[Waypoint, RecordingModel, list[Any]]:
    app = Waypoint()
    model = RecordingModel()
    app.engine = model  # type: ignore[ty:invalid-assignment]  # the model half by shape
    app.progress_interval = 2
    app.last_index = -1
    sent: list[Any] = []
    app._on_loop_ready()
    app.bind_output(broadcast=sent.append, addressed=lambda *args: None, media=lambda chunk: None)
    return app, model, sent


async def test_prepare_step_refuses_while_paused() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    app.state._seed = _SEED
    app.state.paused = True
    with pytest.raises(ApplicationError, match="paused"):
        await app.prepare_step(app.state, None)


async def test_prepare_step_refuses_before_a_seed() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    with pytest.raises(ApplicationError, match="no seed image"):
        await app.prepare_step(app.state, None)


async def test_prepare_step_builds_the_step_input_from_the_state() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    app.state._seed = _SEED
    app.state._seed_id = 3
    app.state.action = "back"
    app.state.mouse_x = 0.5
    app.state.scroll_wheel = -1
    step = await app.prepare_step(app.state, None)
    assert step.buttons == frozenset({0x53})
    assert step.mouse == (0.5, 0.0)
    assert step.scroll_wheel == -1
    assert step.seed is _SEED
    assert step.seed_id == 3


async def test_generate_forwards_to_the_model_half() -> None:
    app, model, _ = _app()
    result = app.generate(_step())
    assert model.steps == [_step()]
    assert result.index == 0


async def test_collect_step_tags_every_frame_with_its_step() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    result = WaypointStepResult(frames=np.zeros((4, 8, 8, 3), dtype=np.uint8), index=5)
    output = await app.collect_step(StepOutcome(result=result, elapsed=0.01))
    assert isinstance(output, WaypointOutput)
    assert output.__metadata__["main_video"] == [{"step": 5}] * 4
    assert app.last_index == 5


async def test_collect_step_sends_the_status_on_the_cadence() -> None:
    app, _, sent = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    for index in range(4):
        result = WaypointStepResult(frames=np.zeros((4, 8, 8, 3), dtype=np.uint8), index=index)
        await app.collect_step(StepOutcome(result=result))
    assert [message.step_index for message in sent] == [0, 2]
    assert all(isinstance(message, WaypointStatus) for message in sent)


async def test_collect_step_reraises_a_model_error() -> None:
    app, _, _ = _app()
    with pytest.raises(NotSeeded):
        await app.collect_step(StepOutcome(error=NotSeeded("no seed")))


async def test_set_image_stages_the_seed_and_bumps_the_seed_id() -> None:
    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (200, 10, 10)).save(buffer, format="PNG")
    upload = UploadedFile(name="seed.png", mime_type="image/png", data=buffer.getvalue())
    status = await app.set_image(upload)
    assert app.state._seed is not None
    assert app.state._seed.shape == (720, 1280, 3)
    assert app.state._seed_id == 1
    assert app.state.image is upload
    assert status.has_image is True


async def test_set_image_rejects_a_file_that_is_not_an_image() -> None:
    from reactor_runtime import CommandError

    app, _, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    upload = UploadedFile(name="notes.txt", mime_type="text/plain", data=b"hello")
    with pytest.raises(CommandError):
        await app.set_image(upload)
    assert app.state._seed is None


async def test_reset_and_session_end_reach_the_model_half() -> None:
    app, model, _ = _app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    app.reset()
    assert model.resets == 1
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert model.resets == 2
