"""The generic FlashDreams application half, driven through its hooks.

No FlashDreams and no GPU: a fake registry and a fake family package stand in
for the two imports ``load()`` makes, a recording model half stands in for the
pipeline, and a recording runner stands in for ``DistributedRunner``. What is
tested is the scaffold every family shares, not any family.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from reactor_runtime import ApplicationError, StepOutcome
from reactor_runtime.core.model import EndReason, SessionEnded, SessionStarted
from reactor_runtime.flashdreams import (
    FlashDreamsApp,
    FlashDreamsOutput,
    FlashDreamsResult,
    PromptSwapUnsupported,
    RolloutExhausted,
    RolloutNotStarted,
    RolloutRestarted,
)
from reactor_runtime.flashdreams import app as app_module
from reactor_runtime.flashdreams import model as model_module
from reactor_runtime.interface.app.reactor_app import _fps_is_author_pinned
from reactor_runtime.interface.model.contract import ModelContract

_FRAMES = np.zeros((4, 8, 8, 3), dtype=np.uint8)


@dataclass(frozen=True)
class Step:
    rollout_id: int
    control: Any = None


class RecordingModel:
    """A model half that records what it was asked and answers with scripted results."""

    def __init__(self) -> None:
        self.load_kwargs: dict[str, Any] | None = None
        self.steps: list[Step] = []
        self.resets = 0

    def load(self, **kwargs: Any) -> None:
        self.load_kwargs = kwargs

    def generate(self, step: Step) -> FlashDreamsResult:
        self.steps.append(step)
        return FlashDreamsResult(frames=_FRAMES, index=len(self.steps), rollout_id=step.rollout_id)

    def reset(self) -> None:
        self.resets += 1


class RecordingRunner:
    """Stands in for ``DistributedRunner``: records its construction and ``start()``."""

    instances: ClassVar[list[RecordingRunner]] = []

    def __init__(self, worker_cls: type, *, world_size: int, load_kwargs: dict[str, Any]) -> None:
        self.worker_cls = worker_cls
        self.world_size = world_size
        self.load_kwargs = load_kwargs
        self.started = False
        self.resets = 0
        RecordingRunner.instances.append(self)

    def start(self) -> None:
        self.started = True

    def generate(self, step: Step) -> FlashDreamsResult:
        return FlashDreamsResult(frames=_FRAMES, index=0, rollout_id=step.rollout_id)

    def reset(self) -> None:
        self.resets += 1


class FakeDesc:
    frames_per_second_for_step = 60


class Action2VApplication:
    """The FlashDreams family application class ``load()`` checks the slug against."""

    def __init__(self) -> None:
        self.defaults = types.SimpleNamespace(total_blocks=10_000)
        self.pipeline_config = object()
        self.env_at_create: dict[str, str | None] = {}

    def session_desc(self) -> FakeDesc:
        return FakeDesc()


class T2VApplication:
    """Another family's application, with the same surface, for the mismatch test."""

    def __init__(self) -> None:
        self.defaults = types.SimpleNamespace(total_blocks=1)
        self.pipeline_config = object()
        self.env_at_create: dict[str, str | None] = {}

    def session_desc(self) -> FakeDesc:
        return FakeDesc()


class App(FlashDreamsApp):
    """A family class reduced to what the generic half needs from it."""

    family = "action2v"
    model_class = RecordingModel  # the model half by shape

    def __init__(self) -> None:
        super().__init__()
        self.configured: list[tuple[Any, dict[str, Any]]] = []

    def configure(self, fd_app: Any, config: dict[str, Any]) -> None:
        self.configured.append((fd_app, config))

    async def process_input(self) -> Step:
        if self.state.paused:
            raise ApplicationError("paused")
        return Step(rollout_id=self.state._rollout_id)


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(App)


@pytest.fixture
def fake_flashdreams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A fake registry, a fake ``action2v`` package, a weights root, and a fake runner."""
    registry: dict[str, Any] = {}

    def create_application(slug: str) -> Any:
        try:
            app = registry[slug]
        except KeyError:
            raise LookupError(f"No FlashDreams v2 application matches {slug!r}.") from None
        app.env_at_create = {
            name: os.environ.get(name)
            for name in ("FLASHDREAMS_CACHE_DIR", "HF_HUB_CACHE", "HF_HUB_OFFLINE")
        }
        return app

    package = types.ModuleType("flashdreams")
    runtime_v2 = types.ModuleType("flashdreams.runtime_v2")
    module = types.ModuleType("flashdreams.runtime_v2.application_registry")
    module.create_application = create_application  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "flashdreams", package)
    monkeypatch.setitem(sys.modules, "flashdreams.runtime_v2", runtime_v2)
    monkeypatch.setitem(sys.modules, "flashdreams.runtime_v2.application_registry", module)

    action2v = types.ModuleType("action2v")
    action2v.Action2VApplication = Action2VApplication  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "action2v", action2v)

    registry["action2v-fake"] = Action2VApplication()
    registry["t2v-fake"] = T2VApplication()

    weights = tmp_path / "weights"
    monkeypatch.setenv("REACTOR_WEIGHTS_PATH", str(weights))
    for name in ("FLASHDREAMS_CACHE_DIR", "HF_HUB_CACHE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(model_module, "_loguru_forwarded", True)

    RecordingRunner.instances = []
    monkeypatch.setattr(app_module, "DistributedRunner", RecordingRunner)
    monkeypatch.setattr(App, "application", None)
    monkeypatch.setattr(App, "world_size", 1)
    monkeypatch.setattr(App, "isolate", False)
    # load() pins fps on the class; keep one test's pin from leaking into the next.
    if "fps" in vars(App):
        monkeypatch.delattr(App, "fps")
    return {"registry": registry, "weights": weights}


def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(text)
    return path


class FlushCounter:
    """The media operations the output handle reaches; counts flushes."""

    def __init__(self) -> None:
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1

    def set_rate(self, fps: float) -> None: ...

    def set_depth(self, depth: int) -> None: ...


def _bound_app() -> tuple[App, RecordingModel, list[Any], FlushCounter]:
    """An app past ``load()``, with a recording model half and its sinks bound."""
    app = App()
    model = RecordingModel()
    app.engine = model
    sent: list[Any] = []
    flushes = FlushCounter()
    app._on_loop_ready()
    app.bind_output(
        broadcast=sent.append,
        addressed=lambda *args: None,
        media=lambda chunk: None,
        media_ops=flushes,  # type: ignore[ty:invalid-argument-type]  # MediaOps by shape
    )
    return app, model, sent, flushes


# -- the client contract ------------------------------------------------------


def test_the_generic_commands_are_the_two_setters_and_reset() -> None:
    assert set(ModelContract.of(FlashDreamsApp).commands) == {"set_paused", "set_seed", "reset"}
    assert set(ModelContract.of(App).commands) == {"set_paused", "set_seed", "reset"}


def test_the_one_output_track_is_main_video() -> None:
    assert list(FlashDreamsOutput.__tracks__) == ["main_video"]
    assert RolloutRestarted.name == "rollout_restarted"


# -- load ---------------------------------------------------------------------


def test_load_reads_the_slug_from_the_class(fake_flashdreams: dict[str, Any]) -> None:
    App.application = "action2v-fake"
    app = App()
    app.load(None)
    assert isinstance(app.engine, RecordingModel)
    assert app.engine.load_kwargs == {
        "application": "action2v-fake",
        "weights_root": str(fake_flashdreams["weights"]),
        "warmup_steps": 0,
    }


def test_load_reads_the_slug_and_warmup_from_the_config(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    app = App()
    app.load(_config(tmp_path, "application: action2v-fake\nwarmup_steps: 5\n"))
    assert app.engine.load_kwargs["application"] == "action2v-fake"
    assert app.engine.load_kwargs["warmup_steps"] == 5


def test_the_class_slug_wins_over_the_config(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    App.application = "action2v-fake"
    app = App()
    app.load(_config(tmp_path, "application: t2v-fake\n"))
    assert app.engine.load_kwargs["application"] == "action2v-fake"


def test_load_without_a_slug_says_where_to_set_one(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=r"config\.yml"):
        App().load(None)
    with pytest.raises(ValueError, match=r"config\.yml"):
        App().load(_config(tmp_path, "warmup_steps: 1\n"))


def test_load_rejects_a_slug_of_another_family(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match="T2VApplication, not a action2v"):
        App().load(_config(tmp_path, "application: t2v-fake\n"))


def test_load_without_the_family_package_names_it(
    fake_flashdreams: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "action2v")
    with pytest.raises(ModuleNotFoundError, match="flashdreams-action2v"):
        App().load(_config(tmp_path, "application: action2v-fake\n"))


def test_load_points_the_caches_at_the_weights_root_before_flashdreams_runs(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    app = App()
    app.load(_config(tmp_path, "application: action2v-fake\n"))
    weights = fake_flashdreams["weights"]
    assert fake_flashdreams["registry"]["action2v-fake"].env_at_create == {
        "FLASHDREAMS_CACHE_DIR": str(weights),
        "HF_HUB_CACHE": str(weights / "huggingface"),
        "HF_HUB_OFFLINE": "1",
    }


def test_load_pins_playout_to_the_adapters_rate(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    app = App()
    assert not _fps_is_author_pinned(App)
    app.load(_config(tmp_path, "application: action2v-fake\n"))
    assert App.fps == 60.0
    assert _fps_is_author_pinned(App)


def test_load_calls_configure_with_the_adapter_and_the_config(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    app = App()
    app.load(_config(tmp_path, "application: action2v-fake\nexample_image: true\n"))
    ((fd_app, config),) = app.configured
    assert fd_app is fake_flashdreams["registry"]["action2v-fake"]
    assert config == {"application": "action2v-fake", "example_image": True}
    # The adapter is read before the model half is built.
    assert app.engine.load_kwargs is not None


def test_isolate_runs_the_model_half_behind_a_runner_on_one_gpu(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    App.isolate = True
    app = App()
    app.load(_config(tmp_path, "application: action2v-fake\nwarmup_steps: 2\n"))
    (runner,) = RecordingRunner.instances
    assert app.engine is runner
    assert runner.worker_cls is RecordingModel
    assert runner.world_size == 1
    assert runner.started
    assert runner.load_kwargs == {
        "application": "action2v-fake",
        "weights_root": str(fake_flashdreams["weights"]),
        "warmup_steps": 2,
    }


def test_world_size_above_one_runs_one_model_process_per_gpu(
    fake_flashdreams: dict[str, Any], tmp_path: Path
) -> None:
    App.world_size = 2
    app = App()
    app.load(_config(tmp_path, "application: action2v-fake\n"))
    (runner,) = RecordingRunner.instances
    assert runner.world_size == 2
    assert runner.started


def test_generate_forwards_to_the_model_half() -> None:
    app, model, _, _ = _bound_app()
    result = app.generate(Step(rollout_id=3, control="w"))
    assert model.steps == [Step(rollout_id=3, control="w")]
    assert result.rollout_id == 3


# -- process_output -----------------------------------------------------------


async def test_frames_go_to_main_video_tagged_with_rollout_and_index() -> None:
    app, _, sent, _ = _bound_app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    result = FlashDreamsResult(frames=_FRAMES, index=7, rollout_id=1)
    output = await app.process_output(StepOutcome(result=result, elapsed=0.01))
    assert isinstance(output, FlashDreamsOutput)
    assert output.__metadata__["main_video"] == [{"rollout": 1, "index": 7}] * 4
    assert app.state._applied_rollout_id == 1
    assert sent == []


async def test_a_new_rollouts_first_frames_flush_the_old_ones() -> None:
    app, _, _, flushes = _bound_app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    for index, rollout in enumerate((1, 1, 1, 2, 2)):
        result = FlashDreamsResult(frames=_FRAMES, index=index, rollout_id=rollout)
        await app.process_output(StepOutcome(result=result))
    # Once for rollout 1's first frames, once for rollout 2's.
    assert flushes.flushes == 2
    assert app.state._applied_rollout_id == 2


async def test_an_exhausted_rollout_restarts_and_tells_the_client() -> None:
    app, _, sent, flushes = _bound_app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    app.state._rollout_id = 1
    output = await app.process_output(StepOutcome(error=RolloutExhausted(10_000)))
    assert output is None
    assert app.state._rollout_id == 2
    (message,) = sent
    assert isinstance(message, RolloutRestarted)
    assert message.steps == 10_000
    assert flushes.flushes == 0


async def test_an_unsupported_prompt_swap_starts_a_new_rollout_quietly() -> None:
    app, _, sent, _ = _bound_app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    app.state._rollout_id = 1
    output = await app.process_output(StepOutcome(error=PromptSwapUnsupported()))
    assert output is None
    assert app.state._rollout_id == 2
    assert sent == []


async def test_any_other_error_is_reraised() -> None:
    app, _, _, _ = _bound_app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    with pytest.raises(RolloutNotStarted):
        await app.process_output(StepOutcome(error=RolloutNotStarted("no seed image")))
    with pytest.raises(RuntimeError, match="CUDA"):
        await app.process_output(StepOutcome(error=RuntimeError("CUDA error")))


# -- reset and the session end ------------------------------------------------


async def test_reset_asks_for_a_new_rollout_and_the_next_step_carries_it() -> None:
    app, model, _, _ = _bound_app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    first = await app.process_input()
    app.reset()
    second = await app.process_input()
    assert second.rollout_id == first.rollout_id + 1
    # The model half is not touched: the new id reaches it on the next step.
    assert model.resets == 0


async def test_the_session_end_resets_the_model_half() -> None:
    app, model, _, _ = _bound_app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert model.resets == 1
    assert app.state is None


async def test_a_subclass_extends_the_session_end_with_super() -> None:
    class Extended(App):
        def __init__(self) -> None:
            super().__init__()
            self.ended = 0

        def end(self) -> None:
            super().end()
            self.ended += 1

    app = Extended()
    model = RecordingModel()
    app.engine = model
    app._on_loop_ready()
    app.bind_output(broadcast=lambda m: None, addressed=lambda *a: None, media=lambda c: None)
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert (model.resets, app.ended) == (1, 1)


async def test_the_state_starts_each_session_with_the_handshake_reset() -> None:
    app, _, _, _ = _bound_app()
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert (app.state.paused, app.state.seed) == (False, 42)
    assert (app.state._rollout_id, app.state._applied_rollout_id) == (0, None)
