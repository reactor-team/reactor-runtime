"""The action2v family, driven through its hooks with fake adapter defaults.

A fake ``action2v`` package supplies ``ActionSnapshot``; fake adapter defaults
supply the action mapper, the seed loader, and the example-image resolver; a
fake pipeline records what ``initialize_cache()`` asks of it. No FlashDreams,
no torch, no GPU.
"""

from __future__ import annotations

import io
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile
from reactor_runtime.core.model import SessionStarted
from reactor_runtime.flashdreams import (
    FlashDreamsOutput,
    FlashDreamsResult,
    RolloutNotStarted,
    RolloutRestarted,
)
from reactor_runtime.flashdreams.action2v import (
    Action2V,
    Action2VInput,
    Action2VModel,
    Action2VState,
)
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class

_FRAMES = np.zeros((4, 8, 8, 3), dtype=np.uint8)


@dataclass(frozen=True, kw_only=True)
class ActionSnapshot:
    """FlashDreams' model-neutral snapshot, by shape."""

    keys: frozenset[str] = field(default_factory=frozenset)
    mouse_buttons: frozenset[int] = field(default_factory=frozenset)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    wheel_x: float = 0.0
    wheel_y: float = 0.0


class RecordingMapper:
    """The adapter's action mapper: records each snapshot and returns a control."""

    def __init__(self, desc: Any, sensitivity: float) -> None:
        self.desc = desc
        self.sensitivity = sensitivity
        self.snapshots: list[ActionSnapshot] = []

    def __call__(self, snapshot: ActionSnapshot) -> dict[str, Any]:
        self.snapshots.append(snapshot)
        return {"keys": sorted(snapshot.keys), "dx": snapshot.mouse_dx}


class FakeDesc:
    frames_per_second_for_step = 60


class FakeDefaults:
    def __init__(self, example: Path) -> None:
        self.total_blocks = 10_000
        self.example = example
        self.resolved: list[dict[str, Any]] = []
        self.loaded: list[tuple[Path, bytes]] = []
        self.mappers: list[RecordingMapper] = []

    def input_resolver(self, values: dict[str, Any]) -> Path:
        self.resolved.append(dict(values))
        return self.example

    def seed_loader(self, path: Path, desc: Any) -> FakeTensor:
        self.loaded.append((path, path.read_bytes()))
        return FakeTensor()

    def action_mapper_factory(self, desc: Any, sensitivity: float) -> RecordingMapper:
        mapper = RecordingMapper(desc, sensitivity)
        self.mappers.append(mapper)
        return mapper


class FakeApp:
    def __init__(self, example: Path) -> None:
        self.defaults = FakeDefaults(example)
        self.pipeline_config = object()

    def session_desc(self) -> FakeDesc:
        return FakeDesc()


class FakeTensor:
    """Records the tensor chain ``initialize_cache()`` walks on the seed frames."""

    def __init__(self) -> None:
        self.ops: list[tuple[str, tuple[Any, ...]]] = []

    def to(self, *args: Any) -> FakeTensor:
        self.ops.append(("to", args))
        return self

    def add(self, value: float) -> FakeTensor:
        self.ops.append(("add", (value,)))
        return self

    def mul(self, value: float) -> FakeTensor:
        self.ops.append(("mul", (value,)))
        return self

    def unsqueeze(self, dim: int) -> FakeTensor:
        self.ops.append(("unsqueeze", (dim,)))
        return self


class FakeRng:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> FakeRng:
        self.seeds.append(seed)
        return self


class FakePipeline:
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.diffusion_model = types.SimpleNamespace(dtype="bf16", rng=FakeRng())
        self.caches: list[Any] = []

    def initialize_cache(self, *, seed_pixels: Any) -> dict[str, Any]:
        cache = {"seed_pixels": seed_pixels, "autoregressive_index": 0}
        self.caches.append(cache)
        return cache


class RecordingModel:
    """A model half that records steps and answers with scripted results."""

    def __init__(self) -> None:
        self.steps: list[Action2VInput] = []
        self.resets = 0

    def generate(self, step: Action2VInput) -> FlashDreamsResult:
        self.steps.append(step)
        return FlashDreamsResult(frames=_FRAMES, index=len(self.steps), rollout_id=step.rollout_id)

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None,
    register_model: Callable[[type], None],
    register: Callable[..., None],
) -> None:
    register_model(Action2V)
    # The track and the message are declared in app.py, outside the family module.
    register(FlashDreamsOutput, RolloutRestarted)


@pytest.fixture
def fake_action2v(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("action2v")
    module.ActionSnapshot = ActionSnapshot  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "action2v", module)


def _png(size: tuple[int, int] = (32, 16), mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, 7).save(buffer, format="PNG")
    return buffer.getvalue()


def _configured(tmp_path: Path, example_image: bool = True) -> tuple[Action2V, FakeApp]:
    """An app past ``configure()``, with a recording model half and its sinks bound."""
    example = tmp_path / "example.png"
    example.write_bytes(_png())
    fd_app = FakeApp(example)
    app = Action2V()
    app.configure(fd_app, {"example_image": example_image})
    app.engine = RecordingModel()
    app._on_loop_ready()
    app.bind_output(broadcast=lambda m: None, addressed=lambda *a: None, media=lambda c: None)
    return app, fd_app


# -- the client contract ------------------------------------------------------


def test_the_commands_are_the_family_setters_plus_the_two_hand_written_ones() -> None:
    assert set(ModelContract.of(Action2V).commands) == {
        "set_paused",
        "set_seed",
        "set_keys",
        "set_image",
        "move",
        "reset",
    }
    spec = ModelContract.of(Action2V).commands["set_image"]
    assert spec.response is None
    assert "image" in spec.command.__command_fields__


def test_the_manifest_import_names_the_family_class_and_the_schema_renders() -> None:
    assert import_model_class("reactor_runtime.flashdreams.action2v:Action2V") is Action2V
    schema = ModelContract.of(Action2V).render_schema()
    assert set(schema.commands) >= {"set_keys", "move", "set_image"}
    assert "main_video" in schema.tracks
    assert "rollout_restarted" in schema.messages


def test_the_state_inherits_the_generic_fields() -> None:
    state = Action2VState()
    assert (state.paused, state.seed, state.keys) == (False, 42, "")
    assert (state._image, state._dx, state._dy, state._wheel) == (None, 0.0, 0.0, 0.0)
    assert (state._rollout_id, state._applied_rollout_id) == (0, None)


# -- configure and the session start -----------------------------------------


def test_configure_keeps_the_adapters_mapper_and_example_image(tmp_path: Path) -> None:
    app, fd_app = _configured(tmp_path, example_image=True)
    assert fd_app.defaults.resolved == [{"image_path": None, "example_data": True}]
    assert app.default_image == _png()
    (mapper,) = fd_app.defaults.mappers
    assert app.map_action is mapper
    assert (mapper.desc.frames_per_second_for_step, mapper.sensitivity) == (60, 1.0)


def test_configure_without_example_image_starts_from_nothing(tmp_path: Path) -> None:
    app, fd_app = _configured(tmp_path, example_image=False)
    assert fd_app.defaults.resolved == []
    assert app.default_image is None


async def test_a_session_starts_from_the_default_image_with_a_new_rollout(
    tmp_path: Path,
) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert app.state._image == _png()
    assert app.state._rollout_id == 1


# -- process_input ------------------------------------------------------------


async def test_process_input_refuses_while_paused(tmp_path: Path) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    app.state.paused = True
    with pytest.raises(ApplicationError, match="paused"):
        await app.process_input()


async def test_process_input_refuses_before_a_seed_image(tmp_path: Path) -> None:
    app, _ = _configured(tmp_path, example_image=False)
    await app._dispatch_reactor_event(SessionStarted("s"))
    with pytest.raises(ApplicationError, match="no seed image"):
        await app.process_input()


async def test_the_image_rides_on_the_step_only_until_the_model_holds_it(
    tmp_path: Path, fake_action2v: None
) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))

    first = await app.process_input()
    assert first.image == _png()
    assert (first.rollout_id, first.seed) == (1, 42)
    await app.process_output(StepOutcome(result=app.generate(first)))

    second = await app.process_input()
    assert second.image is None
    assert second.rollout_id == 1

    app.reset()
    third = await app.process_input()
    assert third.image == _png()
    assert third.rollout_id == 2


async def test_control_is_the_held_keys_and_the_motion_since_the_last_step(
    tmp_path: Path, fake_action2v: None
) -> None:
    app, fd_app = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    app.state.keys = "w, shift,,W "
    app.move(dx=0.1, dy=-0.2)
    app.move(dx=0.05, wheel=1.0)

    step = await app.process_input()
    (mapper,) = fd_app.defaults.mappers
    (snapshot,) = mapper.snapshots
    assert snapshot.keys == frozenset({"w", "shift", "W"})
    assert snapshot.mouse_dx == pytest.approx(0.15)
    assert snapshot.mouse_dy == pytest.approx(-0.2)
    assert snapshot.wheel_y == pytest.approx(1.0)
    assert step.control == {"keys": ["W", "shift", "w"], "dx": pytest.approx(0.15)}

    # The motion is spent; the keys are held.
    await app.process_input()
    assert mapper.snapshots[1].keys == frozenset({"w", "shift", "W"})
    assert (mapper.snapshots[1].mouse_dx, mapper.snapshots[1].wheel_y) == (0.0, 0.0)


# -- set_image ----------------------------------------------------------------


async def test_set_image_keeps_the_upload_as_rgb_and_starts_a_new_rollout(
    tmp_path: Path,
) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    upload = UploadedFile(name="seed.png", mime_type="image/png", data=_png((20, 10), mode="L"))
    await app.set_image(upload)
    assert app.state._rollout_id == 2
    assert app.state._image is not None
    with Image.open(io.BytesIO(app.state._image)) as kept:
        assert (kept.mode, kept.size) == ("RGB", (20, 10))


async def test_set_image_rejects_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    before = app.state._image
    with pytest.raises(CommandError) as excinfo:
        await app.set_image(UploadedFile(name="notes.txt", mime_type="text/plain", data=b"hi"))
    assert excinfo.value.code == "unsupported_media"
    with pytest.raises(CommandError) as excinfo:
        await app.set_image(UploadedFile(name="x.png", mime_type="image/png", data=b"not a png"))
    assert excinfo.value.code == "undecodable_image"
    assert app.state._image is before
    assert app.state._rollout_id == 1


# -- the model half -----------------------------------------------------------


def _model(tmp_path: Path) -> tuple[Action2VModel, FakeApp, FakePipeline]:
    model = Action2VModel()
    fd_app = FakeApp(tmp_path / "unused.png")
    pipeline = FakePipeline()
    model.app = fd_app
    model.desc = fd_app.session_desc()
    model.pipeline = pipeline
    model.max_blocks = 10_000
    return model, fd_app, pipeline


def test_initialize_cache_runs_the_seed_loader_on_the_image_and_seeds_the_rng(
    tmp_path: Path,
) -> None:
    model, fd_app, pipeline = _model(tmp_path)
    step = Action2VInput(rollout_id=1, image=b"png bytes", seed=7, control=None)
    cache = model.initialize_cache(step)

    ((path, contents),) = fd_app.defaults.loaded
    assert contents == b"png bytes"
    assert not path.exists()  # the temporary file is gone once the loader has read it
    assert pipeline.diffusion_model.rng.seeds == [7]
    (recorded,) = pipeline.caches
    assert cache is recorded
    frames = recorded["seed_pixels"]
    assert frames.ops == [
        ("to", ("cuda:0", "bf16")),
        ("add", (1,)),
        ("mul", (0.5,)),
        ("unsqueeze", (0,)),
    ]


def test_initialize_cache_without_an_image_is_the_models_own_error(tmp_path: Path) -> None:
    model, fd_app, pipeline = _model(tmp_path)
    with pytest.raises(RolloutNotStarted):
        model.initialize_cache(Action2VInput(rollout_id=1, image=None, seed=7, control=None))
    assert fd_app.defaults.loaded == []
    assert pipeline.caches == []
