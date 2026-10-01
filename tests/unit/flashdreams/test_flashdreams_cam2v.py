"""The cam2v family, driven through its hooks with fake adapter defaults.

Fake adapter defaults supply the conditioning resolver and the pose
integrator; a fake ``cam2v`` package supplies ``CameraControlInput``; a fake
torch and a fake first-frame loader stand in for the two GPU-side imports the
model half makes. No FlashDreams, no torch, no GPU.
"""

from __future__ import annotations

import io
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
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
from reactor_runtime.flashdreams.cam2v import Cam2V, Cam2VInput, Cam2VModel, Cam2VState
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class

_FRAMES = np.zeros((12, 8, 8, 3), dtype=np.uint8)
_INTRINSICS = (400.0, 400.0, 416.0, 232.0)


@dataclass(frozen=True, kw_only=True)
class CameraControlInput:
    """FlashDreams' per-step camera payload, by shape."""

    intrinsics: Any
    poses: Any
    world_scale: float


class FakeTensor:
    """Enough of a tensor for the model half's camera-input arithmetic."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)
        self.device: Any = None
        self.dtype: Any = None

    def reshape(self, *shape: int) -> FakeTensor:
        return FakeTensor(self.array.reshape(*shape))

    def tolist(self) -> list[Any]:
        return self.array.tolist()

    def repeat(self, *reps: int) -> FakeTensor:
        return FakeTensor(np.tile(self.array, reps))

    def to(self, *, device: Any, dtype: Any) -> FakeTensor:
        out = FakeTensor(self.array)
        out.device, out.dtype = device, dtype
        return out


class FakeDesc:
    frames_per_second_for_step = 16
    video_width = 832
    video_height = 464


class FakeConditioning:
    def __init__(self, prompt: str, first_frame_path: Path, world_scale: float) -> None:
        self.prompt = prompt
        self.first_frame_path = first_frame_path
        self.base_intrinsics = FakeTensor(np.asarray(_INTRINSICS, dtype=np.float32).reshape(1, 4))
        self.world_scale = world_scale


class FakeIntegrator:
    """The adapter's pose integrator: records the chunks and returns identity poses."""

    def __init__(self) -> None:
        self.chunks: list[tuple[list[tuple[float, float, frozenset[str]]], list[float]]] = []

    def integrate_chunk(self, *, segments: list, frame_times: list[float]) -> np.ndarray:
        self.chunks.append((segments, frame_times))
        return np.tile(np.eye(4, dtype=np.float32), (len(frame_times), 1, 1))


class FakeDefaults:
    def __init__(self, example: Path) -> None:
        self.total_blocks = 20
        self.first_frame_dtype = "bf16"
        self.first_frame_interpolation = "cubic"
        self.install_hint = "pip install the model"
        self.example = example
        self.resolved: list[dict[str, Any]] = []
        self.integrators: list[FakeIntegrator] = []
        self.fail_with: Exception | None = None

    def input_resolver(self, values: dict[str, Any]) -> FakeConditioning:
        self.resolved.append(dict(values))
        if self.fail_with is not None:
            raise self.fail_with
        return FakeConditioning("a sunlit valley", self.example, 0.25)

    def pose_integrator_factory(self) -> FakeIntegrator:
        integrator = FakeIntegrator()
        self.integrators.append(integrator)
        return integrator


class FakeApp:
    def __init__(self, example: Path) -> None:
        self.defaults = FakeDefaults(example)
        self.pipeline_config = object()

    def session_desc(self) -> FakeDesc:
        return FakeDesc()


class FakeRng:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> FakeRng:
        self.seeds.append(seed)
        return self


class FakePipeline:
    def __init__(self, rng: FakeRng | None) -> None:
        self.device = "cuda:0"
        self.diffusion_model = types.SimpleNamespace(rng=rng)
        self.caches: list[dict[str, Any]] = []
        self.frame_counts = {0: 9}

    def get_num_output_frames(self, index: int) -> int:
        return self.frame_counts.get(index, 12)

    def initialize_cache(self, *, text: list[str], image: Any) -> dict[str, Any]:
        cache = {"text": text, "image": image, "autoregressive_index": None}
        self.caches.append(cache)
        return cache


class RecordingModel:
    def __init__(self) -> None:
        self.steps: list[Cam2VInput] = []
        self.resets = 0

    def generate(self, step: Cam2VInput) -> FlashDreamsResult:
        self.steps.append(step)
        return FlashDreamsResult(
            frames=_FRAMES, index=len(self.steps) - 1, rollout_id=step.rollout_id
        )

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None,
    register_model: Callable[[type], None],
    register: Callable[..., None],
) -> None:
    register_model(Cam2V)
    register(FlashDreamsOutput, RolloutRestarted)


@pytest.fixture
def fake_gpu_side(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for torch, FlashDreams' first-frame loader, and the cam2v package."""
    record: dict[str, Any] = {"global_seeds": [], "loads": []}

    torch = types.ModuleType("torch")
    torch.float32 = "float32"  # type: ignore[ty:unresolved-attribute]
    torch.device = lambda name: f"device({name})"  # type: ignore[ty:unresolved-attribute]
    torch.as_tensor = lambda array: FakeTensor(np.asarray(array))  # type: ignore[ty:unresolved-attribute]
    torch.from_numpy = lambda array: FakeTensor(array)  # type: ignore[ty:unresolved-attribute]
    torch.manual_seed = record["global_seeds"].append  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "torch", torch)

    def load_first_frame_tensor(path: Path, **kwargs: Any) -> str:
        record["loads"].append({"bytes": Path(path).read_bytes(), **kwargs})
        return "first-frame-tensor"

    flashdreams = types.ModuleType("flashdreams")
    infra = types.ModuleType("flashdreams.infra")
    runner_io = types.ModuleType("flashdreams.infra.runner_io")
    runner_io.load_first_frame_tensor = load_first_frame_tensor  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "flashdreams", flashdreams)
    monkeypatch.setitem(sys.modules, "flashdreams.infra", infra)
    monkeypatch.setitem(sys.modules, "flashdreams.infra.runner_io", runner_io)

    cam2v = types.ModuleType("cam2v")
    cam2v.CameraControlInput = CameraControlInput  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "cam2v", cam2v)
    return record


def _png(size: tuple[int, int] = (32, 16), mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, 7).save(buffer, format="PNG")
    return buffer.getvalue()


def _configured(tmp_path: Path, config: dict[str, Any] | None = None) -> tuple[Cam2V, FakeApp]:
    example = tmp_path / "image.jpg"
    example.write_bytes(_png())
    fd_app = FakeApp(example)
    app = Cam2V()
    app.configure(fd_app, {"example_data": True, "example_idx": 0} if config is None else config)
    app.engine = RecordingModel()
    app._on_loop_ready()
    app.bind_output(broadcast=lambda m: None, addressed=lambda *a: None, media=lambda c: None)
    return app, fd_app


def _step(**overrides: Any) -> Cam2VInput:
    values: dict[str, Any] = {
        "rollout_id": 1,
        "prompt": "a sunlit valley",
        "image": b"jpeg bytes",
        "seed": 7,
        "keys": frozenset({"w"}),
        "intrinsics": _INTRINSICS,
        "world_scale": 0.25,
    }
    values.update(overrides)
    return Cam2VInput(**values)


def _model(tmp_path: Path, rng: FakeRng | None = None) -> tuple[Cam2VModel, FakeApp, FakePipeline]:
    model = Cam2VModel()
    fd_app = FakeApp(tmp_path / "unused.jpg")
    pipeline = FakePipeline(rng if rng is not None else FakeRng())
    model.app = fd_app
    model.desc = fd_app.session_desc()
    model.pipeline = pipeline
    model.device = "cuda:0"
    model.max_blocks = 20
    return model, fd_app, pipeline


# -- the client contract ------------------------------------------------------


def test_the_commands_are_the_family_setters_plus_set_image() -> None:
    assert set(ModelContract.of(Cam2V).commands) == {
        "set_paused",
        "set_seed",
        "set_prompt",
        "set_keys",
        "set_image",
        "reset",
    }
    assert Cam2VState._public_fields["prompt"].moderate is True


def test_the_manifest_import_names_the_family_class_and_the_schema_renders() -> None:
    assert import_model_class("reactor_runtime.flashdreams.cam2v:Cam2V") is Cam2V
    schema = ModelContract.of(Cam2V).render_schema()
    assert set(schema.commands) >= {"set_prompt", "set_keys", "set_image"}
    assert "main_video" in schema.tracks
    assert "rollout_restarted" in schema.messages


# -- configure and the session start -----------------------------------------


def test_configure_resolves_the_conditioning_from_the_config_and_the_adapters_rates(
    tmp_path: Path,
) -> None:
    app, fd_app = _configured(
        tmp_path, {"example_data": True, "example_idx": 2, "world_scale": 0.5, "warmup_steps": 3}
    )
    (values,) = fd_app.defaults.resolved
    assert values == {
        "example_data": True,
        "example_idx": 2,
        "world_scale": 0.5,
        "pixel_height": 464,
        "pixel_width": 832,
        "fps": 16,
    }
    assert app.default_prompt == "a sunlit valley"
    assert app.default_image == _png()
    assert app.intrinsics == _INTRINSICS
    assert app.world_scale == 0.25


def test_configure_fails_with_the_reason_when_the_adapter_cannot_resolve(tmp_path: Path) -> None:
    fd_app = FakeApp(tmp_path / "absent.jpg")
    fd_app.defaults.fail_with = FileNotFoundError("Lingbot Cam2V missing intrinsic_path")
    with pytest.raises(RuntimeError, match="example_data") as excinfo:
        Cam2V().configure(fd_app, {})
    assert "intrinsic_path" in str(excinfo.value.__cause__)


async def test_a_session_starts_from_the_adapters_prompt_frame_and_calibration(
    tmp_path: Path,
) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert app.state.prompt == "a sunlit valley"
    assert app.state._image == _png()
    assert app.state._intrinsics == _INTRINSICS
    assert app.state._world_scale == 0.25
    assert app.state._rollout_id == 1


# -- process_input ------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda s: setattr(s, "paused", True), "paused"),
        (lambda s: setattr(s, "prompt", "  "), "no prompt set"),
        (lambda s: setattr(s, "_image", None), "no first frame"),
        (lambda s: setattr(s, "_intrinsics", None), "no camera calibration"),
    ],
)
async def test_process_input_refuses(
    tmp_path: Path, change: Callable[[Any], None], reason: str
) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    change(app.state)
    with pytest.raises(ApplicationError, match=reason):
        await app.process_input()


async def test_the_step_carries_the_prompt_keys_and_calibration_and_the_image_until_held(
    tmp_path: Path,
) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    app.state.keys = "W, a ,,q"
    app.state.prompt = "  a storm over the sea "

    first = await app.process_input()
    assert first == Cam2VInput(
        rollout_id=1,
        prompt="a storm over the sea",
        image=_png(),
        seed=42,
        keys=frozenset({"w", "a", "q"}),
        intrinsics=_INTRINSICS,
        world_scale=0.25,
    )
    await app.process_output(StepOutcome(result=app.generate(first)))

    second = await app.process_input()
    assert second.image is None
    assert second.rollout_id == 1

    app.reset()
    assert (await app.process_input()).image == _png()


# -- set_image ----------------------------------------------------------------


async def test_set_image_keeps_the_upload_as_rgb_and_starts_a_new_rollout(tmp_path: Path) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app.set_image(UploadedFile(name="f.png", mime_type="image/png", data=_png((20, 10), "L")))
    assert app.state._rollout_id == 2
    assert app.state._image is not None
    with Image.open(io.BytesIO(app.state._image)) as kept:
        assert (kept.mode, kept.size) == ("RGB", (20, 10))


async def test_set_image_rejects_what_it_cannot_use(tmp_path: Path) -> None:
    app, _ = _configured(tmp_path)
    await app._dispatch_reactor_event(SessionStarted("s"))
    before = app.state._image
    cases = [
        (UploadedFile(name="n.txt", mime_type="text/plain", data=b"hi"), "unsupported_media"),
        (UploadedFile(name="x.png", mime_type="image/png", data=b"not a png"), "undecodable_image"),
        (
            UploadedFile(name="h.png", mime_type="image/png", data=_png((5000, 5000), "L")),
            "image_too_large",
        ),
    ]
    for upload, code in cases:
        with pytest.raises(CommandError) as excinfo:
            await app.set_image(upload)
        assert excinfo.value.code == code
    assert app.state._image is before
    assert app.state._rollout_id == 1


# -- the model half -----------------------------------------------------------


def test_initialize_cache_loads_the_frame_seeds_the_rng_and_starts_the_camera(
    tmp_path: Path, fake_gpu_side: dict[str, Any]
) -> None:
    model, fd_app, pipeline = _model(tmp_path)
    cache = model.initialize_cache(_step())

    (load,) = fake_gpu_side["loads"]
    assert load["bytes"] == b"jpeg bytes"
    assert (load["pixel_height"], load["pixel_width"]) == (464, 832)
    assert load["device"] == "device(cuda:0)"
    assert (load["dtype"], load["interpolation"]) == ("bf16", "cubic")
    assert load["install_hint"] == "pip install the model"
    assert pipeline.diffusion_model.rng.seeds == [7]
    assert fake_gpu_side["global_seeds"] == []
    (recorded,) = pipeline.caches
    assert cache is recorded
    assert recorded["text"] == ["a sunlit valley"]
    assert recorded["image"] == "first-frame-tensor"
    assert len(fd_app.defaults.integrators) == 1
    assert model.clock == 0.0
    assert model.world_scale == 0.25
    np.testing.assert_array_equal(model.intrinsics, np.asarray(_INTRINSICS, dtype=np.float32))


def test_initialize_cache_seeds_torch_when_the_pipeline_has_no_generator(
    tmp_path: Path, fake_gpu_side: dict[str, Any]
) -> None:
    model, _, pipeline = _model(tmp_path)
    pipeline.diffusion_model.rng = None
    model.initialize_cache(_step(seed=11))
    assert fake_gpu_side["global_seeds"] == [11]


@pytest.mark.parametrize(
    ("step", "reason"),
    [(_step(image=None), "no first frame"), (_step(prompt=""), "no prompt")],
)
def test_initialize_cache_without_conditioning_is_the_models_own_error(
    tmp_path: Path, fake_gpu_side: dict[str, Any], step: Cam2VInput, reason: str
) -> None:
    model, _, pipeline = _model(tmp_path)
    with pytest.raises(RolloutNotStarted, match=reason):
        model.initialize_cache(step)
    assert pipeline.caches == []
    assert fake_gpu_side["loads"] == []


def test_the_pipeline_input_holds_the_keys_for_one_period_per_output_frame(
    tmp_path: Path, fake_gpu_side: dict[str, Any]
) -> None:
    model, fd_app, _ = _model(tmp_path)
    model.initialize_cache(_step())
    (integrator,) = fd_app.defaults.integrators

    first = model._pipeline_input(_step(keys=frozenset({"w", "a"})), 0)
    second = model._pipeline_input(_step(keys=frozenset()), 1)

    period = 1 / 16
    (segments, times), (segments2, times2) = integrator.chunks
    # Step 0 produces 9 frames, step 1 twelve; each frame is one period later,
    # the keys are held for the whole chunk, and step 1 starts where step 0 ended.
    assert times == pytest.approx([period * (k + 1) for k in range(9)])
    assert segments == [(0.0, pytest.approx(9 * period), frozenset({"w", "a"}))]
    assert times2[0] == pytest.approx(9 * period + period)
    assert segments2 == [(pytest.approx(9 * period), pytest.approx(21 * period), frozenset())]
    assert model.clock == pytest.approx(21 * period)

    assert isinstance(first, CameraControlInput)
    assert first.poses.array.shape == (9, 4, 4)
    assert first.intrinsics.array.shape == (9, 4)
    np.testing.assert_array_equal(first.intrinsics.array[3], np.asarray(_INTRINSICS))
    assert (first.intrinsics.device, first.intrinsics.dtype) == ("cuda:0", "float32")
    assert (first.poses.device, first.poses.dtype) == ("cuda:0", "float32")
    assert first.world_scale == 0.25
    assert second.poses.array.shape == (12, 4, 4)


def test_reset_forgets_the_camera_with_the_rollout(
    tmp_path: Path, fake_gpu_side: dict[str, Any]
) -> None:
    model, _, _ = _model(tmp_path)
    model.initialize_cache(_step())
    model._pipeline_input(_step(), 0)
    model.reset()
    assert model.pose_integrator is None
    assert model.clock == 0.0
    assert model.intrinsics is None
    assert model.cache is None
