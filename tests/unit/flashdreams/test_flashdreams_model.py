"""The generic FlashDreams model half against a fake registry and a fake pipeline.

No FlashDreams, no torch, no GPU: the fakes below stand in for the two imports
``FlashDreamsModel`` makes inside its methods, so what is tested is the rollout
bookkeeping around the pipeline, not the pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from reactor_runtime.flashdreams import (
    FlashDreamsModel,
    FlashDreamsResult,
    PromptSwapUnsupported,
    RolloutExhausted,
    RolloutNotStarted,
)
from reactor_runtime.flashdreams import model as model_module

_FRAMES = np.zeros((4, 8, 16, 3), dtype=np.uint8)


@dataclass(frozen=True)
class Step:
    rollout_id: int
    image: bytes | None = None
    prompt: str | None = None
    control: Any = None


class FakeCache:
    def __init__(self, autoregressive_index: int | None) -> None:
        self.autoregressive_index = autoregressive_index


class FakeVideo:
    """The tensor chain ``_to_frames`` walks, ending in a preset array."""

    ndim = 5

    def __getitem__(self, item: Any) -> FakeVideo:
        return self

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: self

    def numpy(self) -> np.ndarray:
        return _FRAMES


class FakePipeline:
    """Records the pipeline calls the model half makes and advances the cache."""

    def __init__(self) -> None:
        self.generated: list[tuple[int, Any, Any]] = []
        self.finalized: list[int] = []
        self.fail_on: int | None = None

    def generate(self, autoregressive_index: int, cache: Any, input: Any = None) -> FakeVideo:
        if autoregressive_index == self.fail_on:
            raise RuntimeError("CUDA error: device-side assert")
        cache.autoregressive_index = autoregressive_index
        self.generated.append((autoregressive_index, cache, input))
        return FakeVideo()

    def finalize(self, autoregressive_index: int, cache: Any) -> None:
        self.finalized.append(autoregressive_index)


class FakePipelineConfig:
    def __init__(self, pipeline: FakePipeline) -> None:
        self.pipeline = pipeline
        self.device: str | None = None

    def setup(self) -> FakePipelineConfig:
        return self

    def to(self, device: str) -> FakePipelineConfig:
        self.device = device
        return self

    def eval(self) -> FakePipeline:
        return self.pipeline


class FakeDefaults:
    def __init__(self, total_blocks: int) -> None:
        self.total_blocks = total_blocks


class FakeDesc:
    frames_per_second_for_step = 60


class FakeApp:
    """The adapter surface ``load()`` reads off a FlashDreams family application."""

    def __init__(self, pipeline: FakePipeline, total_blocks: int = 10_000) -> None:
        self.defaults = FakeDefaults(total_blocks)
        self.pipeline_config = FakePipelineConfig(pipeline)

    def session_desc(self) -> FakeDesc:
        return FakeDesc()


class SeededModel(FlashDreamsModel):
    """A family's model half: starts a rollout from the step's image."""

    def __init__(self, seed_index: int | None = 0) -> None:
        super().__init__()
        self.seed_index = seed_index
        self.started: list[Step] = []

    def initialize_cache(self, step: Step) -> FakeCache:
        if step.image is None:
            raise RolloutNotStarted("no seed image on the step that starts the rollout")
        self.started.append(step)
        return FakeCache(self.seed_index)


class SwappingModel(SeededModel):
    """A model half that can change its prompt within a rollout."""

    def __init__(self) -> None:
        super().__init__()
        self.swapped: list[str | None] = []

    def _replace_prompt(self, step: Step) -> None:
        self.swapped.append(step.prompt)


@pytest.fixture
def fake_flashdreams(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for FlashDreams' registry and for torch, and reset the loguru forward."""
    pipeline = FakePipeline()
    app = FakeApp(pipeline)
    registry: dict[str, Any] = {"action2v-fake": app}

    package = types.ModuleType("flashdreams")
    runtime_v2 = types.ModuleType("flashdreams.runtime_v2")
    module = types.ModuleType("flashdreams.runtime_v2.application_registry")

    def create_application(slug: str) -> Any:
        try:
            return registry[slug]
        except KeyError:
            raise LookupError(f"No FlashDreams v2 application matches {slug!r}.") from None

    module.create_application = create_application  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "flashdreams", package)
    monkeypatch.setitem(sys.modules, "flashdreams.runtime_v2", runtime_v2)
    monkeypatch.setitem(sys.modules, "flashdreams.runtime_v2.application_registry", module)

    torch = types.ModuleType("torch")
    torch.uint8 = "uint8"  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "torch", torch)

    monkeypatch.setattr(model_module, "_loguru_forwarded", False)
    for name in ("FLASHDREAMS_CACHE_DIR", "HF_HUB_CACHE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    return {"pipeline": pipeline, "app": app, "registry": registry}


def _loaded(
    seed_index: int | None = 0, total_blocks: int = 10_000
) -> tuple[SeededModel, FakePipeline]:
    """A model half past ``load()``, with the pipeline set by hand."""
    model = SeededModel(seed_index)
    pipeline = FakePipeline()
    model.pipeline = pipeline
    model.max_blocks = total_blocks
    return model, pipeline


# -- load ---------------------------------------------------------------------


def test_load_resolves_the_slug_and_builds_the_pipeline_on_the_device(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    model = SeededModel()
    model.load("action2v-fake", weights_root=str(tmp_path), device="cuda:1")
    assert model.app is fake_flashdreams["app"]
    assert model.pipeline is fake_flashdreams["pipeline"]
    assert fake_flashdreams["app"].pipeline_config.device == "cuda:1"
    assert model.max_blocks == 10_000
    assert model.desc.frames_per_second_for_step == 60
    assert model.cache is None
    assert model.rollout_id is None


def test_load_points_both_caches_at_the_weights_root(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    SeededModel().load("action2v-fake", weights_root=str(tmp_path))
    assert os.environ["FLASHDREAMS_CACHE_DIR"] == str(tmp_path)
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "huggingface")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_load_keeps_an_offline_flag_the_environment_already_sets(
    fake_flashdreams: dict[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    SeededModel().load("action2v-fake", weights_root=str(tmp_path))
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_load_forwards_loguru_to_logging_once(
    fake_flashdreams: dict[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    sinks: list[Any] = []

    class FakeLoguru:
        def remove(self) -> None:
            calls.append("remove")

        def add(self, sink: Any, format: str) -> None:
            calls.append("add")
            sinks.append(sink)

    loguru = types.ModuleType("loguru")
    loguru.logger = FakeLoguru()  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "loguru", loguru)

    SeededModel().load("action2v-fake", weights_root=str(tmp_path))
    SeededModel().load("action2v-fake", weights_root=str(tmp_path))
    assert calls == ["remove", "add"]

    received: list[logging.LogRecord] = []
    target = logging.getLogger("flashdreams.test")
    handler = logging.Handler()
    handler.emit = received.append  # type: ignore[ty:invalid-assignment]
    target.addHandler(handler)
    try:
        record = logging.LogRecord("flashdreams.test", logging.INFO, __file__, 1, "hello", (), None)
        sinks[0].emit(record)
    finally:
        target.removeHandler(handler)
    assert [r.getMessage() for r in received] == ["hello"]


def test_load_without_flashdreams_names_the_packages_to_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    for name in list(sys.modules):
        if name == "flashdreams" or name.startswith("flashdreams."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(model_module, "_loguru_forwarded", True)
    with pytest.raises(ModuleNotFoundError, match="flashdreams-action2v"):
        SeededModel().load("action2v-fake", weights_root=str(tmp_path))


def test_load_with_an_unknown_slug_is_the_registrys_error(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    with pytest.raises(LookupError, match="no-such-model"):
        SeededModel().load("no-such-model", weights_root=str(tmp_path))


def test_load_rejects_an_application_without_the_adapter_surface(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    class Bare:
        def session_desc(self) -> None:
            return None

    fake_flashdreams["registry"]["bare"] = Bare()
    with pytest.raises(TypeError, match="defaults, pipeline_config"):
        SeededModel().load("bare", weights_root=str(tmp_path))


def test_load_refuses_warmup_steps_on_a_model_that_does_not_warm_up(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    with pytest.raises(NotImplementedError, match="warmup_steps"):
        SeededModel().load("action2v-fake", weights_root=str(tmp_path), warmup_steps=3)


# -- generate -----------------------------------------------------------------


def test_generate_before_load_is_a_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        SeededModel().generate(Step(rollout_id=1, image=b"png"))


def test_a_new_rollout_id_starts_a_rollout_from_the_step(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = _loaded(seed_index=0)
    result = model.generate(Step(rollout_id=1, image=b"png", control="ctrl"))
    assert model.started == [Step(rollout_id=1, image=b"png", control="ctrl")]
    assert model.rollout_id == 1
    # The seed sits at index 0, so the first generated step is index 1.
    assert [(index, input) for index, _, input in pipeline.generated] == [(1, "ctrl")]
    assert pipeline.finalized == [1]
    assert isinstance(result, FlashDreamsResult)
    assert result.index == 1
    assert result.rollout_id == 1
    assert result.frames is _FRAMES


def test_a_cache_without_a_seed_starts_at_index_zero(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = _loaded(seed_index=None)
    result = model.generate(Step(rollout_id=1, image=b"png"))
    assert result.index == 0
    assert pipeline.finalized == [0]


def test_the_same_rollout_id_continues_the_rollout(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = _loaded()
    model.generate(Step(rollout_id=1, image=b"png"))
    second = model.generate(Step(rollout_id=1, image=None, control="right"))
    assert len(model.started) == 1
    assert second.index == 2
    assert second.rollout_id == 1
    assert pipeline.generated[-1][2] == "right"
    assert pipeline.finalized == [1, 2]


def test_a_new_rollout_id_drops_the_old_rollout_first(fake_flashdreams: dict[str, Any]) -> None:
    model, _ = _loaded()
    model.generate(Step(rollout_id=1, image=b"png"))
    first_cache = model.cache
    result = model.generate(Step(rollout_id=2, image=b"jpg"))
    assert len(model.started) == 2
    assert model.cache is not first_cache
    assert result.rollout_id == 2
    assert result.index == 1


def test_a_new_rollout_without_conditioning_is_the_familys_error(
    fake_flashdreams: dict[str, Any],
) -> None:
    model, pipeline = _loaded()
    model.generate(Step(rollout_id=1, image=b"png"))
    with pytest.raises(RolloutNotStarted):
        model.generate(Step(rollout_id=2, image=None))
    # The old rollout was dropped before the new one failed to start.
    assert model.cache is None
    assert model.rollout_id is None
    assert len(pipeline.generated) == 1


def test_the_rollout_is_exhausted_at_total_blocks(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = _loaded(seed_index=0, total_blocks=3)
    model.generate(Step(rollout_id=1, image=b"png"))
    model.generate(Step(rollout_id=1))
    with pytest.raises(RolloutExhausted) as excinfo:
        model.generate(Step(rollout_id=1))
    assert excinfo.value.args == (3,)
    # The rollout is kept; the application decides what follows.
    assert model.rollout_id == 1
    assert model.cache is not None
    assert pipeline.finalized == [1, 2]


def test_a_pipeline_failure_drops_the_cache_and_propagates(
    fake_flashdreams: dict[str, Any],
) -> None:
    model, pipeline = _loaded()
    model.generate(Step(rollout_id=1, image=b"png"))
    pipeline.fail_on = 2
    with pytest.raises(RuntimeError, match="device-side assert"):
        model.generate(Step(rollout_id=1))
    assert model.cache is None
    assert model.rollout_id is None
    assert pipeline.finalized == [1]
    # The next step with the same id starts the rollout again.
    result = model.generate(Step(rollout_id=1, image=b"png"))
    assert len(model.started) == 2
    assert result.index == 1


def test_a_new_prompt_in_the_same_rollout_is_refused_by_default(
    fake_flashdreams: dict[str, Any],
) -> None:
    model, pipeline = _loaded()
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    with pytest.raises(PromptSwapUnsupported):
        model.generate(Step(rollout_id=1, prompt="a desert"))
    assert model.prompt == "a forest"
    assert model.cache is not None
    assert len(pipeline.generated) == 1


def test_the_same_prompt_or_no_prompt_does_not_swap(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = _loaded()
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    model.generate(Step(rollout_id=1, prompt="a forest"))
    model.generate(Step(rollout_id=1, prompt=None))
    assert len(pipeline.generated) == 3
    assert model.prompt == "a forest"


def test_a_model_that_swaps_its_prompt_keeps_the_rollout(fake_flashdreams: dict[str, Any]) -> None:
    model = SwappingModel()
    pipeline = FakePipeline()
    model.pipeline = pipeline
    model.max_blocks = 100
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    result = model.generate(Step(rollout_id=1, prompt="a desert"))
    assert model.swapped == ["a desert"]
    assert model.prompt == "a desert"
    assert len(model.started) == 1
    assert result.index == 2


def test_a_new_rollout_records_its_prompt_without_swapping(
    fake_flashdreams: dict[str, Any],
) -> None:
    model = SwappingModel()
    model.pipeline = FakePipeline()
    model.max_blocks = 100
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    model.generate(Step(rollout_id=2, image=b"png", prompt="a desert"))
    assert model.swapped == []
    assert model.prompt == "a desert"


# -- reset --------------------------------------------------------------------


def test_reset_forgets_the_rollout_and_keeps_the_pipeline(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = _loaded()
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    model.reset()
    assert model.cache is None
    assert model.rollout_id is None
    assert model.prompt is None
    assert model.pipeline is pipeline
    result = model.generate(Step(rollout_id=1, image=b"png"))
    assert len(model.started) == 2
    assert result.index == 1


def test_initialize_cache_is_the_familys_to_write() -> None:
    with pytest.raises(NotImplementedError, match="initialize_cache"):
        FlashDreamsModel().initialize_cache(Step(rollout_id=1))


# -- frames -------------------------------------------------------------------


def test_to_frames_converts_a_batched_video_to_uint8_frames() -> None:
    torch = pytest.importorskip("torch")
    video = torch.linspace(-1.5, 1.5, 2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5)
    frames = model_module._to_frames(video)
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (2, 4, 5, 3)
    assert frames.dtype == np.uint8
    assert frames.min() == 0
    assert frames.max() == 255


def test_to_frames_accepts_an_unbatched_video_and_keeps_three_channels() -> None:
    torch = pytest.importorskip("torch")
    video = torch.zeros(2, 4, 4, 5)
    frames = model_module._to_frames(video)
    assert frames.shape == (2, 4, 5, 3)
    assert (frames == 128).all()
