"""Fakes the FlashDreams model-half tests share.

They stand in for the two imports ``FlashDreamsModel`` makes inside its
methods, FlashDreams' registry and torch, and for a family's model half, so
the tests exercise the rollout bookkeeping around the pipeline without
FlashDreams, torch, or a GPU. Not a test module: pytest collects nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reactor_runtime.flashdreams import FlashDreamsModel, RolloutNotStarted

FRAMES = np.zeros((4, 8, 16, 3), dtype=np.uint8)


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
        return FRAMES


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


def loaded_model(
    seed_index: int | None = 0, total_blocks: int = 10_000
) -> tuple[SeededModel, FakePipeline]:
    """A model half past ``load()``, with the pipeline set by hand."""
    model = SeededModel(seed_index)
    pipeline = FakePipeline()
    model.pipeline = pipeline
    model.max_blocks = total_blocks
    return model, pipeline
