"""A fake engine every engine-layer test drives.

It is the smallest thing that exercises both parity bars — interactive control
and video-to-video — with no weights: one event, one batched media input, an
initialization with defaults, and a ``generate`` that returns a solid colour.

The declarations are named on the engine with ``declared_inputs`` rather than
left to package scoping, so one test module's classes cannot leak into another's
engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reactor_runtime.core.values import InputFrame
from reactor_runtime.engine_contract import (
    Init,
    InputField,
    ModelInput,
    UserInput,
    VideoChunk,
    VideoInput,
)


class Move(UserInput):
    """Move the cursor."""

    direction: str = InputField(default="left", choices=["left", "right"])
    speed: float = InputField(default=1.0, ge=0.0, le=8.0)


class Camera(VideoInput):
    chunk_size = 2


class FakeInit(Init):
    shade: int = InputField(default=8, ge=0, le=255)


class FakeStepInput(ModelInput):
    shade: int
    moves: int = 0
    source: Any = None


@dataclass
class FakeCache:
    shade: int
    steps: int = 0
    initialized_with: dict[str, Any] = field(default_factory=dict)
    restarts: int = 0


class FakeEngine:
    """A pipeline whose whole rollout is a shade and a step count."""

    declared_inputs = (Move, Camera, FakeInit)

    def __init__(self) -> None:
        self.windows: list[list[UserInput]] = []
        self.generated: list[int] = []
        self.finalized: list[int] = []

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        return 1 if autoregressive_index == 0 else 2

    def initialize_cache(self, **init: Any) -> FakeCache:
        return FakeCache(shade=int(init.get("shade", 8)), initialized_with=dict(init))

    def map_inputs(
        self, autoregressive_index: int, cache: FakeCache, inputs: list[UserInput]
    ) -> FakeStepInput | None:
        self.windows.append(list(inputs))
        source = None
        moves = 0
        for item in inputs:
            if isinstance(item, FakeInit):
                cache.shade = item.shade
                cache.restarts += 1
                return None
            if isinstance(item, Move):
                moves += 1
            elif isinstance(item, Camera):
                source = item.data
        return FakeStepInput(shade=cache.shade, moves=moves, source=source)

    def generate(
        self,
        autoregressive_index: int,
        cache: FakeCache,
        input: FakeStepInput | None = None,
    ) -> VideoChunk:
        self.generated.append(autoregressive_index)
        assert input is not None
        frames = self.get_num_output_frames(autoregressive_index)
        return np.full((frames, 3, 2, 2), input.shade, dtype=np.uint8)

    def finalize(self, autoregressive_index: int, cache: FakeCache) -> dict[str, float] | None:
        self.finalized.append(autoregressive_index)
        cache.steps += 1
        return {"step_seconds": 0.0}


class MediaOnlyEngine(FakeEngine):
    """A video-to-video engine: no complete source chunk means no step."""

    declared_inputs = (Camera,)

    def map_inputs(
        self, autoregressive_index: int, cache: FakeCache, inputs: list[UserInput]
    ) -> FakeStepInput | None:
        self.windows.append(list(inputs))
        chunks = [item for item in inputs if isinstance(item, Camera)]
        if not chunks:
            return None
        return FakeStepInput(shade=cache.shade, source=chunks[-1].data)


class StrictInit(Init):
    prompt: str


class StrictEngine(FakeEngine):
    """An engine a client must initialize before any rollout can exist."""

    declared_inputs = (StrictInit,)


def frame(value: int = 0) -> InputFrame:
    """Return a decoded inbound frame."""
    return InputFrame(data=np.full((2, 2, 3), value, dtype=np.uint8), pts=float(value))
