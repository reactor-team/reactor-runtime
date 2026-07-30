"""An engine that names nothing, so discovery has to read its package.

Kept in its own module because package scoping is what is under test: anything
declared beside an engine belongs to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reactor_runtime.engine_contract import ModelInput, UserInput, VideoChunk, VideoInput


class Jump(UserInput):
    height: float = 1.0


class Lens(VideoInput):
    pass


class ScopedStepInput(ModelInput):
    jumps: int = 0


class ScopedEngine:
    """Declares no ``declared_inputs``; its package is the declaration."""

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        return 1

    def initialize_cache(self, **init: Any) -> dict[str, Any]:
        return dict(init)

    def map_inputs(
        self, autoregressive_index: int, cache: Any, inputs: list[UserInput]
    ) -> ScopedStepInput:
        return ScopedStepInput(jumps=sum(isinstance(item, Jump) for item in inputs))

    def generate(
        self, autoregressive_index: int, cache: Any, input: ScopedStepInput | None = None
    ) -> VideoChunk:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def finalize(self, autoregressive_index: int, cache: Any) -> dict[str, float] | None:
        return None
