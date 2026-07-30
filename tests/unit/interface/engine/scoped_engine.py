"""An engine that names nothing, so discovery has to read its package.

Kept in its own module because package scoping is what is under test: anything
declared beside an engine belongs to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reactor_runtime.engine_contract import Frames, ModelInput, UserInput, VideoInput


class Jump(UserInput):
    height: float = 1.0


class Lens(VideoInput):
    pass


class ScopedStepInput(ModelInput):
    jumps: int = 0


class ScopedEngine:
    """Declares no ``declared_inputs``; its package is the declaration."""

    def initialize_cache(self, **init: Any) -> dict[str, Any]:
        return dict(init)

    def map_inputs(self, inputs: list[UserInput], cache: Any) -> ScopedStepInput:
        return ScopedStepInput(jumps=sum(isinstance(item, Jump) for item in inputs))

    def generate(self, index: int, cache: Any, input: ScopedStepInput) -> Frames:
        return Frames(main_video=np.zeros((2, 2, 3), dtype=np.uint8))

    def finalize(self, index: int, cache: Any) -> None:
        return None
