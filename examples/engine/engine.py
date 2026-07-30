"""A worked engine: everything an inference engine declares, and nothing else.

There is no runtime import here, and there is no serving code. The engine
declares the events a client can send, the media it consumes, the state a
rollout starts from, and the conditioning of one step — then implements the
four calls that drive it. A runtime reads those declarations and serves them.

The "inference" is a cursor painting on a canvas, so the whole thing runs on a
laptop with no weights and no GPU: enough to exercise interactive control, media
input, initialization, and the ordered window end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from reactor_runtime.engine_contract import (
    Frames,
    Init,
    InputField,
    ModelInput,
    UserInput,
    VideoInput,
)

CHUNK_FRAMES = 4
"""How many source frames one :class:`Camera` instance carries."""

Direction = Literal["up", "down", "left", "right"]
"""The directions the brush understands."""

_DIRECTIONS: dict[Direction, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class Move(UserInput):
    """Move the brush one step in a direction."""

    direction: Direction
    speed: float = InputField(default=8.0, ge=1.0, le=64.0, description="Pixels per step.")


class Brush(UserInput):
    """Set the colour the brush paints with."""

    red: int = InputField(default=255, ge=0, le=255)
    green: int = InputField(default=64, ge=0, le=255)
    blue: int = InputField(default=64, ge=0, le=255)


class Camera(VideoInput):
    """Source video the canvas is tinted with, when a client publishes it."""

    chunk_size = CHUNK_FRAMES


class PaintInit(Init):
    """The canvas a rollout starts from."""

    width: int = InputField(default=256, ge=32, le=1024)
    height: int = InputField(default=256, ge=32, le=1024)
    background: int = InputField(default=16, ge=0, le=255, description="Grey level, 0-255.")


class PaintStepInput(ModelInput):
    """One step's conditioning: where the brush lands, in what colour, over what source."""

    positions: list[tuple[int, int]]
    colour: tuple[int, int, int]
    source: Any = None


@dataclass
class PaintCache:
    """A rollout's memory: the canvas, the brush, and where the cursor is."""

    canvas: np.ndarray
    row: int
    col: int
    colour: tuple[int, int, int] = (255, 64, 64)
    final_state: np.ndarray | None = field(default=None, repr=False)


class PaintPipeline:
    """The engine. Satisfies the streaming-pipeline protocol structurally."""

    def initialize_cache(self, **init: Any) -> PaintCache:
        """Open a rollout on a fresh canvas."""
        height = int(init.get("height", 256))
        width = int(init.get("width", 256))
        background = int(init.get("background", 16))
        canvas = np.full((height, width, 3), background, dtype=np.uint8)
        return PaintCache(canvas=canvas, row=height // 2, col=width // 2)

    def map_inputs(self, inputs: list[UserInput], cache: PaintCache) -> PaintStepInput | None:
        """Fold one window into the next step's conditioning.

        The fold happens in one pass, in arrival order: a colour is last-value
        wins, the moves are integrated into a path so a burst of them travels
        further than a single one, and the newest complete chunk of source video
        is the one this step conditions on. A colour or a cursor that has to
        survive into the next window lives on the cache, because the cache is
        the rollout.
        """
        height, width = cache.canvas.shape[:2]
        positions: list[tuple[int, int]] = []
        source = None
        for item in inputs:
            if isinstance(item, PaintInit):
                # A fresh init means a new sequence. Re-initializing is the
                # engine's own business, so it happens here.
                self._restart(cache, item)
                return None
            if isinstance(item, Brush):
                cache.colour = (item.red, item.green, item.blue)
            elif isinstance(item, Move):
                d_row, d_col = _DIRECTIONS[item.direction]
                cache.row = int(np.clip(cache.row + d_row * item.speed, 0, height - 1))
                cache.col = int(np.clip(cache.col + d_col * item.speed, 0, width - 1))
                positions.append((cache.row, cache.col))
            elif isinstance(item, Camera):
                source = item.data
        return PaintStepInput(
            positions=positions or [(cache.row, cache.col)],
            colour=cache.colour,
            source=source,
        )

    def generate(self, index: int, cache: PaintCache, input: PaintStepInput) -> Frames:
        """Paint this step's path onto the canvas and hand back the frame."""
        canvas = cache.canvas.copy()
        if input.source is not None:
            canvas = _tint(canvas, input.source)
        for row, col in input.positions:
            canvas[
                max(row - 4, 0) : row + 4,
                max(col - 4, 0) : col + 4,
            ] = input.colour
        cache.final_state = canvas
        return Frames(main_video=canvas)

    def finalize(self, index: int, cache: PaintCache) -> None:
        """Commit the step into the rollout's memory."""
        if cache.final_state is not None:
            cache.canvas = cache.final_state
            cache.final_state = None

    def _restart(self, cache: PaintCache, init: PaintInit) -> None:
        """Reset the rollout in place, the way an engine with pinned buffers must."""
        cache.canvas = np.full((init.height, init.width, 3), init.background, dtype=np.uint8)
        cache.row = init.height // 2
        cache.col = init.width // 2
        cache.final_state = None


def _tint(canvas: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Blend the last source frame of a chunk into the canvas."""
    frame = source[-1] if source.ndim == 4 else source
    height, width = canvas.shape[:2]
    resized = frame[:height, :width]
    blended = canvas.astype(np.int16)
    blended[: resized.shape[0], : resized.shape[1]] += resized.astype(np.int16) // 4
    return np.clip(blended, 0, 255).astype(np.uint8)
