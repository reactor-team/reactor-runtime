"""Plasma, the model half: the palette loaded once, every session's frame rendered in one step.

The model server process holds what a real model's weights would be, a palette
and a phase field, and renders the frames of every session that is waiting in
one vectorized call. Integer arithmetic keeps each session's frame identical
whether it was rendered alone or in a batch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from plasma_types import HEIGHT, WIDTH, Frame, Tick

from reactor_runtime.model_server import ModelServer, ServerRequest

logger = logging.getLogger(__name__)


class PlasmaModel(ModelServer[Tick, Frame]):
    """Render one frame per waiting session, up to four at a time."""

    max_batch = 4
    batch_window = 0.002

    def load(self, config_path: Path | None) -> None:
        """Build the palette and the phase field: the weights every session shares."""
        y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
        field = np.sin(x / 23.0) + np.sin(y / 17.0) + np.sin((x + y) / 31.0)
        self.phase = ((field + 3.0) / 6.0 * 255.0).astype(np.int32)
        steps = np.arange(256) / 256.0 * 2 * np.pi
        self.palette = np.stack(
            [
                (np.sin(steps) * 127 + 128),
                (np.sin(steps + 2.1) * 127 + 128),
                (np.sin(steps + 4.2) * 127 + 128),
            ],
            axis=1,
        ).astype(np.uint8)
        self.frames_served: dict[int, int] = {}

    def step(self, requests: list[ServerRequest[Tick]]) -> list[Frame]:
        """Render every request's frame at its tick and hue, in one call."""
        ticks = np.array([r.payload.tick for r in requests], dtype=np.int32)
        hues = np.array([r.payload.hue for r in requests], dtype=np.int32)
        index = (self.phase[None] + ticks[:, None, None] + hues[:, None, None]) % 256
        frames = self.palette[index]
        for request in requests:
            self.frames_served[request.slot] = self.frames_served.get(request.slot, 0) + 1
        return [Frame(pixels=np.ascontiguousarray(frame)) for frame in frames]

    def session_opened(self, slot: int, generation: int) -> None:
        """Start counting this slot's frames."""
        self.frames_served[slot] = 0

    def session_closed(self, slot: int, generation: int) -> None:
        """Report what the slot was served and forget it."""
        logger.info("slot %d closed after %d frames", slot, self.frames_served.pop(slot, 0))
