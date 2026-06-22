"""A minimal example model.

The smallest model worth shipping: one video output track and one command. It
emits a solid grey frame whose brightness a client can set, so a reader can see
the shape of a model — a declared track, an ``@event`` command, and a ``run``
loop that emits — with no model weights and no dependencies beyond the runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reactor_runtime.model import Output, ReactorModel, Video, event

_WIDTH = 512
_HEIGHT = 512


class PassthroughOutput(Output):
    """The model's single outbound video track."""

    video: Video


class Passthrough(ReactorModel):
    """Emit a solid frame whose brightness a client can set."""

    output: PassthroughOutput

    def load(self, config: dict[str, Any]) -> None:
        """Start from the configured brightness, mid-grey by default."""
        self._brightness = int(config.get("brightness", 128))

    @event(name="set_brightness")
    async def set_brightness(self, value: int = 128) -> None:
        """Set the brightness of the emitted frame, clamped to 0-255."""
        self._brightness = max(0, min(255, value))

    async def run(self) -> None:
        """Emit a solid frame forever, paced by the output buffer."""
        while True:
            frame = np.full((_HEIGHT, _WIDTH, 3), self._brightness, dtype=np.uint8)
            await self.emit(PassthroughOutput(video=frame))
