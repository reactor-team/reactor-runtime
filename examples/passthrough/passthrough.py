"""A minimal example model.

The smallest model worth shipping: one video output track and one command. It
emits a solid grey frame whose brightness a client can set, so a reader can see
the shape of a model — a declared track, an ``@event`` command, and a ``run``
loop that emits — with no model weights and no dependencies beyond the runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from reactor_runtime import InputField, Output, ReactorModel, Video, event

_WIDTH = 512
_HEIGHT = 512


class PassthroughOutput(Output):
    """The model's single outbound video track."""

    video: Video


class Passthrough(ReactorModel):
    """Emit a solid frame whose brightness a client can set."""

    output: PassthroughOutput

    def load(self, config_path: Path | None) -> None:
        """Start from the brightness in the config file, mid-grey by default.

        Shows the config contract: the runtime hands the model the path to its
        config file (or ``None``), and the model reads it however it likes.
        """
        config = yaml.safe_load(config_path.read_text()) if config_path else {}
        self._brightness = int(config.get("brightness", 128))

    @event(name="set_brightness", description="Set the brightness of the emitted frame")
    async def set_brightness(
        self,
        value: int = InputField(default=128, ge=0, le=255, description="Frame brightness, 0-255"),
    ) -> None:
        """Set the brightness of the emitted frame.

        The value is bounded to 0-255 by the contract, so a client UI renders it
        as a slider over that range.
        """
        self._brightness = value

    async def run(self) -> None:
        """Emit a solid frame forever, paced by the output buffer."""
        while True:
            frame = np.full((_HEIGHT, _WIDTH, 3), self._brightness, dtype=np.uint8)
            await self.emit(PassthroughOutput(video=frame))
