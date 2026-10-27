"""Plasma, the session half: an ordinary ReactorApp that asks the worker for its frames.

One of these runs in each session process. It holds no palette and imports
nothing heavy: ``load`` connects to the worker, and ``generate`` sends the
session's tick and hue and returns the frame that comes back. The settings,
the commands, and the media path are the runtime's own, as in any model.
"""

from __future__ import annotations

from pathlib import Path

from plasma_worker import HEIGHT, WIDTH

from reactor_runtime import InputField, InputState, Output, ReactorApp, Video, session_started
from reactor_runtime.model_worker import WorkerClient

__all__ = ["HEIGHT", "WIDTH", "Plasma", "PlasmaOutput", "PlasmaState"]


class PlasmaOutput(Output):
    """The rendered field, one frame per step."""

    main_video: Video


class PlasmaState(InputState):
    """What a client can set. Each public field is a ``set_<field>`` command."""

    speed: int = InputField(default=2, ge=0, le=16, description="Palette steps per frame.")
    hue: int = InputField(default=0, ge=0, le=255, description="Palette offset.")


class Plasma(ReactorApp):
    """Drift the field at the client's speed, rendered by the shared worker."""

    state: PlasmaState
    fps = 30

    def load(self, config_path: Path | None) -> None:
        """Connect to the worker; the client keeps the session's availability up to date."""
        self.worker = WorkerClient.connect(self)
        self.tick = 0

    def generate(self, input: PlasmaState) -> PlasmaOutput:
        """Ask the worker for this session's next frame."""
        frame = self.worker.step({"tick": self.tick, "hue": input.hue})
        self.tick += input.speed
        return PlasmaOutput(main_video=frame)

    @session_started
    def on_session_started(self) -> None:
        """Start every session from the first frame, with fresh state on the worker."""
        self.tick = 0
        self.worker.reset()
