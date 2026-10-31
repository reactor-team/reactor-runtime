"""Plasma, the application half: an ordinary ReactorApp that asks the model server for its frames.

One of these runs in each session runtime. It holds no palette and imports
nothing heavy, and its three step hooks split as in any model:

- ``process_input()`` is application code. It reads the session's state and
  builds the step input, the session's ``Tick``, from the hue the client set
  and the tick it keeps, which advances at the speed the client set.
- ``generate()`` is the model call and nothing else: it sends the ``Tick`` to
  the model server and returns the ``Frame`` that comes back.
- ``process_output()`` is application code again. It puts the frame on the
  video track, or re-raises what went wrong.

A lost or stuck model server ends this runtime. Mid-session, the step raises
``WorkerLost`` or a ``StepUnanswered`` and ``process_output()`` re-raises it;
between sessions, the client's ``on_lost`` calls ``self.crash()``. Either way
the process exits, and its restart waits in ``load()`` until the model server
is back.
"""

from __future__ import annotations

from pathlib import Path

from plasma_types import HEIGHT, WIDTH, Frame, Tick

from reactor_runtime import (
    InputField,
    InputState,
    Output,
    ReactorApp,
    StepOutcome,
    Video,
    session_started,
)
from reactor_runtime.model_server import ModelClient

__all__ = ["HEIGHT", "WIDTH", "Plasma", "PlasmaOutput", "PlasmaState"]


class PlasmaOutput(Output):
    """The rendered field, one frame per step."""

    main_video: Video


class PlasmaState(InputState):
    """What a client can set. Each public field is a ``set_<field>`` command."""

    speed: int = InputField(default=2, ge=0, le=16, description="Palette steps per frame.")
    hue: int = InputField(default=0, ge=0, le=255, description="Palette offset.")


class Plasma(ReactorApp):
    """Drift the field at the client's speed, rendered by the shared model server."""

    state: PlasmaState
    fps = 30

    def load(self, config_path: Path | None) -> None:
        """Connect to the model server, waiting for it; losing it ends this runtime."""
        self.model: ModelClient[Tick, Frame] = ModelClient.connect(on_lost=self.crash)
        self.tick = 0

    @session_started
    def on_session_started(self) -> None:
        """Start every session from the first frame, with fresh state on the model server."""
        self.tick = 0
        self.model.reset()

    async def process_input(self) -> Tick:
        """Say which frame of the palette this session shows next, and advance it."""
        tick = Tick(tick=self.tick, hue=self.state.hue)
        # The palette index is taken modulo 256, so the wrapped tick renders the
        # same frames and stays a small integer however long the session runs.
        self.tick = (self.tick + self.state.speed) % 256
        return tick

    def generate(self, input: Tick) -> Frame:
        """One frame, rendered by the model server."""
        return self.model.step(input)

    async def process_output(self, outcome: StepOutcome) -> PlasmaOutput:
        """Put the frame on the video track."""
        if outcome.error is not None:
            # PlasmaModel raises nothing for a request, so an error is the model
            # server lost or stuck (WorkerLost, StepUnanswered) or a bug. Neither
            # is repaired by a reset: re-raising ends this runtime, and its
            # restart waits in load() until the model server is back.
            raise outcome.error
        frame: Frame = outcome.result
        return PlasmaOutput(main_video=frame.pixels)
