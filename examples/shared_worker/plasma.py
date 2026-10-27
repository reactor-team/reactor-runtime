"""Plasma, the application half: an ordinary ReactorApp that asks the worker for its frames.

One of these runs in each session process. It holds no palette and imports
nothing heavy: ``load`` connects to the worker, and ``generate`` sends the
session's tick and hue and returns the frame that comes back. The settings,
the commands, and the media path are the runtime's own, as in any model.

It decides what a lost or a stuck worker means with option 2 of
:mod:`reactor_runtime.model_worker`: the session ends with the reason, the
process stays up and refuses sessions until the worker is back, and one watcher
thread of its own reconnects.
"""

from __future__ import annotations

import threading
from pathlib import Path

from plasma_contract import HEIGHT, WIDTH, Frame, Tick

from reactor_runtime import (
    ApplicationError,
    InputField,
    InputState,
    Output,
    ReactorApp,
    StepCompleted,
    StepOutcome,
    Video,
    session_started,
)
from reactor_runtime.model_worker import (
    ClientClosed,
    StepUnanswered,
    WorkerClient,
    WorkerLost,
    WorkerRejected,
    WorkerTimeout,
    WorkerUnreachable,
)

__all__ = ["HEIGHT", "WIDTH", "Plasma", "PlasmaOutput", "PlasmaState"]

# How long each wait of the watcher lasts, so the process can always exit.
_WATCH_SECONDS = 1.0
_RECONNECT_SECONDS = 5.0


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
        """Connect to the worker, waiting for it, and start watching the connection."""
        self.worker = WorkerClient.connect(request_type=Tick, result_type=Frame)
        self.tick = 0
        self._give_up = threading.Event()
        threading.Thread(target=self._watch, name="plasma-watch", daemon=True).start()

    @session_started
    def on_session_started(self) -> None:
        """Start every session from the first frame, with fresh state on the worker."""
        self.tick = 0
        try:
            self.worker.reset()
        except (WorkerLost, StepUnanswered, WorkerRejected) as exc:
            self._lose_worker(f"worker_unavailable: the reset failed: {exc}")

    async def process_input(self) -> PlasmaState:
        """Step while the worker serves; refuse every step until it is back."""
        if self.unavailable is not None:
            raise ApplicationError(self.unavailable)
        return await super().process_input()

    def generate(self, input: PlasmaState) -> PlasmaOutput:
        """Ask the worker for this session's next frame."""
        frame = self.worker.step(Tick(tick=self.tick, hue=input.hue))
        self.tick += input.speed
        return PlasmaOutput(main_video=frame.pixels)

    async def process_output(self, outcome: StepOutcome) -> Output | StepCompleted | None:
        """End the session on a lost or a stuck worker; anything else is the default's."""
        if isinstance(outcome.error, (WorkerLost, StepUnanswered)):
            self._lose_worker(str(outcome.error))
            return None
        return await super().process_output(outcome)

    def _lose_worker(self, reason: str) -> None:
        """Give up on this connection: end the session, refuse new ones, and reconnect."""
        self.report_unavailable(reason)
        self._give_up.set()

    def _watch(self) -> None:
        """Wait for the connection to be lost or given up on, then reconnect, for good."""
        while True:
            try:
                reason = self.worker.wait_lost(timeout=_WATCH_SECONDS)
            except TimeoutError:
                if not self._give_up.is_set():
                    continue
                reason = self.unavailable or "worker_unavailable"
            except ClientClosed:
                return
            self.report_unavailable(reason)
            while True:
                try:
                    self.worker.reconnect(timeout=_RECONNECT_SECONDS)
                    break
                except (WorkerTimeout, WorkerRejected, WorkerUnreachable):
                    continue
                except ClientClosed:
                    return
            self._give_up.clear()
            self.report_available()
