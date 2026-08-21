"""Stepped — the runtime drives the model, and the application describes the wire.

The application half of the example, and it imports nothing that computes: no
NumPy math, no model internals. It declares what the client may set
(:class:`SteppedState`), what media flows (:class:`SteppedInput` /
:class:`SteppedOutput`), and three small methods that bridge the two:

- ``map_step`` reads the state, drains the webcam, and returns the arguments the
  model wants — or raises :class:`NotReady` when frames have not arrived yet.
- ``to_output`` puts what the model produced on a track, tags every frame with
  the hue that produced it, and sends a :class:`StepDone` message alongside.
- ``on_step`` is where a per-step application effect would go.

There is no ``run()`` here. The runtime steps the model while a client is
connected, paces playout from how long each step took, and holds the stream
whenever the mapping declines. A model whose loop does not fit that shape
overrides ``run()`` and keeps everything else on this page.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from kaleidoscope import Kaleidoscope

from reactor_runtime import (
    Input,
    InputBuffer,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    NotReady,
    Output,
    ReactorModel,
    ReadMode,
    StepStats,
    TrackPayload,
    Video,
    event,
)
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

BLOCK = 4
"""Driving frames the model consumes per step."""

CAMERA_FPS = 30
"""The rate the generated frames play out at, matching the webcam that drove them."""


class SteppedInput(Input):
    """The client's webcam, which drives generation."""

    webcam: Video


class SteppedOutput(Output):
    """The generated video sent back."""

    main_video: Video


class SteppedState(InputState):
    """What a client may change mid-session. Each public field becomes a command."""

    mirror: str = InputField(
        default="none",
        choices=["none", "vertical", "horizontal", "both"],
        description="How the generated frames are mirrored.",
    )
    drift: float = InputField(
        default=0.01,
        ge=0.0,
        le=0.5,
        description="How far the tint moves each frame, as a fraction of the colour wheel.",
    )


class StepDone(ModelMessage):
    """Sent after each generated block."""

    chunk: int = MessageField(
        description="Index of the block within the current tint, which `restart` sets back to zero."
    )
    step: int = MessageField(
        description="Blocks generated since the model started, which nothing sets back to zero."
    )
    hue: float = MessageField(description="Average tint of the block, as a fraction of the wheel.")


class Stepped(ReactorModel):
    """Tint and mirror the client's webcam, driven by the runtime one step at a time."""

    state: SteppedState
    input: SteppedInput
    model: Kaleidoscope

    def load(self, config_path: Path | None) -> None:
        """Build the model and hand it the workspace's config.

        A model that takes no constructor arguments can be named as a class
        attribute instead — ``model = Kaleidoscope`` — and the runtime builds it
        and calls its ``load()``.

        Args:
            config_path: The config file the workspace declared, or ``None``.
        """
        self.model = Kaleidoscope()
        self.model.load(config_path)

    def map_step(self, state: SteppedState, input: SteppedInput) -> dict:
        """Build one step's arguments from the state and the frames that arrived.

        No side effects: it reads, drains, and returns. ``LATEST`` keeps
        generation on the live edge — when the model runs slower than the camera,
        the backlog is dropped rather than played out later and later.

        Frames are resampled to the newest one's size, since a mid-block
        resolution change would otherwise not stack into a single array.

        Args:
            state: The session's state, as the client last set it.
            input: The inbound tracks, as live frame buffers.

        Returns:
            The keyword arguments the model's ``generate`` takes.

        Raises:
            NotReady: Fewer than ``BLOCK`` frames have arrived.
        """
        # The runtime binds a live InputBuffer to each declared input track; the
        # track annotations (Video/Audio) only carry the kind for the contract.
        webcam = cast(InputBuffer, input.webcam)
        frames = webcam.try_read(BLOCK, mode=ReadMode.LATEST)
        if frames is None:
            raise NotReady(f"waiting for {BLOCK} webcam frames")

        # WebRTC rescales an inbound track as bandwidth and CPU move, so a
        # resolution change lands mid-block. A block is one array, so the older
        # frames are resampled up to the newest frame's size.
        newest = frames[-1].data.shape
        block = [_resized(frame.data, newest) for frame in frames]

        return {
            "driving": np.stack(block),
            "mirror": state.mirror,
            "drift": state.drift,
        }

    def to_output(
        self, *, frames: np.ndarray, hue: float, chunk: int, stats: StepStats
    ) -> tuple[StepDone, SteppedOutput]:
        """Put the block on the video track and report the step that made it.

        The message goes out before the media, so a client tracking progress
        hears about a step without waiting behind the frames it produced.

        Two counts, from the two parties that own them: ``chunk`` is where the
        model is in its rollout and starts over when ``restart`` restarts it,
        while ``stats.step`` is what the runtime has driven and never starts
        over.

        The output states its own rate. This model transforms a webcam, so its
        frames play at the cadence they arrived at — tinting them takes a
        fraction of that, and pacing playout from how fast the tint ran would
        play the video back at many times life speed.

        Args:
            frames: The generated block, one entry per frame.
            hue: The average tint of the block.
            chunk: The model's position in its current rollout.
            stats: What the runtime measured about this step.

        Returns:
            The message to send and the media to emit.
        """
        return (
            StepDone(chunk=chunk, step=stats.step, hue=hue),
            SteppedOutput(
                main_video=TrackPayload(frames, metadata=[{"hue": hue}] * len(frames)),
                fps=CAMERA_FPS,
            ),
        )

    def on_step(self, stats: StepStats) -> None:
        """Log the slow steps, which is the sort of effect this hook is for."""
        if stats.compute_time > 0.5:
            logger.warning("slow step", step=stats.step, seconds=stats.compute_time)

    @event(name="restart", description="Start the tint over from the beginning.")
    async def restart(self) -> None:
        """Cut playout, then restart the rollout.

        Two effects that belong to two owners: dropping the frames already
        queued is the application's, and what a restart means to the rollout is
        the model's own method.
        """
        self.output.flush()
        self.model.restart()


def _resized(frame: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Resample *frame* to *shape*, by nearest neighbour, and return it unchanged if it fits.

    Nearest neighbour keeps the example dependency-free: a model that cares
    about resample quality would reach for a real image library here.

    Args:
        frame: The frame to resample.
        shape: The target shape, as ``(height, width, channels)``.

    Returns:
        A frame of *shape*.
    """
    if frame.shape == shape:
        return frame
    height, width = shape[0], shape[1]
    rows = np.arange(height) * frame.shape[0] // height
    cols = np.arange(width) * frame.shape[1] // width
    return frame[rows[:, None], cols]
