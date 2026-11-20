"""A director model for the endurance suite: plain shots, built on the CPU in a moment.

Each shot is one gray level, a step brighter than the shot it continues, so a
continuation is visible in its frames. The class it subclasses owns the
queues, the step each shot reports, and playout.
"""

from __future__ import annotations

import numpy as np

from reactor_runtime import Audio, InputField, Output, Video
from reactor_runtime.apps.director import ReactorDirectorApp, Shot

FPS = 24
SIZE = 64


class ReelOutput(Output):
    """One shot's frames and its silence."""

    main_video: Video
    main_audio: Audio


class DirectorReel(ReactorDirectorApp):
    """Plain gray shots, each a step brighter than the shot it continues."""

    fps = FPS

    def generate(
        self,
        prompt: str = InputField(max_length=200, description="A label for the shot."),
        frames: int = InputField(default=12, ge=1, le=48, description="How many frames."),
        continue_from: Shot | None = None,
    ) -> ReelOutput:
        """Build one plain shot, a step brighter than the shot it continues."""
        level = 0
        if continue_from is not None and continue_from.last_frame is not None:
            level = int(continue_from.last_frame[0, 0, 0])
        level = (level + 16) % 256
        return ReelOutput(
            main_video=np.full((frames, SIZE, SIZE, 3), level, np.uint8),
            main_audio=np.zeros((1, round(frames * 48_000 / FPS)), np.int16),
        )
