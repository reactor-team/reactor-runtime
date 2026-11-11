"""The inner contract between a FlashDreams application half and its model half.

One step result crosses back from the model half, and three exceptions say why
a step could not run. Every family shares these; each family declares its own
step input beside its application class. Nothing here imports the runtime's
interface or FlashDreams, so the model half and its tests read this module
without either installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlashDreamsResult:
    """What one pipeline step produced.

    Attributes:
        frames: The frames this step generated, uint8 ``(T, H, W, 3)``, on the CPU.
        index: The pipeline's own step count within the current rollout. The
            step that starts a rollout from a seed reports ``1`` when the
            pipeline commits the seed as index ``0``, and ``0`` otherwise.
        rollout_id: The rollout the model now holds. The application reads it
            to know when the model has applied a new rollout id.
    """

    frames: np.ndarray
    index: int
    rollout_id: int


class RolloutNotStarted(Exception):  # noqa: N818 (the model's own error, named for the state)
    """A new rollout was asked for, and the step carries nothing to start it from."""


class RolloutExhausted(Exception):  # noqa: N818 (the model's own error, named for the state)
    """The rollout reached the adapter's ``total_blocks``.

    The argument is the index the next step would have had.
    """


class PromptSwapUnsupported(Exception):  # noqa: N818 (the model's own error, named for the state)
    """The prompt changed within a rollout, and this model cannot swap it in place."""
