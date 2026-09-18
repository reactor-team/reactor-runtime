"""Echo, the model half.

Applies a video effect to one frame with OpenCV. Nothing in this file knows
about clients, tracks, commands, or the runtime's loop, and nothing here
imports ``reactor_runtime``. :class:`EchoModel` has the three methods every
model half has, ``load``, ``generate``, and ``reset``, and holds no state
between frames: the effects are pure functions of the frame they are given.

The step input and the step result are the contract between the two halves.
Both are plain dataclasses the two files agree on; the runtime never reads
their fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Effect = Literal["none", "grayscale", "sepia", "edges", "invert", "blur", "pixelate"]
EFFECTS: list[str] = ["none", "grayscale", "sepia", "edges", "invert", "blur", "pixelate"]

_SEPIA_KERNEL = np.array(
    [
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class EchoInput:
    """What one step needs.

    Attributes:
        frame: The client's frame, uint8 ``(H, W, 3)`` RGB.
        effect: One of :data:`EFFECTS`.
        intensity: How strongly the effect applies, ``0`` for none, ``1`` for full.
        caption: Text drawn over the frame; empty draws nothing.
    """

    frame: np.ndarray
    effect: str
    intensity: float
    caption: str


@dataclass(frozen=True)
class EchoResult:
    """What one step produced.

    Attributes:
        frame: The processed frame, uint8 ``(H, W, 3)`` RGB, the size it came in.
    """

    frame: np.ndarray


class EchoModel:
    """Video effects behind ``load`` / ``generate`` / ``reset``."""

    def load(self) -> None:
        """Nothing to load: the effects are OpenCV calls with no weights."""

    def generate(self, input: EchoInput) -> EchoResult:
        """Apply the effect to the frame, then draw the caption over it."""
        frame = apply_effect(input.frame, input.effect, input.intensity)
        if input.caption:
            frame = draw_caption(frame, input.caption)
        return EchoResult(frame=frame)

    def reset(self) -> None:
        """Nothing to forget: no state is held between frames."""


# -- Effects -------------------------------------------------------------------


def apply_effect(frame: np.ndarray, effect: str, intensity: float) -> np.ndarray:
    """Apply *effect* at *intensity* to an RGB frame, returning a new RGB frame.

    An effect below full intensity is blended with the original frame, so
    ``intensity`` reads as a wet/dry mix.
    """
    if effect == "none" or intensity == 0.0:
        return frame

    if effect == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif effect == "sepia":
        processed = np.clip(cv2.transform(frame, _SEPIA_KERNEL), 0, 255).astype(np.uint8)
    elif effect == "edges":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif effect == "invert":
        processed = np.subtract(255, frame)
    elif effect == "blur":
        kernel_size = max(1, int(21 * intensity) | 1)
        if kernel_size <= 1:
            return frame
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    elif effect == "pixelate":
        h, w = frame.shape[:2]
        pixel_size = max(2, int(32 * intensity))
        small = cv2.resize(
            frame,
            (max(1, w // pixel_size), max(1, h // pixel_size)),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        return frame

    if intensity >= 1.0:
        return processed
    return cv2.addWeighted(frame, 1.0 - intensity, processed, intensity, 0).astype(np.uint8)


def draw_caption(frame: np.ndarray, caption: str) -> np.ndarray:
    """Draw *caption* near the bottom of the frame, outlined for legibility."""
    out = frame.copy()
    origin = (12, max(24, frame.shape[0] - 16))
    for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(
            out, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA
        )
    return out
