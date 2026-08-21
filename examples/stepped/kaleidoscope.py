"""The model half of the stepped example: weights, caches, and math.

Nothing in this file knows Reactor exists. It imports no ``reactor_runtime``,
takes plain arrays and plain values, and hands back plain arrays — so it runs in
a plain script with Reactor uninstalled. A real model would load weights in
:meth:`Kaleidoscope.load` and run a forward pass in
:meth:`Kaleidoscope.generate`; this one shifts hue and mirrors frames so the
example needs no GPU.

Everything that persists between steps is private to this class: the step
counter, the hue it has drifted to, and the conditioning it was last given.
That is what lets the application stay a description of the wire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_MIRRORS = {"none": 0, "vertical": 1, "horizontal": 2, "both": 3}


class Kaleidoscope:
    """Tints and mirrors incoming frames, drifting the tint as it goes."""

    def __init__(self) -> None:
        self._hue = 0.0
        self._step = 0
        self._conditioning: tuple[str, float] | None = None

    def load(self, config: Path | None) -> None:
        """Prepare the model. A real one would build and warm up its network here.

        Args:
            config: The config file the workspace declared, or ``None``. This
                model has nothing to read, so it only notes that it loaded.
        """
        self._reset_rollout()

    def generate(
        self, *, driving: np.ndarray, mirror: str, drift: float
    ) -> dict[str, np.ndarray | float | int]:
        """Produce one step from a block of driving frames.

        The conditioning rides along with every step, so this method owns
        noticing that it changed: a new mirror or drift reopens the rollout,
        which is the model's own business and never the application's.

        The returned ``chunk`` is this model's own position in its rollout, and
        it starts over whenever the rollout does. The runtime's step tally never
        does, so a caller that wants the restarting number gets it from here.

        Args:
            driving: The driving frames, shaped ``(n, height, width, 3)``.
            mirror: Which way to mirror the frames.
            drift: Hue shift per step, in turns.

        Returns:
            The produced ``frames``, the ``hue`` each one was tinted with, and
            the ``chunk`` index within the current rollout.
        """
        conditioning = (mirror, drift)
        if conditioning != self._conditioning:
            self._conditioning = conditioning
            self._reset_rollout()

        frames = np.empty_like(driving)
        hues: list[float] = []
        for index, frame in enumerate(driving):
            self._hue = (self._hue + drift) % 1.0
            frames[index] = _mirror(_tint(frame, self._hue), mirror)
            hues.append(self._hue)
        chunk = self._step
        self._step += 1
        return {"frames": frames, "hue": float(np.mean(hues)), "chunk": chunk}

    def restart(self) -> None:
        """Start the rollout over, keeping the conditioning already in effect.

        The application calls this from a command handler. There is no framework
        primitive for it: what a restart means belongs to the model.
        """
        self._reset_rollout()

    @property
    def step(self) -> int:
        """How many steps this rollout has produced."""
        return self._step

    def _reset_rollout(self) -> None:
        self._hue = 0.0
        self._step = 0


def _tint(frame: np.ndarray, hue: float) -> np.ndarray:
    """Scale a frame's channels by a rotating weight, standing in for a forward pass."""
    angle = hue * 2.0 * np.pi
    weights = 0.6 + 0.4 * np.cos(angle + np.array([0.0, 2.0, 4.0], dtype=np.float32))
    return np.clip(frame.astype(np.float32) * weights, 0, 255).astype(np.uint8)


def _mirror(frame: np.ndarray, mirror: str) -> np.ndarray:
    """Mirror a frame vertically, horizontally, both, or not at all."""
    mode = _MIRRORS.get(mirror, 0)
    if mode in (1, 3):
        frame = frame[:, ::-1]
    if mode in (2, 3):
        frame = frame[::-1, :]
    return frame
