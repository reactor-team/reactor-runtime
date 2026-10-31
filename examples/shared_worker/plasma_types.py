"""What Plasma's two halves exchange, in a module both import.

Only data crosses: a session's tick and hue one way, and its rendered frame
back, through shared memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTH = 320
HEIGHT = 192


@dataclass(frozen=True)
class Tick:
    """One session's next frame: where it is in the palette, and its offset."""

    tick: int
    hue: int


@dataclass(frozen=True)
class Frame:
    """The rendered frame, ``HEIGHT`` x ``WIDTH`` x 3 ``uint8``."""

    pixels: np.ndarray
