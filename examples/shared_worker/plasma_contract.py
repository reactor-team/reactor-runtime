"""What Plasma's two halves exchange: one request and one result type, imported by both.

The worker declares them (``request_type``, ``result_type``) and the session
half passes the same classes to ``WorkerClient.connect``. Only their fields
travel, the frame through shared memory; each side builds the type it declares
from them, and the worker refuses a session whose types differ from its own.
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
