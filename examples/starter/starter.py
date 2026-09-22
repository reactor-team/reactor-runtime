"""Starter: the Reactor logo turning over a field of coarse static.

The smallest complete model. One class does everything: ``load`` reads the
config and the logo from the weights directory, ``generate`` renders one frame
from the settings the client holds, and a session end puts the animation back
to its first frame. Every public field on :class:`StarterState` is a command
the client can send, so the class writes no handler of its own.

This file keeps the model's own state (the angle, the static, the frame count)
next to the code that answers the client, in one class. That is the plainest
shape a model can take and the right one at this size. A model with weights
worth testing on their own keeps the two apart: see ``examples/echo`` and
``examples/waypoint``, and the ``application-model-isolation`` skill.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, UnidentifiedImageError

from reactor_runtime import (
    InputField,
    InputState,
    Output,
    ReactorApp,
    Video,
    get_weights_path,
    session_ended,
)

# A logger of the model's own. It propagates to the runtime's root handler, so
# every line lands in the session's log stream with the runtime's stamps.
logger = logging.getLogger(__name__)

# Colors from https://www.reactor.inc/brand.
GRADIENT_TOP = (0xFD, 0xF5, 0xC6)
GRADIENT_BOTTOM = (0xC7, 0xC0, 0x99)
STATIC_COLOR = (0x55, 0x52, 0x40)
PLACEHOLDER_COLOR = (0x00, 0x00, 0x00)


class StarterOutput(Output):
    """The rendered video, one frame per step."""

    main_video: Video


class StarterState(InputState):
    """What a client can set. Each public field is a ``set_<field>`` command."""

    spin_speed: float = InputField(
        default=0.2,
        ge=0.05,
        le=5.0,
        description="Spin speed in turns per second; use `set_paused` to stop it.",
    )
    paused: bool = InputField(
        default=False,
        description=(
            "True holds the logo at its current angle; false resumes. The static keeps moving."
        ),
    )
    static_interval: int = InputField(
        default=8,
        ge=1,
        le=300,
        description="Frames between two re-rolls of the static (1 = every frame).",
    )


class Starter(ReactorApp):
    """Spin the logo from the weights over static the client tunes.

    ``generate`` receives the live :class:`StarterState`, which is what the
    default ``process_input`` hands it, and returns a :class:`StarterOutput`,
    which the default ``process_output`` emits as it is. Nothing else of the
    step is written here.
    """

    state: StarterState
    fps = 30

    def load(self, config_path: Path | None) -> None:
        """Read the frame size from the config and the logo from the weights.

        Args:
            config_path: The file ``runtime.config`` in ``reactor.yaml`` names,
                or ``None`` for the defaults. Holds ``width``, ``height``, and
                the logo's file name. The logo is read from the weights
                directory ``get_weights_path()`` resolves; a missing or
                unreadable file draws a placeholder ring instead.
        """
        config: dict[str, Any] = {}
        if config_path is not None:
            config = yaml.safe_load(config_path.read_text()) or {}
        width = int(config.get("width", 800))
        height = int(config.get("height", 500))
        if width < 1 or height < 1:
            raise ValueError(f"frame size must be positive, got {width}x{height}")

        self.seconds_per_frame = 1.0 / self.fps
        self.static_cell = max(8, width // 20)
        self.background = render_gradient(width, height)
        # The logo fits half the frame width, or 70% of its height, whichever
        # binds first.
        box = (width // 2, height * 7 // 10)
        logo_file = get_weights_path() / str(config.get("logo", "logo.png"))
        self.logo = load_sprite(logo_file, box) or render_placeholder(box)
        self.rng = np.random.default_rng()
        self.reset()
        logger.info("starter model loaded: %dx%d", width, height)

    def generate(self, input: StarterState) -> StarterOutput:
        """Render one frame from the current settings and advance the animation."""
        if self.static is None or self.frames_since_roll >= input.static_interval:
            height, width, _ = self.background.shape
            self.static = roll_static(self.rng, width, height, self.static_cell)
            self.frames_since_roll = 0
        self.frames_since_roll += 1

        frame = self.background.copy()
        frame[self.static] = STATIC_COLOR
        draw_logo(frame, self.logo, math.cos(self.angle))

        if not input.paused:
            self.angle += 2 * math.pi * input.spin_speed * self.seconds_per_frame
        self.index += 1
        return StarterOutput(main_video=frame)

    def reset(self) -> None:
        """Return to the first frame: angle zero, index zero, fresh static."""
        self.angle = 0.0
        self.index = 0
        self.static: np.ndarray | None = None
        self.frames_since_roll = 0

    @session_ended
    def on_session_ended(self) -> None:
        """The next session starts the animation over."""
        self.reset()


# -- Rendering -----------------------------------------------------------------


def render_gradient(width: int, height: int) -> np.ndarray:
    """Build the background: a top-to-bottom ramp between the two brand tones."""
    ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    top = np.array(GRADIENT_TOP, dtype=np.float32)
    bottom = np.array(GRADIENT_BOTTOM, dtype=np.float32)
    gradient = top + (bottom - top) * ramp
    frame: np.ndarray = np.broadcast_to(gradient, (height, width, 3)).astype(np.uint8)
    return frame


def roll_static(
    rng: np.random.Generator, width: int, height: int, cell: int, density: float = 0.15
) -> np.ndarray:
    """Pick a fresh set of static blocks, as an ``(H, W)`` boolean mask.

    The blocks are *cell* pixels square, so the static reads as chunky noise
    behind a logo that stays at full resolution.
    """
    rows = -(-height // cell)
    cols = -(-width // cell)
    cells = rng.random((rows, cols)) < density
    mask: np.ndarray = np.repeat(np.repeat(cells, cell, axis=0), cell, axis=1)
    return mask[:height, :width]


def load_sprite(path: Path, box: tuple[int, int]) -> Image.Image | None:
    """Load the image at *path* as RGBA, scaled to fit inside *box*.

    The image keeps its aspect ratio, so one side of the box binds. An alpha
    channel, when present, is what the compositor uses, so a transparent PNG
    floats over the background.

    Returns ``None`` when there is no file at *path* or it is not an image, and
    logs why, so a workspace with no weights still starts.
    """
    if not path.is_file():
        logger.info("no image at %s, drawing the placeholder", path)
        return None
    try:
        with Image.open(path) as img:
            sprite = img.convert("RGBA")
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("unreadable image at %s (%s), drawing the placeholder", path, exc)
        return None
    scale = min(box[0] / sprite.width, box[1] / sprite.height)
    size = (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale)))
    return sprite.resize(size, Image.Resampling.LANCZOS)


def render_placeholder(box: tuple[int, int]) -> Image.Image:
    """Draw the stand-in for a missing image: a ring, as RGBA.

    It is visibly not the shipped logo, so a run without weights reads as one
    at a glance.
    """
    side = max(1, min(box))
    supersample = 4
    big = side * supersample
    img = Image.new("RGBA", (big, big), (*PLACEHOLDER_COLOR, 0))
    draw = ImageDraw.Draw(img)
    ring = max(1, big // 6)
    draw.ellipse((0, 0, big - 1, big - 1), fill=(*PLACEHOLDER_COLOR, 255))
    draw.ellipse((ring, ring, big - 1 - ring, big - 1 - ring), fill=(*PLACEHOLDER_COLOR, 0))
    return img.resize((side, side), Image.Resampling.LANCZOS)


def draw_logo(frame: np.ndarray, logo: Image.Image, horizontal_scale: float) -> None:
    """Composite the RGBA *logo* onto the center of *frame*, in place.

    *horizontal_scale* runs from 1 through 0 to -1 and back as the logo turns:
    the width shrinks to a sliver at 0 and the image is mirrored past it, which
    reads as a spin about the vertical axis.
    """
    full_w, logo_h = logo.size
    logo_w = max(1, round(full_w * abs(horizontal_scale)))
    view = logo.resize((logo_w, logo_h), Image.Resampling.BILINEAR)
    if horizontal_scale < 0:
        view = view.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    frame_h, frame_w, _ = frame.shape
    x0 = (frame_w - logo_w) // 2
    y0 = (frame_h - logo_h) // 2
    region = frame[y0 : y0 + logo_h, x0 : x0 + logo_w]

    rgba = np.asarray(view, dtype=np.float32)
    alpha = rgba[..., 3:] / 255.0
    region[...] = (region * (1.0 - alpha) + rgba[..., :3] * alpha).astype(np.uint8)
