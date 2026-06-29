"""Brightness — a generated, client-controllable gradient, built on ReactorPipeline.

The smallest complete :class:`ReactorPipeline`: no model weights, no client
media. It generates an animated gradient on ``main_video`` and a matching sine
tone on ``main_audio``, driven entirely by a typed :class:`InputState`. The
public state fields become ``set_brightness`` / ``set_paused`` /
``set_resolution`` commands automatically — no handler boilerplate — and pausing
yields :data:`Idle` so the stream holds without producing frames.

It exercises the pipeline spine end to end: the ``inference()`` generator, typed
state with auto-generated setters, ``Idle`` skips, adaptive-free fixed FPS, and
multi-track (video + audio) output — all in pure NumPy, so it runs anywhere.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from reactor_runtime import (
    Audio,
    Idle,
    InputField,
    InputState,
    Output,
    ReactorPipeline,
    Video,
)
from reactor_runtime.interface.pipeline.idle import _IdleType
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

FPS = 30
SAMPLE_RATE = 48_000
FREQ_LOW = 200.0
FREQ_HIGH = 2000.0
TONE_AMPLITUDE = 8000

_RESOLUTIONS = {
    "480p": (480, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


class BrightnessOutput(Output):
    """The generated video and the matching audio tone."""

    main_video: Video
    main_audio: Audio


class BrightnessState(InputState):
    """The generation parameters a client can change live.

    Each public field becomes a ``set_<field>`` command automatically, so a
    client drives the look and pitch without the model declaring any handler.
    """

    brightness: float = InputField(
        default=1.0, ge=0.0, le=2.0, description="Brightness multiplier (0=black, 1=half, 2=white)."
    )
    paused: bool = InputField(default=False, description="Pause frame generation.")
    resolution: str = InputField(
        default="480p",
        choices=["480p", "720p", "1080p"],
        description="Output resolution.",
    )


class Brightness(ReactorPipeline):
    """Generate an animated gradient and tone whose look and pitch track the state."""

    state: BrightnessState
    output: BrightnessOutput
    fps = FPS
    buffer_size = 2

    def load(self, config_path: Path | None) -> None:
        """Nothing to load — the generator is pure NumPy. Reads no config."""
        logger.info("brightness pipeline ready")

    def inference(self) -> Iterator[BrightnessOutput | _IdleType]:
        """Emit a gradient frame and a tone each turn; yield Idle while paused."""
        frame_count = 0
        tone_phase = 0.0
        while True:
            if self.state.paused:
                yield Idle
                continue

            height, width = _RESOLUTIONS.get(self.state.resolution, (480, 640))
            brightness = self.state.brightness
            frame = _generate_frame(width, height, frame_count, brightness)
            audio, tone_phase = _generate_tone(brightness, tone_phase)
            frame_count += 1
            yield BrightnessOutput(main_video=frame, main_audio=audio)


def _generate_frame(width: int, height: int, frame_count: int, brightness: float) -> np.ndarray:
    """Build an animated gradient frame with the brightness multiplier applied.

    A vertical red ramp gives a fixed reference, a green bar sweeps across so
    motion is obvious, and the whole frame scales by *brightness* (base 128, so
    1.0 is half-bright and 2.0 saturates to white without clipping artefacts).
    """
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, 0] = np.linspace(0, 128, height, dtype=np.float32)[:, None]

    bar_x = (frame_count * 4) % width
    bar_w = max(1, width // 20)
    img[:, bar_x : min(width, bar_x + bar_w), 1] = 128

    frame: np.ndarray = np.clip(img * brightness, 0, 255).astype(np.uint8)
    return frame


def _generate_tone(brightness: float, phase: float) -> tuple[np.ndarray, float]:
    """Build one video-frame's worth of a sine tone whose pitch tracks brightness.

    Returns ``(1, N)`` int16 samples and the updated phase, so the tone stays
    continuous across frames.
    """
    n_samples = SAMPLE_RATE // FPS
    freq = FREQ_LOW + (brightness / 2.0) * (FREQ_HIGH - FREQ_LOW)
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    wave = np.sin(2.0 * math.pi * freq * t + phase) * TONE_AMPLITUDE
    next_phase = (phase + 2.0 * math.pi * freq * n_samples / SAMPLE_RATE) % (2.0 * math.pi)
    samples: np.ndarray = wave.astype(np.int16).reshape(1, -1)
    return samples, next_phase
