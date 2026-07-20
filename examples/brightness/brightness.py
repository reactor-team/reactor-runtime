"""Brightness — a generated, client-controllable gradient, built on ReactorPipeline.

A compact :class:`ReactorPipeline`: no model weights, no client media. It
generates an animated gradient on ``main_video`` and a matching sine tone on
``main_audio``, driven entirely by a typed :class:`InputState`. The public state
fields become ``set_paused`` / ``set_resolution`` / ``set_text`` commands
automatically — no handler boilerplate — and pausing yields :data:`Idle` so the
stream holds without producing frames. The resolution goes up to ``2160p``
(3840x2160), and a client-set caption is drawn over every frame.

It also shows commands that reply with a typed message, so the schema links a
command to its response: ``set_brightness`` overrides the auto-generated setter
to return a :class:`BrightnessSet` confirming the value now in effect, and
``set_image`` takes an uploaded file and returns an :class:`ImageSet` ack, while
``get_state`` returns a full :class:`BrightnessSnapshot`. Run this module directly
to print the model's OpenAPI schema and see each message defined once under
``components/schemas`` and referenced by ``$ref`` from both its webhook and the
command's ``responses.200``.

It exercises the pipeline spine end to end: the ``inference()`` generator, typed
state with auto-generated setters, ``Idle`` skips, adaptive-free fixed FPS, and
multi-track (video + audio) output. The frame synthesis is NumPy; the caption is
drawn with Pillow (``pillow``; see ``requirements.txt``).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reactor_runtime import (
    Audio,
    Idle,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    ReactorPipeline,
    UploadedFile,
    Video,
    event,
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
    "2160p": (2160, 3840),
}

# The caption height as a fraction of the frame height, so the text stays
# legible from 480p up to 2160p rather than shrinking to a few pixels at 4K.
_TEXT_HEIGHT_FRACTION = 0.05


class BrightnessOutput(Output):
    """The generated video and the matching audio tone."""

    main_video: Video
    main_audio: Audio


class BrightnessSnapshot(ModelMessage):
    """The live generation parameters, returned when a client asks for them."""

    brightness: float = MessageField(description="Active brightness multiplier.")
    paused: bool = MessageField(description="Whether frame generation is paused.")
    resolution: str = MessageField(description="Active output resolution.")
    text: str = MessageField(description="Caption currently drawn over each frame.")


class BrightnessSet(ModelMessage):
    """Confirmation that the brightness was applied, echoing the value in effect."""

    brightness: float = MessageField(description="Brightness multiplier now in effect.")


class ImageSet(ModelMessage):
    """Acknowledgement that an uploaded reference image was accepted."""

    filename: str = MessageField(description="Name of the image now in effect.")


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
        choices=["480p", "720p", "1080p", "2160p"],
        description="Output resolution (2160p is 4K UHD).",
    )
    text: str = InputField(
        default="",
        max_length=200,
        description="Caption drawn over every frame; empty draws nothing.",
    )


class Brightness(ReactorPipeline):
    """Generate an animated gradient and tone whose look and pitch track the state."""

    state: BrightnessState
    output: BrightnessOutput
    fps = FPS
    _reference: UploadedFile | None = None

    def load(self, config_path: Path | None) -> None:
        """Nothing to load — the generator is pure NumPy. Reads no config."""
        logger.info("brightness pipeline ready")

    @event(name="get_state", description="Return the current generation parameters.")
    def get_state(self) -> BrightnessSnapshot:
        """Reply to the caller with a snapshot of the live state.

        Returning a :class:`ModelMessage` makes this a request/response command:
        the schema renders its ``responses.200`` as a reference to the
        ``BrightnessSnapshot`` component.
        """
        return BrightnessSnapshot(
            brightness=self.state.brightness,
            paused=self.state.paused,
            resolution=self.state.resolution,
            text=self.state.text,
        )

    @event(name="set_brightness", description="Set the brightness and confirm the value in effect.")
    def set_brightness(
        self,
        brightness: float = InputField(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Brightness multiplier (0=black, 1=half, 2=white).",
        ),
    ) -> BrightnessSet:
        """Override the auto-generated setter to reply with a typed confirmation.

        Applies the value and returns a :class:`BrightnessSet`, so the schema
        renders this command's ``responses.200`` as a reference to that message.
        """
        self.state.brightness = brightness
        return BrightnessSet(brightness=brightness)

    @event(name="set_image", description="Set the reference image and acknowledge it.")
    def set_image(self, image: UploadedFile) -> ImageSet:
        """Accept an uploaded image and reply with a typed acknowledgement.

        The ``UploadedFile`` parameter makes the command carry an upload reference
        in its request body; returning an :class:`ImageSet` gives the client a
        confirmation carrying the accepted file's name.
        """
        self._reference = image
        logger.info("reference image set", name=image.name, size=len(image.data))
        return ImageSet(filename=image.name)

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
            if self.state.text:
                frame = _draw_text(frame, self.state.text)
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


def _draw_text(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw *text* near the bottom of the frame, outlined for legibility.

    The font scales with the frame height (:data:`_TEXT_HEIGHT_FRACTION`) so the
    caption reads the same at 480p and at 2160p.
    """
    height = frame.shape[0]
    font = ImageFont.load_default(size=max(12, int(height * _TEXT_HEIGHT_FRACTION)))
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    origin = (int(height * 0.02), height - int(height * 0.10))
    draw.text(
        origin,
        text,
        font=font,
        fill=(255, 255, 255),
        stroke_width=max(1, height // 240),
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(image)


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


if __name__ == "__main__":
    # Print the model's OpenAPI schema so the command/message/response wiring is
    # inspectable without standing up the server: `python brightness.py`.
    import json

    from reactor_runtime.interface.model import ModelContract

    print(json.dumps(ModelContract.of(Brightness).render_schema().to_openapi(), indent=2))
