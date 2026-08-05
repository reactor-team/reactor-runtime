"""Echo — receive the client's webcam and mic, apply a video effect, send both back.

A small but complete model: two inbound tracks (webcam + mic), two outbound
tracks (the processed video + the echoed audio), live commands to pick a video
effect and its intensity, and a typed message back when either changes. It
exercises the whole spine — inbound media, commands, lifecycle hooks, and
bidirectional A/V — with no model weights.

Effects use OpenCV (``opencv-python-headless``); see ``requirements.txt``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np

from reactor_runtime import (
    Audio,
    Input,
    InputBuffer,
    InputField,
    InputFrame,
    ModelMessage,
    Output,
    ReactorModel,
    ReadMode,
    UploadedFile,
    Video,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

Effect = Literal["none", "grayscale", "sepia", "edges", "invert", "blur", "pixelate"]
_EFFECTS = ["none", "grayscale", "sepia", "edges", "invert", "blur", "pixelate"]

_SEPIA_KERNEL = np.array(
    [
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ],
    dtype=np.float32,
)

SAMPLE_RATE = 48_000
FPS = 30

# Max audio samples held between video ticks. Two frames' worth at FPS absorbs
# normal jitter; a burst (e.g. after a client-side audio pause) is trimmed from
# the head so playback stays in real-time sync.
_MAX_BACKLOG_SAMPLES = int(2.0 / FPS * SAMPLE_RATE)


class EchoInput(Input):
    """The client's inbound webcam and microphone."""

    webcam: Video
    mic: Audio


class EchoOutput(Output):
    """The processed video and the echoed audio sent back."""

    main_video: Video
    main_audio: Audio


class EffectChanged(ModelMessage):
    """Sent when the active effect or intensity changes."""

    effect: str
    intensity: float


class Echo(ReactorModel):
    """Echo the client's A/V back, optionally applying a real-time video effect."""

    input: EchoInput
    fps = FPS

    def load(self, config_path: Path | None) -> None:
        """Start with no effect at full intensity and no overlay. Reads no config."""
        self.effect: Effect = "none"
        self.intensity: float = 1.0
        # An uploaded image blended over every output frame, set via
        # ``set_overlay_image``; ``None`` until a client uploads one.
        self._overlay: np.ndarray | None = None
        self._overlay_strength: float = 0.5
        # A free-text caption drawn over every output frame, set via
        # ``set_caption``; empty means no caption.
        self._caption: str = ""
        # Connected client count; the first client of a session resets shared state.
        self._connected_count = 0

    @session_started
    async def on_session_start(self) -> None:
        """Log the session boundary (fires once, before any client connects)."""
        logger.info("model lifecycle: session_started")

    @session_ended
    async def on_session_end(self) -> None:
        """Log the session boundary (fires once, after the last client leaves)."""
        logger.info("model lifecycle: session_ended")

    @connected
    async def on_connect(self) -> None:
        """Reset shared state for the first client of a session; log every join."""
        if self._connected_count == 0:
            self.effect = "none"
            self.intensity = 1.0
            self._overlay = None
            self._overlay_strength = 0.5
            self._caption = ""
        self._connected_count += 1
        logger.info("model lifecycle: connected", clients=self._connected_count)

    @disconnected
    async def on_disconnect(self) -> None:
        """Note a client leaving; log every leave."""
        self._connected_count = max(0, self._connected_count - 1)
        logger.info("model lifecycle: disconnected", clients=self._connected_count)

    @event(name="set_effect", description="Video effect to apply")
    async def set_effect(
        self, effect: str = InputField(default="none", choices=_EFFECTS)
    ) -> EffectChanged:
        """Pick the active video effect and acknowledge with the new state.

        Returning the message makes it the command's correlated reply, so a
        client awaiting the command resolves with the applied state.
        """
        # The contract bounds `effect` to _EFFECTS, so the widened str is one of
        # the Effect literals by the time the handler runs.
        self.effect = cast(Effect, effect)
        return EffectChanged(effect=self.effect, intensity=self.intensity)

    @event(name="set_intensity", description="Effect intensity (0=none, 1=full)")
    async def set_intensity(
        self, intensity: float = InputField(default=1.0, ge=0.0, le=1.0)
    ) -> EffectChanged:
        """Set the effect intensity and acknowledge with the new state."""
        self.intensity = intensity
        return EffectChanged(effect=self.effect, intensity=self.intensity)

    @event(name="set_caption", description="Draw a text caption over the output video")
    async def set_caption(
        self,
        caption: str = InputField(
            default="", max_length=200, description="Caption text; empty clears it"
        ),
    ) -> None:
        """Set the free-text caption drawn over every output frame.

        The one free-text command on this model: its value is user-authored
        prose rather than a typed knob, which also makes it the natural fixture
        for exercising content moderation of command text.
        """
        self._caption = caption

    @event(name="set_overlay_image", description="Blend an uploaded image over the output video")
    async def set_overlay_image(
        self,
        overlay_image: UploadedFile,
        overlay_strength: float = InputField(
            default=0.5, ge=0.0, le=1.0, description="Overlay opacity (0=hidden, 1=opaque)"
        ),
    ) -> None:
        """Decode an uploaded image and blend it over every output frame.

        The runtime resolves the upload reference to bytes before the handler
        runs, so ``overlay_image`` arrives as a fetched file. A non-image upload
        or one OpenCV cannot decode clears the overlay rather than failing.
        """
        self._overlay_strength = overlay_strength
        if not overlay_image.mime_type.startswith("image/"):
            logger.info(
                "ignoring non-image upload", name=overlay_image.name, mime=overlay_image.mime_type
            )
            self._overlay = None
            return
        decoded = cv2.imdecode(np.frombuffer(overlay_image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            logger.warning("could not decode uploaded image", name=overlay_image.name)
            self._overlay = None
            return
        self._overlay = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        logger.info("overlay image set", name=overlay_image.name, strength=overlay_strength)

    async def run(self) -> None:
        """Echo A/V in sync, applying the active effect to each video frame.

        Each tick reads one video frame, drains every queued mic chunk into a
        backlog, trims the backlog to ~2 video frames of audio (dropping bursts
        from client-side pauses), and emits both. Rate-matching the two streams
        keeps playback in sync on the client.
        """
        # The runtime binds a live InputBuffer to each declared input track; the
        # track annotations (Video/Audio) only carry the kind for the contract.
        webcam = cast(InputBuffer, self.input.webcam)
        mic = cast(InputBuffer, self.input.mic)

        audio_backlog: list[InputFrame] = []
        while True:
            await self.connected.wait()
            audio_backlog.clear()

            while self.connected.is_set():
                frames = webcam.try_read(1)
                if frames is None:
                    await asyncio.sleep(0)
                    continue

                # FIFO pops one chunk and leaves the rest; LATEST would clear the
                # buffer and drop every chunk but the newest, producing
                # sample-level discontinuities.
                while True:
                    next_chunks = mic.try_read(1, mode=ReadMode.FIFO)
                    if next_chunks is None:
                        break
                    audio_backlog.extend(next_chunks)

                _trim_backlog(audio_backlog, _MAX_BACKLOG_SAMPLES)
                aligned = audio_backlog[:]
                audio_backlog.clear()

                if aligned:
                    main_audio = np.concatenate(
                        [np.ascontiguousarray(c.data, dtype=np.int16).ravel() for c in aligned]
                    ).reshape(1, -1)
                else:
                    main_audio = np.zeros((1, 0), dtype=np.int16)

                processed = _apply_effect(frames[0].data, self.effect, self.intensity)
                if self._overlay is not None:
                    processed = _overlay_image(processed, self._overlay, self._overlay_strength)
                if self._caption:
                    processed = _draw_caption(processed, self._caption)
                await self.emit(EchoOutput(main_video=processed, main_audio=main_audio))


def _trim_backlog(backlog: list[InputFrame], max_samples: int) -> None:
    """Drop oldest chunks from *backlog* until total samples are within *max_samples*."""
    total = sum(c.data.size for c in backlog)
    while backlog and total > max_samples:
        total -= backlog.pop(0).data.size


def _draw_caption(frame: np.ndarray, caption: str) -> np.ndarray:
    """Draw *caption* near the bottom of the frame, outlined for legibility."""
    out = frame.copy()
    origin = (12, max(24, frame.shape[0] - 16))
    for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(
            out, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA
        )
    return out


def _overlay_image(frame: np.ndarray, overlay: np.ndarray, strength: float) -> np.ndarray:
    """Blend *overlay* over *frame* at *strength*, resized to the frame."""
    resized = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame, 1.0 - strength, resized, strength, 0).astype(np.uint8)


def _apply_effect(frame: np.ndarray, effect: Effect, intensity: float) -> np.ndarray:
    """Apply *effect* at *intensity* to an RGB frame, returning a new RGB frame."""
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
            frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        return frame

    if intensity >= 1.0:
        return processed
    return cv2.addWeighted(frame, 1.0 - intensity, processed, intensity, 0).astype(np.uint8)
