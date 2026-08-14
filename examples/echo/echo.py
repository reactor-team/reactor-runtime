"""Echo — receive the client's webcam and mic, apply a video effect, send both back.

A small but complete model: two inbound tracks (webcam + mic), two outbound
tracks (the processed video + the echoed audio), live commands to pick a video
effect and its intensity, and a typed message back when either changes. It
exercises the whole spine — inbound media, commands, lifecycle hooks, and
bidirectional A/V — with no model weights.

Per-frame metadata round-trips too: whatever a client attaches to a webcam frame
comes back on the processed frame it produced, so a client can correlate the two
without a side channel. Frames the client sent untagged come back untagged.

Effects use OpenCV (``opencv-python-headless``); see ``requirements.txt``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal, cast

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
    TrackPayload,
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
        """Set the starting effect state. Reads no config."""
        self._reset_state()

    def _reset_state(self) -> None:
        """Return the shared effect controls to their session defaults.

        Called at load and again at the start of every session, so an effect,
        caption, or overlay a previous session's clients set never carries into
        the next session on the same instance.
        """
        self.effect: Effect = "none"
        self.intensity: float = 1.0
        # How many frames pile up before an emit. One is a steady tick; more
        # makes the model produce the way a batching one does, in bursts.
        self.burst: int = 1
        # An uploaded image blended over every output frame, set via
        # ``set_overlay_image``; ``None`` until a client uploads one.
        self._overlay: np.ndarray | None = None
        self._overlay_strength: float = 0.5
        # A free-text caption drawn over every output frame, set via
        # ``set_caption``; empty means no caption.
        self._caption: str = ""

    @session_started
    async def on_session_start(self) -> None:
        """Reset the shared effect state for the session that is starting.

        Fires once, before any client connects, and owns the state shared by
        every client in the session. Per-client work belongs in ``on_connect``.
        """
        self._reset_state()
        logger.info("model lifecycle: session_started")

    @session_ended
    async def on_session_end(self) -> None:
        """Release the overlay the session held (fires once, after the last client leaves).

        A server-side session close tears every client down at once, so this
        runs even when no ``on_disconnect`` does.
        """
        self._overlay = None
        logger.info("model lifecycle: session_ended")

    @connected
    async def on_connect(self) -> None:
        """Note a client joining. Per client, and leaves shared state alone."""
        logger.info("model lifecycle: connected")

    @disconnected
    async def on_disconnect(self) -> None:
        """Note a client leaving. Per client, distinct from the session ending."""
        logger.info("model lifecycle: disconnected")

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

    @event(name="set_burst", description="Frames to batch into each emit")
    async def set_burst(
        self,
        burst: int = InputField(
            default=1, ge=1, le=30, description="1 emits every frame; higher emits in bursts"
        ),
    ) -> None:
        """Set how many frames pile up before an emit.

        A model that batches produces in bursts rather than on a steady tick,
        and the wire has to absorb the difference. Raising this turns the echo
        into that shape on demand — the same media, delivered unevenly — which
        is what makes the transport's pacing and gap-filling observable in a
        live session instead of only under a synthetic load.
        """
        self.burst = burst

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
        from client-side pauses), and pairs the two. Rate-matching the streams
        keeps playback in sync on the client.

        Frames leave in groups of ``burst`` — one by default, so every tick
        emits. A larger burst holds them back and sends them together, which is
        how a batching model produces and what the wire has to smooth out.

        The inbound frame's metadata rides back out on the frame produced from it.
        It is echoed as the bytes it arrived as — this model does not interpret
        it, so whatever the client encoded is what the client gets back.
        """
        # The runtime binds a live InputBuffer to each declared input track; the
        # track annotations (Video/Audio) only carry the kind for the contract.
        webcam = cast(InputBuffer, self.input.webcam)
        mic = cast(InputBuffer, self.input.mic)

        audio_backlog: list[InputFrame] = []
        # What a burst has accumulated so far: one entry per frame, in step.
        pending_video: list[np.ndarray] = []
        pending_audio: list[np.ndarray] = []
        pending_metadata: list[bytes | None] = []
        while True:
            await self.connected.wait()
            audio_backlog.clear()
            pending_video.clear()
            pending_audio.clear()
            pending_metadata.clear()

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

                frame = frames[0]
                processed = _apply_effect(frame.data, self.effect, self.intensity)
                if self._overlay is not None:
                    processed = _overlay_image(processed, self._overlay, self._overlay_strength)
                if self._caption:
                    processed = _draw_caption(processed, self._caption)
                pending_video.append(processed)
                pending_audio.append(main_audio)
                pending_metadata.append(frame.metadata)
                if len(pending_video) < self.burst:
                    continue

                # One emit carries the whole burst: the video frames stacked
                # into a batch and the audio they span concatenated. The runtime
                # splits both back apart and paces them out, so the media is the
                # same either way — only its arrival is lumpier.
                batched = np.stack(pending_video) if self.burst > 1 else pending_video[0]
                # A bare array when no frame in the burst carried anything, so
                # "attached nothing" stays a single case on the client too.
                video: np.ndarray | TrackPayload = batched
                if any(m is not None for m in pending_metadata):
                    # A burst needs one entry per frame, so frames that carried
                    # nothing take an empty trailer — which is how the runtime
                    # already spells "attached nothing" on the way back out.
                    metadata: bytes | list[dict[str, Any] | bytes] = (
                        [m or b"" for m in pending_metadata]
                        if self.burst > 1
                        else pending_metadata[0] or b""
                    )
                    video = TrackPayload(batched, metadata=metadata)
                await self.emit(
                    EchoOutput(
                        main_video=video,
                        main_audio=np.concatenate(pending_audio, axis=1),
                    )
                )
                pending_video.clear()
                pending_audio.clear()
                pending_metadata.clear()


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
