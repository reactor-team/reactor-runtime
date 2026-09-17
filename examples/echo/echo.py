"""Echo, the application half.

Receive the client's webcam and microphone, apply a video effect, and send
both back. The :class:`ReactorApp` the runtime drives: it declares the two
inbound tracks, the two outbound tracks, and the settings a client can change,
and holds the model half from ``echo_model.py`` under ``self.engine``. The two
files meet on two dataclasses: the app builds an :class:`EchoInput` from the
newest webcam frame and the state, and reads an :class:`EchoResult` back.

Two things this example shows that the others do not. Inbound media:
``process_input`` refuses a step until a webcam frame has arrived, and drains
the microphone into the audio that plays alongside the frame it pairs with.
And batching: ``burst`` frames pile up before one emit, so ``process_output``
returns ``None`` until a batch is full, which is how a model that produces in
bursts drives the same loop.

Per-frame metadata round-trips: whatever a client attaches to a webcam frame
comes back on the frame it produced, so a client can correlate the two
without a side channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from echo_model import EFFECTS, EchoInput, EchoModel, EchoResult

from reactor_runtime import (
    ApplicationError,
    Audio,
    InputBuffer,
    InputField,
    InputFrame,
    InputState,
    MediaInput,
    Output,
    ReactorApp,
    ReadMode,
    StepOutcome,
    TrackPayload,
    Video,
    session_ended,
)

SAMPLE_RATE = 48_000
FPS = 30

# The most audio one step may carry: two video frames' worth at FPS. That
# absorbs normal jitter; a burst after a client-side audio pause is trimmed
# from the head so playback stays in real-time sync.
_MAX_AUDIO_SAMPLES = int(2.0 / FPS * SAMPLE_RATE)


class EchoMedia(MediaInput):
    """The client's inbound webcam and microphone."""

    webcam: Video
    mic: Audio


class EchoOutput(Output):
    """The processed video and the echoed audio sent back."""

    main_video: Video
    main_audio: Audio


class EchoState(InputState):
    """What a client can set. Each public field is a ``set_<field>`` command."""

    effect: str = InputField(
        default="none", choices=EFFECTS, description="Video effect applied to every frame."
    )
    intensity: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Effect intensity: 0 leaves the frame as is, 1 is full.",
    )
    caption: str = InputField(
        default="",
        max_length=200,
        moderate=True,
        description="Text drawn over every frame; empty clears it.",
    )
    burst: int = InputField(
        default=1,
        ge=1,
        le=120,
        description=(
            "Frames batched into one emit: 1 sends every frame; 120 is four seconds at 30 fps."
        ),
    )

    # Session scratch the client never sees: what `process_input` read alongside
    # the frame, for `process_output` to pair with the result.
    _audio: np.ndarray | None = None
    _metadata: bytes | None = None


class Echo(ReactorApp):
    """Echo the client's A/V back, with a real-time video effect."""

    media: EchoMedia
    state: EchoState
    fps = FPS

    def load(self, config_path: Path | None) -> None:
        """Construct the model half. There are no weights and no config to read."""
        self.engine = EchoModel()
        self.engine.load()
        self.burst = _Burst()

    # -- the step -------------------------------------------------------------

    async def process_input(self) -> EchoInput:
        """Refuse until a webcam frame is here; otherwise say what the model gets.

        The newest webcam frame is the step. The microphone is drained in
        arrival order and kept on the state, with the frame's metadata, for
        ``process_output`` to pair with the result.
        """
        webcam = cast(InputBuffer, self.media.webcam)
        mic = cast(InputBuffer, self.media.mic)
        frames = webcam.try_read(1)
        if frames is None:
            raise ApplicationError("waiting for a webcam frame")
        frame = frames[0]
        self.state._audio = _drain(mic)
        self.state._metadata = frame.metadata
        return EchoInput(
            frame=frame.data,
            effect=self.state.effect,
            intensity=self.state.intensity,
            caption=self.state.caption,
        )

    def generate(self, input: EchoInput) -> EchoResult:
        """One frame. The model half does the work."""
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> EchoOutput | None:
        """Pair the frame with its audio and metadata; emit once a batch is full.

        A batch is one array, so every frame in it has the same size. WebRTC
        rescales an inbound track as bandwidth and CPU move, so a resolution
        change lands mid-burst; it ends the batch, and the new size opens the
        next one.
        """
        if outcome.error is not None:
            raise outcome.error
        result: EchoResult = outcome.result
        audio = self.state._audio if self.state._audio is not None else _silence()

        if self.burst.frames and result.frame.shape != self.burst.frames[0].shape:
            early = self.burst.take()
            self.burst.add(result.frame, audio, self.state._metadata)
            return early

        self.burst.add(result.frame, audio, self.state._metadata)
        if len(self.burst) >= self.state.burst:
            return self.burst.take()
        return None

    # -- lifecycle ------------------------------------------------------------

    @session_ended
    def on_session_ended(self) -> None:
        """Drop a half-gathered batch and reset the model half."""
        self.burst.clear()
        self.engine.reset()


class _Burst:
    """The frames gathered for one emit, with the audio and metadata they carry."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.audio: list[np.ndarray] = []
        self.metadata: list[bytes | None] = []

    def __len__(self) -> int:
        return len(self.frames)

    def add(self, frame: np.ndarray, audio: np.ndarray, metadata: bytes | None) -> None:
        """Gather one frame and what plays and rides with it."""
        self.frames.append(frame)
        self.audio.append(audio)
        self.metadata.append(metadata)

    def take(self) -> EchoOutput:
        """Build the emit from what was gathered and leave the burst empty.

        One frame goes out as a plain frame; more go out as one batch, the
        video stacked and the audio concatenated. The runtime splits both back
        apart and paces them out, so the media is the same as a frame-at-a-time
        model would send; only its arrival is lumpier.
        """
        batched = np.stack(self.frames) if len(self.frames) > 1 else self.frames[0]
        video: np.ndarray | TrackPayload = batched
        if any(entry is not None for entry in self.metadata):
            # A batch needs one entry per frame, so a frame that carried nothing
            # takes an empty trailer, which is how the runtime spells "attached
            # nothing" on the way back out.
            metadata: bytes | list[dict[str, Any] | bytes] = (
                [entry or b"" for entry in self.metadata]
                if len(self.frames) > 1
                else self.metadata[0] or b""
            )
            video = TrackPayload(batched, metadata=metadata)
        output = EchoOutput(main_video=video, main_audio=np.concatenate(self.audio, axis=1))
        self.clear()
        return output

    def clear(self) -> None:
        """Forget what was gathered."""
        self.frames.clear()
        self.audio.clear()
        self.metadata.clear()


def _drain(mic: InputBuffer) -> np.ndarray:
    """Take every queued microphone chunk, in arrival order, as one ``(1, N)`` array.

    FIFO pops chunks in order and keeps the samples continuous; LATEST would
    drop every chunk but the newest and put a click at each seam. The head is
    trimmed to :data:`_MAX_AUDIO_SAMPLES`, so a backlog after a client-side
    pause does not play out late.
    """
    chunks: list[InputFrame] = []
    while (more := mic.try_read(1, mode=ReadMode.FIFO)) is not None:
        chunks.extend(more)
    total = sum(chunk.data.size for chunk in chunks)
    while chunks and total > _MAX_AUDIO_SAMPLES:
        total -= chunks.pop(0).data.size
    if not chunks:
        return _silence()
    return np.concatenate(
        [np.ascontiguousarray(chunk.data, dtype=np.int16).ravel() for chunk in chunks]
    ).reshape(1, -1)


def _silence() -> np.ndarray:
    """No audio for this frame: an empty ``(1, 0)`` int16 array."""
    return np.zeros((1, 0), dtype=np.int16)
