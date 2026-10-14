"""Stepped echo: one frame back for every step, with a report of what the step saw.

A model built to be stepped. A client sends a step with webcam frames, state
values, and commands; the step answers with one frame on ``main_video`` and a
JSON report of what the model saw, on that frame's metadata and as the step's
``report.json``. Every value in the report is something the client sent, so a
client can check that its request arrived whole: the label it set, the tags it
sent with this step, how many frames it attached, the metadata on the last of
them, and the file it uploaded.

The class writes the three step hooks and leaves ``run()`` alone, so the same
code serves a streamed session too. There, each turn of the loop is one step:
the frames that arrived since the last turn are the step's frames, and a turn
that finds none sends the black frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from reactor_runtime import (
    InputBuffer,
    InputField,
    InputFrame,
    InputState,
    MediaInput,
    Output,
    ReactorApp,
    ReadMode,
    StepCompleted,
    StepOutcome,
    TrackPayload,
    UploadedFile,
    Video,
    event,
    session_started,
)

REPORT_FILE = "report.json"
"""The extra file each successful step keeps: the step's report, as JSON."""

BLACK_SIZE = 16
"""The side, in pixels, of the black frame a step with no webcam frame answers with."""


class SteppedEchoMedia(MediaInput):
    """The client's inbound webcam."""

    webcam: Video


class SteppedEchoOutput(Output):
    """The one frame each step sends back."""

    main_video: Video


class SteppedEchoState(InputState):
    """What a client can set. Each public field is a ``set_<field>`` command."""

    label: str = InputField(
        default="",
        description="Free text the model reports back on every step until it changes.",
    )
    overlay: UploadedFile | None = InputField(
        default=None,
        description="A file the model reports back by name and size on every step.",
    )


@dataclass(frozen=True)
class StepInput:
    """Everything one step works from, gathered by ``process_input``.

    Attributes:
        frames: The webcam frames this step carried, oldest first.
        label: The label in force for the step.
        tags: The values ``tag`` received for this step, in the order sent.
        overlay: The uploaded file in force for the step, if any.
        failure: The reason ``fail`` gave for this step, if it was sent.
    """

    frames: list[InputFrame]
    label: str
    tags: list[str]
    overlay: UploadedFile | None
    failure: str | None


@dataclass(frozen=True)
class StepResult:
    """What one step produced.

    Attributes:
        frame: The frame to send, uint8 ``(H, W, 3)`` RGB.
        report: The step's report, UTF-8 JSON.
        failure: The reason the step failed, if ``fail`` was sent for it.
    """

    frame: np.ndarray
    report: bytes
    failure: str | None


class SteppedEcho(ReactorApp):
    """Answer each step with its last webcam frame and a report of what it saw."""

    media: SteppedEchoMedia
    state: SteppedEchoState
    fps = 30

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.failure: str | None = None

    @event(name="tag", description="Attach a value to the current step's report.")
    def tag(self, value: str = InputField(description="The value to report.")) -> None:
        """Add *value* to the tags of the step this command precedes."""
        self.tags.append(value)

    @event(name="fail", description="Make the current step fail with a reason.")
    def fail(self, reason: str = InputField(description="The step's error.")) -> None:
        """Mark the step this command precedes as failed, with *reason* as its error."""
        self.failure = reason

    @session_started
    def on_session_started(self) -> None:
        """Start each session with no tags and no failure pending."""
        self.tags = []
        self.failure = None

    # -- the step -------------------------------------------------------------

    async def process_input(self) -> StepInput:
        """Take every queued webcam frame, the state, and this step's tags.

        The tags and the failure belong to one step, so they are handed over
        and cleared here; the next step starts without them. The label and the
        overlay are state, so every step reads them until a client changes
        them. A step never waits for a frame: one with none is still a step.
        """
        webcam = cast(InputBuffer, self.media.webcam)
        frames = webcam.try_read(webcam.available, mode=ReadMode.FIFO) or []
        step = StepInput(
            frames=frames,
            label=self.state.label,
            tags=self.tags,
            overlay=self.state.overlay,
            failure=self.failure,
        )
        self.tags = []
        self.failure = None
        return step

    def generate(self, input: StepInput) -> StepResult:
        """Pick the frame to send back and write the report."""
        if input.frames:
            frame = input.frames[-1].data
            last_metadata = input.frames[-1].metadata
        else:
            frame = np.zeros((BLACK_SIZE, BLACK_SIZE, 3), dtype=np.uint8)
            last_metadata = None
        report = build_report(input, last_metadata)
        return StepResult(frame=frame, report=report, failure=input.failure)

    async def process_output(self, outcome: StepOutcome) -> StepCompleted:
        """Send the frame with the report on it, and keep the report as a file.

        A step ``fail`` was sent for still sends its frame, and reports the
        reason as the step's error. It keeps no ``report.json``: the runtime
        keeps no files for a failed step.
        """
        if outcome.error is not None:
            raise outcome.error
        result: StepResult = outcome.result
        output = SteppedEchoOutput(main_video=TrackPayload(result.frame, metadata=result.report))
        if result.failure is not None:
            return StepCompleted(output=output, error=result.failure)
        return StepCompleted(output=output, files={REPORT_FILE: result.report})


def build_report(input: StepInput, last_metadata: bytes | None) -> bytes:
    """Return the step's report as UTF-8 JSON.

    The keys are ``label``, ``tags``, ``media_input`` (how many webcam frames
    the step carried), ``input_metadata`` (the last frame's metadata as text, or
    ``null``), ``overlay`` (the uploaded file's ``name``, ``mime_type``, and
    ``size``, or ``null``), and ``error`` (the reason ``fail`` gave, or
    ``null``).
    """
    overlay: dict[str, Any] | None = None
    if input.overlay is not None:
        overlay = {
            "name": input.overlay.name,
            "mime_type": input.overlay.mime_type,
            "size": input.overlay.size,
        }
    report = {
        "label": input.label,
        "tags": list(input.tags),
        "media_input": len(input.frames),
        "input_metadata": (
            None if last_metadata is None else last_metadata.decode("utf-8", errors="replace")
        ),
        "overlay": overlay,
        "error": input.failure,
    }
    return json.dumps(report).encode("utf-8")
