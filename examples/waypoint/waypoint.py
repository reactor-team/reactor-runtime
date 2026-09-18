"""Waypoint 1.5, the application half.

The :class:`ReactorApp` the runtime drives. It declares the client contract
(the state, the output track, the status message), owns ``process_input`` and
``process_output``, and holds the model half from ``waypoint_model.py`` under
``self.engine``. The two files meet on two dataclasses: the app builds a
:class:`WaypointInput` from the client's state, and reads a
:class:`WaypointResult` back.

Every public field on :class:`WaypointState` is a command the client can send.
The app writes two commands by hand: ``set_image`` takes the upload, decodes
and fits it, and keeps only the fitted frame as the seed, and ``reset``
restarts the world from the seed it already has.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from waypoint_model import WaypointInput, WaypointModel, WaypointResult

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    UploadedFile,
    Video,
    connected,
    event,
    get_logger,
    session_ended,
)

logger = get_logger(__name__)

WIDTH, HEIGHT = 1280, 720

# WASD shorthand to the Owl-Control VK keycodes the model reads
# (W=0x57, A=0x41, S=0x53, D=0x44).
_ACTION_KEYS: dict[str, frozenset[int]] = {
    "idle": frozenset(),
    "forward": frozenset({0x57}),
    "back": frozenset({0x53}),
    "left": frozenset({0x41}),
    "right": frozenset({0x44}),
    "forward_left": frozenset({0x57, 0x41}),
    "forward_right": frozenset({0x57, 0x44}),
    "back_left": frozenset({0x53, 0x41}),
    "back_right": frozenset({0x53, 0x44}),
}


class WaypointOutput(Output):
    """The generated world, four frames per step."""

    main_video: Video


class WaypointState(InputState):
    """What a client can set. Each public field is a ``set_<field>`` command."""

    paused: bool = InputField(
        default=False, description="Hold generation. The stream freezes on the last frame."
    )
    action: str = InputField(
        default="idle",
        choices=list(_ACTION_KEYS),
        description="Movement shorthand, applied on every step until changed.",
    )
    buttons: str = InputField(
        default="",
        max_length=128,
        description="Extra pressed keycodes as a comma-separated list, for example `32,16`.",
    )
    mouse_x: float = InputField(
        default=0.0,
        description="Horizontal mouse velocity, held until changed; 0 keeps the view still.",
    )
    mouse_y: float = InputField(
        default=0.0,
        description="Vertical mouse velocity, held until changed; 0 keeps the view still.",
    )
    scroll_wheel: int = InputField(
        default=0,
        ge=-1,
        le=1,
        description="Scroll direction, held until changed: -1 down, 0 none, 1 up.",
    )

    # Session scratch the client never sees: the seed frame, already decoded
    # and fitted; the id that tells one seed from the next; and the id of the
    # seed the model reported it holds, so the frame rides on a step input only
    # when the model has not applied it yet. The upload itself is not kept;
    # nothing reads it after the fit.
    _seed: np.ndarray | None = None
    _seed_id: int = 0
    _applied_seed_id: int | None = None

    def button_set(self) -> frozenset[int]:
        """Merge the action's keycodes with the extra buttons."""
        keys = set(_ACTION_KEYS.get(self.action, frozenset()))
        for token in self.buttons.split(","):
            token = token.strip()
            if token.isdigit() and 0 <= int(token) <= 255:
                keys.add(int(token))
        return frozenset(keys)


class WaypointStatus(ModelMessage):
    """A snapshot of the session, sent on connect, after a change, and on a cadence."""

    has_image: bool = MessageField(description="Whether a seed frame has been accepted.")
    paused: bool = MessageField(description="Whether generation is held.")
    step_index: int = MessageField(
        description="The last completed step within the current world, or -1 before the first."
    )

    @classmethod
    def of(cls, state: WaypointState, step_index: int) -> WaypointStatus:
        """Build the snapshot from the live state and the model's step index."""
        return cls(has_image=state._seed is not None, paused=state.paused, step_index=step_index)


class Waypoint(ReactorApp):
    """An explorable world seeded from an uploaded frame and driven by controls."""

    state: WaypointState

    def load(self, config_path: Path | None) -> None:
        """Construct the model half and load its weights once."""
        self.engine = WaypointModel()
        self.engine.load(config_path)
        self.progress_interval = 50
        self.last_index = -1

    # -- the step -------------------------------------------------------------

    async def process_input(self) -> WaypointInput:
        """Refuse while paused or before a seed; otherwise say what the model gets."""
        if self.state.paused:
            raise ApplicationError("paused")
        if self.state._seed is None:
            raise ApplicationError("no seed image")
        # The seed rides on the step input only when the model has not applied
        # this id yet; the step result reports which id the model holds.
        new_seed = self.state._seed_id != self.state._applied_seed_id
        return WaypointInput(
            buttons=self.state.button_set(),
            mouse=(self.state.mouse_x, self.state.mouse_y),
            scroll_wheel=self.state.scroll_wheel,
            seed=self.state._seed if new_seed else None,
            seed_id=self.state._seed_id,
        )

    def generate(self, input: WaypointInput) -> WaypointResult:
        """One step of the world. The model half does the work."""
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> WaypointOutput | None:
        """Tag the frames with their step and report progress on a cadence."""
        if outcome.error is not None:
            # No error the model raises is expected here: process_input() refuses
            # before a step without a seed can reach it, so NotSeeded would be a
            # bug in this file, and a CUDA failure is not something a reset
            # repairs. Re-raising ends the model loop and the session with an
            # error, which is better than serving a dead world in silence.
            raise outcome.error
        result: WaypointResult = outcome.result
        self.state._applied_seed_id = result.seed_id
        self.last_index = result.index
        if result.index % self.progress_interval == 0:
            await self.send(WaypointStatus.of(self.state, result.index))
        metadata = [{"step": result.index}] * result.frames.shape[0]
        return WaypointOutput(main_video=TrackPayload(result.frames, metadata=metadata))

    # -- commands and hooks ---------------------------------------------------

    @event(
        name="set_image",
        description=(
            "Upload the seed frame. The next step starts a new world from it. "
            "PNG or JPEG; the image is fitted to 1280x720."
        ),
    )
    async def set_image(
        self, image: UploadedFile = InputField(description="The seed frame.")
    ) -> WaypointStatus:
        """Decode and fit the upload off the loop, then make it the seed."""
        if not image.mime_type.startswith("image/"):
            raise CommandError("unsupported_media", f"{image.name} is not an image.")
        try:
            seed = await asyncio.to_thread(_fit, image.data)
        except Exception as exc:
            raise CommandError("undecodable_image", "The file is not a decodable image.") from exc
        self.state._seed = seed
        self.state._seed_id += 1
        self.output.flush()
        logger.info("seed accepted", name=image.name, seed_id=self.state._seed_id)
        return WaypointStatus.of(self.state, -1)

    @event(name="reset", description="Restart the world from the current seed frame.")
    def reset(self) -> WaypointStatus:
        """Forget the world. The next step sends the seed again."""
        self.engine.reset()
        self.state._applied_seed_id = None
        self.output.flush()
        self.last_index = -1
        return WaypointStatus.of(self.state, -1)

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        """Hand a joining client the current snapshot."""
        await client.send(WaypointStatus.of(self.state, self.last_index))

    @session_ended
    def on_session_ended(self) -> None:
        """A session end is the application's cue to reset the model."""
        self.engine.reset()
        self.output.flush()
        self.last_index = -1


def _fit(data: bytes) -> np.ndarray:
    """Decode an upload, honour its EXIF orientation, and fit it to the world's size."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    fitted = ImageOps.fit(image, (WIDTH, HEIGHT))
    return np.asarray(fitted, dtype=np.uint8)
