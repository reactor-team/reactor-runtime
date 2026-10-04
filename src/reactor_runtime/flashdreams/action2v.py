"""The ``action2v`` family: a world driven by keyboard and mouse, started from an image.

:class:`Action2V` is the application half and :class:`Action2VModel` the model
half for every FlashDreams model that joins the ``action2v`` family, Waypoint
among them. The family supplies the defaults a keyboard-and-mouse model needs:
the client's state and commands, a ``process_input()`` that turns the held keys
and the pointer motion since the last step into the model's own control type
through the adapter's action mapper, and an ``initialize_cache()`` that starts
a rollout from a seed image through the adapter's seed loader.

A workspace serves an ``action2v`` model by naming this class and the slug::

    # reactor.yaml
    runtime:
      import: reactor_runtime.flashdreams.action2v:Action2V
      config: config.yml

    # config.yml
    application: action2v-waypoint-1-5-1b
    example_image: true

For more control, subclass :class:`Action2V` and override ``process_input()``,
``process_output()``, or name a :class:`Action2VModel` subclass as
``model_class``.

``set_image`` decodes the upload with Pillow, which the FlashDreams model
packages depend on; a workspace that serves this family lists it in its
requirements.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reactor_runtime.flashdreams.app import FlashDreamsApp, FlashDreamsState
from reactor_runtime.flashdreams.contract import RolloutNotStarted
from reactor_runtime.flashdreams.model import FlashDreamsModel
from reactor_runtime.interface import (
    ApplicationError,
    CommandError,
    InputField,
    UploadedFile,
    event,
    session_started,
)


@dataclass(frozen=True)
class Action2VInput:
    """What one ``action2v`` step needs.

    Attributes:
        rollout_id: The rollout the application wants. An id the model does not
            hold starts a new rollout from ``image``.
        image: The seed image as encoded bytes, sent only until the model
            reports it holds ``rollout_id``; ``None`` on every other step.
        seed: The noise seed for a rollout that starts on this step.
        control: This step's actions in the model's own control type, built by
            the adapter's action mapper. The pipeline receives it as its input.
    """

    rollout_id: int
    image: bytes | None
    seed: int
    control: Any


class Action2VState(FlashDreamsState):
    """What a client can set on an ``action2v`` model, plus the session's private scratch.

    Keys are a field, because the model reads them as held until released.
    Pointer motion is not: it adds up between steps through the ``move``
    command and is spent on the next step.
    """

    keys: str = InputField(
        default="",
        max_length=256,
        description=(
            "Keys held down, comma-separated, for example `w,shift`. Letters, digits, "
            "`space`, `shift`, `ctrl`, `enter`, `tab`, and the arrow keys; held until changed."
        ),
    )

    _image: bytes | None = None
    _dx: float = 0.0
    _dy: float = 0.0
    _wheel: float = 0.0


class Action2VModel(FlashDreamsModel):
    """The ``action2v`` model half: a rollout starts from a seed image."""

    def initialize_cache(self, step: Action2VInput) -> Any:
        """Start a rollout from the step's seed image.

        Runs the adapter's ``seed_loader`` on the image, which decodes it and
        sizes it for the model, seeds the pipeline's noise generator with the
        step's ``seed``, and builds the pipeline's cache with the seed frames
        committed as the rollout's first action.

        Raises:
            RolloutNotStarted: The step carries no image.
        """
        if step.image is None:
            raise RolloutNotStarted("no seed image on the step that starts the rollout")
        # The adapter's loader takes a path, so the upload goes through a file.
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            file.write(step.image)
            file.flush()
            frames = self.app.defaults.seed_loader(Path(file.name), self.desc)
        frames = frames.to(self.pipeline.device, self.pipeline.diffusion_model.dtype)
        self.pipeline.diffusion_model.rng.manual_seed(step.seed)
        return self.pipeline.initialize_cache(seed_pixels=frames.add(1).mul(0.5).unsqueeze(0))


class Action2V(FlashDreamsApp):
    """A world started from an image and steered live with the keyboard and mouse.

    Commands: ``set_image``, ``set_keys``, ``move``, ``set_seed``, ``set_paused``,
    and ``reset``. Frames arrive on ``main_video`` at the size and rate the
    adapter declares.

    Config keys read from ``config.yml``, beyond the generic ones:
    ``example_image`` (default ``false``), which starts every session from the
    adapter's own example image, so a client sees a world before it uploads one.
    The adapter fetches that image the first time it is asked for and keeps it
    in the FlashDreams cache under the weights root; the fetch does not read
    ``HF_HUB_OFFLINE``, so a bundle filled with ``example_image: true`` holds
    the image and a bundle filled without it does not. ``load()`` fails with
    the reason when the image is missing and cannot be fetched.

    Attributes:
        default_image: The image every session starts from, or ``None``.
        map_action: The adapter's action mapper, from a held-keys-and-motion
            snapshot to the model's control type.
    """

    state: Action2VState
    family = "action2v"
    model_class = Action2VModel

    default_image: bytes | None
    map_action: Any

    def configure(self, fd_app: Any, config: dict[str, Any]) -> None:
        """Keep the adapter's action mapper, and its example image when asked for.

        Raises:
            RuntimeError: ``example_image`` is set and the adapter's example
                image is not in the weights bundle and could not be fetched.
        """
        self.default_image = None
        if config.get("example_image"):
            try:
                path = fd_app.defaults.input_resolver({"image_path": None, "example_data": True})
            except Exception as exc:
                raise RuntimeError(
                    "The adapter's example image is not in the weights bundle and could not "
                    "be fetched. FlashDreams fetches it on first use whatever HF_HUB_OFFLINE "
                    "says, so fill the bundle once with `example_image: true` and downloads "
                    "allowed, or set `example_image: false`."
                ) from exc
            self.default_image = Path(path).read_bytes()
        self.map_action = fd_app.defaults.action_mapper_factory(fd_app.session_desc(), 1.0)

    @session_started
    def start(self) -> None:
        """Start the session from the default image, when there is one."""
        self.state._image = self.default_image
        self.state._rollout_id += 1

    async def process_input(self) -> Action2VInput:
        """Refuse while paused or before a seed image; otherwise build the step.

        The image rides on the step only until the model reports it holds the
        rollout. The control is this step's held keys and the pointer motion
        accumulated since the last step, mapped by the adapter; taking it
        resets the motion.
        """
        if self.state.paused:
            raise ApplicationError("paused")
        if self.state._image is None:
            raise ApplicationError("no seed image")
        starting = self.state._rollout_id != self.state._applied_rollout_id
        return Action2VInput(
            rollout_id=self.state._rollout_id,
            image=self.state._image if starting else None,
            seed=self.state.seed,
            control=self.map_action(self._take_actions()),
        )

    def _take_actions(self) -> Any:
        """Snapshot the held keys and spend the pointer motion since the last step."""
        from action2v import ActionSnapshot  # type: ignore[ty:unresolved-import]

        keys = frozenset(key.strip() for key in self.state.keys.split(",") if key.strip())
        snapshot = ActionSnapshot(
            keys=keys,
            mouse_dx=self.state._dx,
            mouse_dy=self.state._dy,
            wheel_y=self.state._wheel,
        )
        self.state._dx = self.state._dy = self.state._wheel = 0.0
        return snapshot

    @event(
        name="set_image",
        description=(
            "Upload the image the world starts from. PNG or JPEG, up to 4096x4096 pixels. "
            "The next step starts a new world from it."
        ),
    )
    async def set_image(
        self, image: UploadedFile = InputField(description="The seed image.")
    ) -> None:
        """Check the upload decodes, off the loop, then make it the seed."""
        if not image.mime_type.startswith("image/"):
            raise CommandError("unsupported_media", f"{image.name} is not an image.")
        try:
            data = await asyncio.to_thread(_as_rgb_png, image.data)
        except ImportError:
            # Pillow is missing from the image: a deployment fault, not a bad
            # upload. The runtime answers internal_error and logs the traceback.
            raise
        except _ImageTooLargeError as exc:
            raise CommandError("image_too_large", str(exc)) from exc
        except Exception as exc:
            raise CommandError("undecodable_image", "The file is not a decodable image.") from exc
        self.state._image = data
        self.state._rollout_id += 1

    @event(
        name="move",
        description=(
            "Pointer and wheel motion since the last `move`, as a fraction of the frame: "
            "`dx` of its width, `dy` of its height, positive right and down. Adds up; the "
            "next step spends it."
        ),
    )
    def move(self, dx: float = 0.0, dy: float = 0.0, wheel: float = 0.0) -> None:
        """Add pointer and wheel motion for the next step."""
        self.state._dx += dx
        self.state._dy += dy
        self.state._wheel += wheel


MAX_SEED_PIXELS = 4096 * 4096
"""The most pixels a seed upload may hold.

The adapter sizes the seed to the model's frame itself, so nothing is gained
above this, and the bound is what keeps a small file that declares a huge
image from allocating hundreds of megabytes when it is decoded.
"""


class _ImageTooLargeError(ValueError):
    """The upload declares more pixels than :data:`MAX_SEED_PIXELS`."""


def _as_rgb_png(data: bytes) -> bytes:
    """Decode an upload, apply its EXIF orientation, and re-encode it as an RGB PNG.

    The adapter's seed loader accepts RGB and RGBA images and sizes them itself,
    so the seed is kept at its own size and only its mode is settled here. The
    declared size is checked against :data:`MAX_SEED_PIXELS` before any pixel
    is decoded.
    """
    from PIL import Image, ImageOps  # a dependency of the workspace, not of the runtime

    with Image.open(io.BytesIO(data)) as decoded:
        width, height = decoded.size
        if width * height > MAX_SEED_PIXELS:
            raise _ImageTooLargeError(
                f"The image is {width}x{height}; a seed image may hold at most "
                f"{MAX_SEED_PIXELS} pixels."
            )
        image = ImageOps.exif_transpose(decoded).convert("RGB")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
