"""The ``cam2v`` family: a world described by a prompt, started from an image, flown by camera.

:class:`Cam2V` is the application half and :class:`Cam2VModel` the model half
for every FlashDreams model that joins the ``cam2v`` family, Lingbot World
among them. The family supplies the defaults a camera-driven model needs: the
client's state and commands, a ``process_input()`` that carries the prompt, the
held camera keys, and the calibration a rollout starts with, and a model half
that integrates the keys into camera poses with the adapter's own integrator
and starts a rollout from the prompt and the first frame.

A workspace serves a ``cam2v`` model by naming this class and the slug::

    # reactor.yaml
    runtime:
      import: reactor_runtime.flashdreams.cam2v:Cam2V
      config: config.yml

    # config.yml
    application: cam2v-lingbot
    example_data: true
    example_idx: 0

The adapter resolves the first frame, the prompt, the camera intrinsics, and
the world scale from the config: its own example data when ``example_data``
is set, else the ``prompt``, ``image_path``, ``intrinsic_path``, and
``pose_path`` or ``world_scale`` keys, with paths relative to the workspace.
A client then changes the prompt and the image, and flies the camera with the
keys; the intrinsics and the world scale stay those the adapter resolved.

``set_image`` decodes the upload with Pillow, which the FlashDreams model
packages depend on; a workspace that serves this family lists it in its
requirements.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reactor_runtime.flashdreams.app import FlashDreamsApp, FlashDreamsState
from reactor_runtime.flashdreams.contract import RolloutNotStarted
from reactor_runtime.flashdreams.model import FlashDreamsModel
from reactor_runtime.flashdreams.seed_image import ImageTooLargeError, as_rgb_png
from reactor_runtime.interface import (
    ApplicationError,
    CommandError,
    InputField,
    UploadedFile,
    event,
    session_started,
)

CAMERA_KEYS = frozenset({"w", "a", "s", "d", "q", "e", "i", "k", "j", "l"})
"""The keys the camera integrator reads; any other key is ignored.

``w``/``s`` move forward and back, ``q``/``e`` strafe, ``a``/``d`` and
``j``/``l`` turn, ``i``/``k`` look up and down.
"""

_RESOLVER_KEYS = (
    "prompt",
    "prompt_path",
    "image_path",
    "pose_path",
    "intrinsic_path",
    "world_scale",
    "example_data",
    "example_idx",
)
"""The ``config.yml`` keys handed to the adapter's conditioning resolver."""


@dataclass(frozen=True)
class Cam2VInput:
    """What one ``cam2v`` step needs.

    Attributes:
        rollout_id: The rollout the application wants. An id the model does not
            hold starts a new rollout from ``prompt`` and ``image``.
        prompt: The text the world follows. A new prompt within a rollout is
            swapped in place when the model can, and starts a new rollout
            otherwise.
        image: The first frame as encoded bytes, sent only until the model
            reports it holds ``rollout_id``; ``None`` on every other step.
        seed: The noise seed for a rollout that starts on this step.
        keys: The camera keys held during this step, from :data:`CAMERA_KEYS`.
        intrinsics: The camera intrinsics ``(fx, fy, cx, cy)`` in pixels, read
            when a rollout starts.
        world_scale: The scale the model's camera encoder applies to
            translations, read when a rollout starts.
    """

    rollout_id: int
    prompt: str
    image: bytes | None
    seed: int
    keys: frozenset[str]
    intrinsics: tuple[float, float, float, float]
    world_scale: float


class Cam2VState(FlashDreamsState):
    """What a client can set on a ``cam2v`` model, plus the session's private scratch."""

    prompt: str = InputField(
        default="",
        max_length=1000,
        moderate=True,
        description=(
            "The text the world follows. Changing it starts a new world from the current "
            "image on a model that cannot change its prompt mid-way."
        ),
    )
    keys: str = InputField(
        default="",
        max_length=64,
        description=(
            "Camera keys held down, comma-separated: `w`/`s` forward and back, `q`/`e` "
            "strafe, `a`/`d` or `j`/`l` turn, `i`/`k` look up and down. Held until changed."
        ),
    )

    _image: bytes | None = None
    _intrinsics: tuple[float, float, float, float] | None = None
    _world_scale: float | None = None


class Cam2VModel(FlashDreamsModel):
    """The ``cam2v`` model half: a rollout starts from a prompt and a first frame.

    The camera is rollout state: the adapter's pose integrator carries the
    camera from step to step, and a new rollout starts it again at the origin.
    Each step turns the keys held during it into one camera pose per output
    frame and hands those poses with the intrinsics to the adapter's
    ``generate_step`` hook, which is the pipeline's own ``generate()`` unless
    the adapter rewrites the camera input for its model first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pose_integrator: Any = None
        self.clock = 0.0
        self.intrinsics: np.ndarray | None = None
        self.world_scale = 0.0

    def reset(self) -> None:
        """Forget the rollout and the camera with it."""
        super().reset()
        self.pose_integrator = None
        self.clock = 0.0
        self.intrinsics = None
        self.world_scale = 0.0

    def initialize_cache(self, step: Cam2VInput) -> Any:
        """Start a rollout from the step's prompt and first frame.

        Loads the first frame with FlashDreams' own loader, which sizes it for
        the model, seeds the pipeline's generator with the step's ``seed`` (or
        torch's global one when the pipeline has none), starts the adapter's
        pose integrator at the origin, and builds the cache from the prompt
        and the frame.

        Raises:
            RolloutNotStarted: The step carries no image or an empty prompt.
        """
        if step.image is None:
            raise RolloutNotStarted("no first frame on the step that starts the rollout")
        if not step.prompt:
            raise RolloutNotStarted("no prompt on the step that starts the rollout")
        import torch  # type: ignore[ty:unresolved-import]  # installed in the image only
        from flashdreams.infra.runner_io import (  # type: ignore[ty:unresolved-import]
            load_first_frame_tensor,
        )

        defaults = self.app.defaults
        device = torch.device(self.device)
        # The loader takes a path, so the upload goes through a file.
        with tempfile.NamedTemporaryFile(suffix=".png") as file:
            file.write(step.image)
            file.flush()
            first_frame = load_first_frame_tensor(
                Path(file.name),
                pixel_height=int(self.desc.video_height),
                pixel_width=int(self.desc.video_width),
                device=device,
                dtype=defaults.first_frame_dtype,
                interpolation=defaults.first_frame_interpolation,
                install_hint=defaults.install_hint,
            )
        rng = self.pipeline.diffusion_model.rng
        if rng is not None:
            rng.manual_seed(step.seed)
        else:
            torch.manual_seed(step.seed)
        self.pose_integrator = defaults.pose_integrator_factory()
        self.clock = 0.0
        self.intrinsics = np.asarray(step.intrinsics, dtype=np.float32)
        self.world_scale = float(step.world_scale)
        return self.pipeline.initialize_cache(text=[step.prompt], image=first_frame)

    def _pipeline_input(self, step: Cam2VInput, index: int) -> Any:
        """Integrate this step's held keys into one camera pose per output frame.

        The pipeline says how many frames step *index* produces; the keys are
        held for that many frame periods at the adapter's rate, and the poses
        and intrinsics go to the pipeline on its device as FlashDreams'
        ``CameraControlInput``.
        """
        import torch  # type: ignore[ty:unresolved-import]  # installed in the image only
        from cam2v import CameraControlInput  # type: ignore[ty:unresolved-import]

        frame_count = int(self.pipeline.get_num_output_frames(index))
        if frame_count <= 0:
            raise ValueError(f"the pipeline reports {frame_count} output frames for step {index}")
        period = 1.0 / float(self.desc.frames_per_second_for_step)
        start = self.clock
        frame_times = [start + period * (k + 1) for k in range(frame_count)]
        poses = self.pose_integrator.integrate_chunk(
            segments=[(start, frame_times[-1], frozenset(step.keys))],
            frame_times=frame_times,
        )
        self.clock = frame_times[-1]
        device = self.pipeline.device
        intrinsics = torch.as_tensor(self.intrinsics).reshape(1, 4).repeat(frame_count, 1)
        return CameraControlInput(
            intrinsics=intrinsics.to(device=device, dtype=torch.float32),
            poses=torch.from_numpy(poses).to(device=device, dtype=torch.float32),
            world_scale=self.world_scale,
        )

    def _run_step(self, index: int, pipeline_input: Any) -> Any:
        """Run the step through the adapter's ``generate_step`` hook.

        Every cam2v adapter declares one. Most keep FlashDreams' default, the
        pipeline's own ``generate()``; an adapter whose model takes its own
        conditioning, such as SANA-WM, turns the camera input into it here
        and keeps the camera's history across the rollout.
        """
        return self.app.defaults.generate_step(self.pipeline, index, self.cache, pipeline_input)


class Cam2V(FlashDreamsApp):
    """A world described by a prompt, started from an image, and flown with the keyboard.

    Commands: ``set_prompt``, ``set_image``, ``set_keys``, ``set_seed``,
    ``set_paused``, and ``reset``. Frames arrive on ``main_video`` at the size
    and rate the adapter declares.

    Config keys read from ``config.yml``, beyond the generic ones, are the
    adapter's conditioning inputs: ``example_data`` and ``example_idx`` for
    its example data, or ``prompt`` (or ``prompt_path``), ``image_path``,
    ``intrinsic_path``, and ``pose_path`` or ``world_scale``. The adapter
    fetches example data the first time it is asked for and keeps it in the
    FlashDreams cache under the weights root; the fetch does not read
    ``HF_HUB_OFFLINE``, so a bundle filled with the deployment's settings holds
    it. ``load()`` fails with the reason when the conditioning cannot be
    resolved.

    Attributes:
        default_prompt: The prompt every session starts with.
        default_image: The first frame every session starts from.
        intrinsics: The camera intrinsics ``(fx, fy, cx, cy)`` the adapter resolved.
        world_scale: The world scale the adapter resolved.
    """

    state: Cam2VState
    family = "cam2v"
    model_class = Cam2VModel

    default_prompt: str
    default_image: bytes
    intrinsics: tuple[float, float, float, float]
    world_scale: float

    def configure(self, fd_app: Any, config: dict[str, Any]) -> None:
        """Resolve the first frame, prompt, and camera calibration through the adapter.

        Raises:
            RuntimeError: The adapter could not resolve its conditioning from
                the config: a file is missing, or example data is not in the
                weights bundle and could not be fetched.
        """
        desc = fd_app.session_desc()
        values = {key: config[key] for key in _RESOLVER_KEYS if key in config}
        values.update(
            pixel_height=int(desc.video_height),
            pixel_width=int(desc.video_width),
            fps=desc.frames_per_second_for_step,
        )
        try:
            conditioning = fd_app.defaults.input_resolver(values)
        except Exception as exc:
            raise RuntimeError(
                "The adapter could not resolve the world's starting conditions from "
                "config.yml. Set `example_data: true` to use its example data, or name "
                "`prompt`, `image_path`, `intrinsic_path`, and `pose_path` or `world_scale`. "
                "Example data is fetched on first use whatever HF_HUB_OFFLINE says; fill the "
                "weights bundle once with the deployment's settings and downloads allowed."
            ) from exc
        self.default_prompt = str(conditioning.prompt)
        self.default_image = Path(conditioning.first_frame_path).read_bytes()
        fx, fy, cx, cy = (float(v) for v in conditioning.base_intrinsics.reshape(-1).tolist())
        self.intrinsics = (fx, fy, cx, cy)
        self.world_scale = float(conditioning.world_scale)

    @session_started
    def start(self) -> None:
        """Start the session from the adapter's prompt, first frame, and calibration."""
        self.state.prompt = self.default_prompt
        self.state._image = self.default_image
        self.state._intrinsics = self.intrinsics
        self.state._world_scale = self.world_scale
        self.state._rollout_id += 1

    async def process_input(self) -> Cam2VInput:
        """Refuse while paused or without a prompt or image; otherwise build the step.

        The image rides on the step only until the model reports it holds the
        rollout. The keys are the ones held now; the model half turns them into
        camera poses for the frames this step produces.
        """
        if self.state.paused:
            raise ApplicationError("paused")
        if not self.state.prompt.strip():
            raise ApplicationError("no prompt set")
        if self.state._image is None:
            raise ApplicationError("no first frame")
        if self.state._intrinsics is None or self.state._world_scale is None:
            raise ApplicationError("no camera calibration")
        starting = self.state._rollout_id != self.state._applied_rollout_id
        keys = frozenset(key.strip().lower() for key in self.state.keys.split(",") if key.strip())
        return Cam2VInput(
            rollout_id=self.state._rollout_id,
            prompt=self.state.prompt.strip(),
            image=self.state._image if starting else None,
            seed=self.state.seed,
            keys=keys,
            intrinsics=self.state._intrinsics,
            world_scale=self.state._world_scale,
        )

    @event(
        name="set_image",
        description=(
            "Upload the image the world starts from. PNG or JPEG, up to 4096x4096 pixels. "
            "The next step starts a new world from it, with the current prompt."
        ),
    )
    async def set_image(
        self, image: UploadedFile = InputField(description="The first frame.")
    ) -> None:
        """Check the upload decodes, off the loop, then make it the first frame."""
        if not image.mime_type.startswith("image/"):
            raise CommandError("unsupported_media", f"{image.name} is not an image.")
        try:
            data = await asyncio.to_thread(as_rgb_png, image.data)
        except ImportError:
            # Pillow is missing from the image: a deployment fault, not a bad
            # upload. The runtime answers internal_error and logs the traceback.
            raise
        except ImageTooLargeError as exc:
            raise CommandError("image_too_large", str(exc)) from exc
        except Exception as exc:
            raise CommandError("undecodable_image", "The file is not a decodable image.") from exc
        self.state._image = data
        self.state._rollout_id += 1
