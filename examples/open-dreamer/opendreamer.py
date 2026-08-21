"""Serve OpenDreamer as an interactive Minecraft world model.

The application half. It declares what a client may set, turns those values into
the arguments one model step wants, and puts the frame that comes back on a
track. It imports no JAX and touches no cache: the world model behind
``self.model`` owns the weights, the caches, and the decision of when a change
of seed is worth restarting its rollout for.

The scenes are this half's, because a scene is a product decision rather than a
model one: how many starting points a client may choose between, what they are
called, which one ``random`` resolves to, and that an upload counts as one. This
class reads the catalogue at load time, asks the model to build a conditioning
window per entry, and hands over the window a client picked. The model receives
windows and never learns their names.

Every control is a field on :class:`OpenDreamerState`, so the runtime generates,
validates, and documents a ``set_<field>`` command for each one, and a value a
client sets is simply read on the next generated frame.

Three commands are hand-written, because none of them is setting a value.
``load_scene`` and ``set_image`` choose what the world starts from — one names a
demo, the other carries a file — and ``reset`` restarts it. Until a scene has
been chosen, the mapping declines every step and nothing is generated.

There is no ``run()`` either. The runtime steps the model while a client is
connected and paces playout from how long each step took, which for this model
is its real frame rate.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np
from opendreamer_model import OpenDreamerModel
from opendreamer_types import OpenDreamerOutput, OpenDreamerState, StateUpdate
from opendreamer_utils import (
    DEMO_CHOICES,
    RolloutConditioning,
    ensure_demo_assets,
    read_config,
    upstream_asset,
    upstream_root,
)

from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    NotReady,
    ReactorModel,
    UploadedFile,
    connected,
    event,
    session_started,
)
from reactor_runtime.log import get_logger
from reactor_runtime.paths import get_weights_path

logger = get_logger(__name__)

RANDOM_SCENE = "random"
"""The scene a client names to be given one of the demos."""

UPLOADED_SCENE = "uploaded"
"""The scene reported while the world is running from an uploaded image."""


class OpenDreamer(ReactorModel):
    """Stream an interactive Minecraft rollout from a dataset demo or uploaded image."""

    state: OpenDreamerState
    model: OpenDreamerModel

    def load(self, config_path: Path | None) -> None:
        """Build the world model, load it, then build the scenes it can start from.

        The scenes come second because a conditioning window is cut to the
        checkpoint's frame shape, which is only known once the checkpoint is
        loaded.

        Args:
            config_path: Path to the model YAML named by ``reactor.yaml``.
        """
        self.model = OpenDreamerModel(
            checkpoint_cache=get_weights_path() / "open-dreamer" / "huggingface",
        )
        self.model.load(config_path)
        self._scenes: dict[str, RolloutConditioning] = self._load_scenes(config_path)

    def _load_scenes(self, config_path: Path | None) -> dict[str, RolloutConditioning]:
        """Read the configured demos and turn each one into a conditioning window.

        Args:
            config_path: Path to the model YAML, which names the demo clips.

        Returns:
            One window per demo, keyed by the name a client asks for.
        """
        source_root = upstream_root()
        demos = read_config(config_path).demos
        ensure_demo_assets(source_root, demos)
        scenes = {
            demo.name: self.model.conditioning_from_clip(
                upstream_asset(source_root, demo.video),
                upstream_asset(source_root, demo.actions),
                start_frame=demo.start_frame,
            )
            for demo in demos
        }
        logger.info("scenes ready", count=len(scenes), scenes=", ".join(scenes))
        return scenes

    def map_step(self, state: OpenDreamerState, input: None) -> dict:
        """Read the controls a client is holding and hand them to the model.

        Every line here is a read: the controls are level-triggered, so a step is
        a function of whatever they hold right now, and nothing about the world
        changes by asking what to generate. The keys and buttons arrive as the
        space-separated strings a client set and leave as sets of names; the
        model ignores any name its action space does not cover.

        There is no check for whether a scene has been chosen. The model produces
        nothing until it has one, and the runtime reads that as a declined step,
        so the condition lives in one place instead of two.

        Args:
            state: The world's controls, as the client last set them.
            input: Unused; this model takes no inbound media.

        Returns:
            The keyword arguments the model's ``generate`` takes.

        Raises:
            NotReady: Generation is paused, so there is no step to take.
        """
        if state.paused:
            raise NotReady("generation is paused")
        return {
            "seed": state.seed,
            "keys": frozenset(state.keys.split()),
            "buttons": frozenset(state.buttons.split()),
            "delta_x": state.look_x,
            "delta_y": state.look_y,
            "wheel_delta": state.hotbar,
        }

    def to_output(self, frame: np.ndarray) -> OpenDreamerOutput:
        """Put the generated frame on the video track.

        Args:
            frame: The RGB frame the model generated.

        Returns:
            The output to emit.
        """
        return OpenDreamerOutput(main_video=frame)

    async def on_state_changed(self, state: OpenDreamerState) -> None:
        """Tell every client what the world is set to, whenever that changes.

        The runtime calls this around each step, so a client learns about a
        change it did not make, and about a reset that stopped generation
        altogether — which no per-frame message could carry, because after a
        reset there are no frames.

        Args:
            state: The controls as they now stand.
        """
        await self.send(StateUpdate.from_state(state))

    @event(
        name="load_scene",
        description=(
            "Start a fresh world from one of the configured dataset demos, or `random` to let "
            "the model choose. Every control returns to its default first, so this is a clean "
            "start rather than a change of scenery. Frames begin once the scene's conditioning "
            "frames have been observed, which takes a moment."
        ),
    )
    def load_scene(
        self,
        scene: str = InputField(
            default=RANDOM_SCENE,
            choices=[*DEMO_CHOICES, RANDOM_SCENE],
            description="Dataset demo to start from, or `random` to let the model pick one.",
        ),
    ) -> None:
        """Start a fresh world from a dataset demo.

        Resolving `random` is this class's call, not the model's: which demo a
        client gets when it does not care is a product choice, and the model is
        handed the window that choice landed on.
        """
        self._reset_session()
        if scene == RANDOM_SCENE:
            scene = secrets.choice(sorted(self._scenes))
            logger.info("starting from a demo the model picked", scene=scene)
        self.state._scene = scene
        self.model.reset(self._scenes[scene])

    @event(
        name="set_image",
        description=(
            "Start a fresh world from an uploaded Minecraft screenshot. Every control returns "
            "to its default first. The image is orientation-corrected, center-cropped, and "
            "resized to the model's resolution; one that cannot be decoded is rejected with "
            "`invalid_image` and leaves the session reset. Frames begin once it has been "
            "observed, which takes a moment."
        ),
    )
    def set_image(
        self,
        image: UploadedFile = InputField(
            moderate=True,
            description="Minecraft screenshot to start the world from.",
        ),
    ) -> None:
        """Start a fresh world from an uploaded screenshot.

        Only the bytes are kept: what crosses into the model is plain data, never
        the upload itself. The window is built now, while a client is still
        waiting, so a bad image is answered rather than discovered later.

        Raises:
            CommandError: If the image cannot be decoded.
        """
        self._reset_session()
        try:
            conditioning = self.model.conditioning_from_image(image.data)
        except (ValueError, OSError) as error:
            raise CommandError("invalid_image", str(error)) from error
        self.model.reset(conditioning)
        self.state._scene = UPLOADED_SCENE

    @event(
        name="reset",
        description=(
            "Return everything to how a session starts: every control back to its default, no "
            "scene selected, and the world discarded. Nothing is generated afterwards until "
            "`load_scene` or `set_image` starts a world again."
        ),
    )
    def reset(self) -> None:
        """Put the session back to its factory state.

        A reset is an effect rather than a value, which is why it is a command
        and not a state field: the mapping that decides what to generate never
        changes anything.
        """
        self._reset_session()

    @session_started
    def on_session_started(self) -> None:
        """Start the session from a clean slate.

        The state arrives at its defaults, so the controls are already clear.
        What the runtime cannot know is that the world the model still holds
        belongs to the session that ended, and a new session must not inherit it.
        """
        self._reset_session()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Tell one joining viewer about the world it is joining."""
        await client.send(StateUpdate.from_state(self.state))

    def _reset_session(self) -> None:
        """Drop queued frames, return the controls to their defaults, discard the world.

        The three parts of the effect stay with their owners: cutting playout and
        replacing the controls are this class's, and what discarding a world
        costs is the model's own method. The runtime notices the state changed
        and broadcasts it, so clients hear about the reset without being told
        here.
        """
        self.output.flush()
        self.state = OpenDreamerState()
        self.model.reset()
