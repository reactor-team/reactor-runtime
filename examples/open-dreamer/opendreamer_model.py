"""The OpenDreamer world model: weights, caches, and one autoregressive step.

Nothing in this file knows Reactor exists. It takes plain values — held keys as
strings, camera deltas as floats, a seed — and hands back a plain RGB frame, so
it runs in a script with Reactor uninstalled.

Everything that persists between steps is private to this class: the RNG, the
dynamics and tokenizer KV caches, how much of the conditioning window has been
observed, and how far into the rollout it is. The seed rides with every step, so
this class is also what notices that it changed and reopens the rollout — which
is right, because it is the only thing that knows what a rollout costs to
reopen.

What a world starts from is a conditioning window, and this class builds one out
of a video clip or a single image because doing so needs the checkpoint's frame
shape and action space. It holds no catalogue of them and no names for them: how
many starting points a client may choose between, what they are called, and
which one an empty choice resolves to are product decisions, so they belong to
the application, which hands the window it picked to :meth:`OpenDreamerModel.reset`.

The action space is the checkpoint's, so the mapping from ordinary key names to
VPT actions lives here as well; :data:`KEYS` and :data:`MOUSE_BUTTONS` are what
the model accepts, and the application publishes them as the values a client may
send.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from opendreamer_utils import (
    OpenDreamerConfig,
    RolloutConditioning,
    decode_conditioning_image,
    load_dependencies,
    mesh_context,
    prepare_process_environment,
    read_conditioning_sequence,
    read_config,
    upstream_root,
    verify_source_revision,
)

logger = logging.getLogger(__name__)

KEYS = (
    "w",
    "a",
    "s",
    "d",
    "space",
    "shift",
    "ctrl",
    "e",
    "q",
    "escape",
    "f",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "f3",
)
"""Keyboard keys the checkpoint's action space covers."""

MOUSE_BUTTONS = ("left", "right", "middle")
"""Mouse buttons the checkpoint's action space covers."""

_KEY_TO_VPT_NAME = {
    "w": "key.keyboard.w",
    "a": "key.keyboard.a",
    "s": "key.keyboard.s",
    "d": "key.keyboard.d",
    "space": "key.keyboard.space",
    "shift": "key.keyboard.left.shift",
    "ctrl": "key.keyboard.left.control",
    "e": "key.keyboard.e",
    "q": "key.keyboard.q",
    "escape": "key.keyboard.escape",
    "f": "key.keyboard.f",
    "1": "key.keyboard.1",
    "2": "key.keyboard.2",
    "3": "key.keyboard.3",
    "4": "key.keyboard.4",
    "5": "key.keyboard.5",
    "6": "key.keyboard.6",
    "7": "key.keyboard.7",
    "8": "key.keyboard.8",
    "9": "key.keyboard.9",
    "f3": "key.keyboard.f3",
}
_BUTTON_TO_VPT_NAME = {
    "left": "mouse.0",
    "right": "mouse.1",
    "middle": "mouse.2",
}


class OpenDreamerModel:
    """Generate Minecraft frames one autoregressive step at a time."""

    def __init__(self, checkpoint_cache: Path) -> None:
        """Bind where the checkpoint is cached.

        Args:
            checkpoint_cache: Directory the checkpoint is downloaded into. The
                caller owns it, because where weights live is a property of the
                host rather than of the model.
        """
        self._checkpoint_cache = checkpoint_cache
        self._config: OpenDreamerConfig | None = None
        self._deps: dict[str, Any] = {}
        self._mesh: Any = None
        self._tokenizer: Any = None
        self._dynamics: Any = None
        self._schedule: Any = None
        self._latent_shape: tuple[int, int, int, int] | None = None
        self._frame_shape: tuple[int, int, int] | None = None
        self._empty_dynamics_cache: Any = None
        self._empty_tokenizer_cache: Any = None
        self._next_frame_jit: Callable[..., Any] | None = None
        self._observe_frame_jit: Callable[..., Any] | None = None
        self._key_to_index: dict[str, int] = {}

        # Rollout state. `_conditioning` is what the world is built from, set by
        # reset(); None means there is nothing to generate from. `_opened` is the
        # seed the current rollout runs under, or None when the next step should
        # open one.
        self._opened: int | None = None
        self._rng: Any = None
        self._dynamics_cache: Any = None
        self._tokenizer_cache: Any = None
        self._conditioning: RolloutConditioning | None = None
        self._observed = 0
        self._step = 0

    # -- loading --------------------------------------------------------------

    def load(self, config: Path | None) -> None:
        """Load the pinned OpenDreamer source and its checkpoint, once.

        Args:
            config: Path to the model's YAML, which pins the upstream revision,
                the checkpoint, and the conditioning windows.

        Raises:
            RuntimeError: If no accelerator is available, or the checkpoint does
                not carry the models and action space this adapter expects.
        """
        settings = read_config(config)
        source_root = upstream_root()
        verify_source_revision(source_root, settings.source_revision)
        prepare_process_environment(settings)
        self._config = settings
        self._deps = load_dependencies(source_root)

        jax = self._deps["jax"]
        nnx = self._deps["nnx"]
        checkpoint_path = self._download_checkpoint(settings)
        if jax.default_backend() == "cpu":
            raise RuntimeError("OpenDreamer requires a CUDA accelerator")

        mesh, _data_sharding, mesh_rules = self._deps["build_parallel"]("data")
        self._mesh = mesh
        with mesh_context(jax, mesh):
            bundle = self._deps["bundle_type"].from_pretrained(
                checkpoint_path,
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(settings.seed),
                model_names={"dynamics_ema", "tokenizer"},
            )
            if bundle.dynamics_ema is None or bundle.tokenizer is None:
                raise RuntimeError("checkpoint does not contain dynamics_ema and tokenizer")
            self._dynamics = bundle.dynamics_ema
            self._tokenizer = bundle.tokenizer
            self._configure_inference(settings)
            self._warm_inference(settings)
            self._validate_action_space()

        logger.info(
            "OpenDreamer ready on %s with %d device(s), %d conditioning frames",
            jax.default_backend(),
            len(jax.devices()),
            settings.conditioning_frames,
        )

    def _download_checkpoint(self, settings: OpenDreamerConfig) -> str:
        """Fetch the pinned checkpoint into the cache the caller configured."""
        self._checkpoint_cache.mkdir(parents=True, exist_ok=True)
        return str(
            self._deps["snapshot_download"](
                repo_id=settings.checkpoint_repo_id,
                revision=settings.checkpoint_revision,
                cache_dir=self._checkpoint_cache,
            )
        )

    # -- the protocol ---------------------------------------------------------

    def generate(
        self,
        *,
        seed: int,
        keys: frozenset[str],
        buttons: frozenset[str],
        delta_x: float,
        delta_y: float,
        wheel_delta: int,
    ) -> np.ndarray | None:
        """Advance the world by one frame, or produce nothing yet.

        Produces nothing until :meth:`reset` has given it a conditioning window
        to build a world from, and nothing again while it observes that window,
        one frame per call — the world is not watchable until it has been
        observed. The seed rides with the step, so changing it opens the world
        again from the same window.

        Args:
            seed: The seed the rollout runs under.
            keys: Keyboard keys held for this frame. Names outside :data:`KEYS`
                are ignored.
            buttons: Mouse buttons held for this frame. Names outside
                :data:`MOUSE_BUTTONS` are ignored.
            delta_x: Horizontal camera movement for this frame.
            delta_y: Vertical camera movement for this frame.
            wheel_delta: Hotbar scroll for this frame; only its sign matters.

        Returns:
            The generated ``(height, width, 3)`` uint8 RGB frame, or ``None``
            while there is no world or the rollout is still being seeded.

        Raises:
            RuntimeError: If the model was never loaded.
        """
        if self._next_frame_jit is None or self._observe_frame_jit is None:
            raise RuntimeError("OpenDreamer was not loaded")
        if self._conditioning is None:
            return None
        jax = self._deps["jax"]

        with mesh_context(jax, self._mesh):
            if self._opened != seed:
                self._open_rollout(seed)
            if self._observed < self._conditioning.frames.shape[0]:
                self._observe_next_frame()
                return None
            return self._step_once(keys, buttons, delta_x, delta_y, wheel_delta)

    # -- this model's own API, called from the application --------------------

    def reset(self, conditioning: RolloutConditioning | None = None) -> None:
        """Discard the world and set what the next one is built from.

        Called instead of leaving the model to notice: the application knows when
        a client picked something different, and being told means the picking
        happens where a client is still waiting for an answer rather than inside
        a step with nobody to tell.

        Args:
            conditioning: The window the next world is built from, from
                :meth:`conditioning_from_clip` or
                :meth:`conditioning_from_image`. ``None`` leaves the model with
                no world, which generates nothing.

        Raises:
            RuntimeError: If the model was never loaded.
        """
        if self._config is None or self._frame_shape is None:
            raise RuntimeError("OpenDreamer was not loaded")
        self._opened = None
        self._observed = 0
        self._step = 0
        self._conditioning = conditioning

    def conditioning_from_clip(
        self,
        video: Path,
        actions: Path,
        *,
        start_frame: int = 0,
    ) -> RolloutConditioning:
        """Read one window of frames and aligned actions out of a gameplay clip.

        Args:
            video: The MP4 to read frames from.
            actions: The VPT action file recorded alongside it.
            start_frame: Where in the clip the window begins.

        Returns:
            A window :meth:`reset` accepts.

        Raises:
            RuntimeError: If the model was never loaded.
            ValueError: If the clip is too short or cannot be decoded.
        """
        if self._config is None or self._frame_shape is None:
            raise RuntimeError("OpenDreamer was not loaded")
        return read_conditioning_sequence(
            video,
            actions,
            self._frame_shape,
            start_frame=start_frame,
            required_frames=self._config.conditioning_frames,
            dependencies=self._deps,
        )

    def conditioning_from_image(self, image: bytes) -> RolloutConditioning:
        """Turn one still image into a conditioning window, held for its length.

        Args:
            image: The encoded image to build the window from. It is
                orientation-corrected, centre-cropped, and resized to the
                checkpoint's frame shape.

        Returns:
            A window :meth:`reset` accepts.

        Raises:
            RuntimeError: If the model was never loaded.
            ValueError: If the image cannot be decoded.
        """
        if self._config is None or self._frame_shape is None:
            raise RuntimeError("OpenDreamer was not loaded")
        frame = decode_conditioning_image(image, self._frame_shape)
        return RolloutConditioning(
            frames=np.repeat(frame[None], self._config.conditioning_frames, axis=0).copy(),
            actions=self._repeated_noop_actions(self._config.conditioning_frames),
        )

    # -- the rollout ----------------------------------------------------------

    def _open_rollout(self, seed: int) -> None:
        """Start the rollout under *seed*, from empty caches."""
        jax = self._deps["jax"]
        self._opened = seed
        self._rng = jax.random.PRNGKey(seed)
        self._dynamics_cache = self._empty_dynamics_cache
        self._tokenizer_cache = self._empty_tokenizer_cache
        self._observed = 0
        self._step = 0

    def _observe_next_frame(self) -> None:
        """Seed the caches with one conditioning frame and its aligned action."""
        assert self._conditioning is not None
        assert self._observe_frame_jit is not None
        jax = self._deps["jax"]
        jnp = self._deps["jnp"]
        self._dynamics_cache, self._tokenizer_cache = self._observe_frame_jit(
            self._tokenizer,
            self._dynamics,
            jnp.asarray(self._conditioning.frames[self._observed]),
            self._action_at(self._conditioning.actions, self._observed),
            self._dynamics_cache,
            self._tokenizer_cache,
        )
        jax.block_until_ready((self._dynamics_cache, self._tokenizer_cache))
        self._observed += 1

    def _step_once(
        self,
        keys: frozenset[str],
        buttons: frozenset[str],
        delta_x: float,
        delta_y: float,
        wheel_delta: int,
    ) -> np.ndarray:
        """Run one denoising step and return its decoded RGB frame."""
        assert self._latent_shape is not None
        assert self._next_frame_jit is not None
        jax = self._deps["jax"]
        action = self._build_action(keys, buttons, delta_x, delta_y, wheel_delta)
        self._rng, step_rng = jax.random.split(self._rng)
        frame, self._dynamics_cache, self._tokenizer_cache, self._rng = self._next_frame_jit(
            self._tokenizer,
            self._dynamics,
            action,
            self._latent_shape,
            self._dynamics_cache,
            self._tokenizer_cache,
            step_rng,
        )
        jax.block_until_ready(frame)
        self._step += 1
        pixels = np.asarray(frame[0, 0])
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(pixels)

    # -- inference setup ------------------------------------------------------

    def _configure_inference(self, config: OpenDreamerConfig) -> None:
        """Create schedules, empty caches, and compiled inference callables."""
        jnp = self._deps["jnp"]
        nnx = self._deps["nnx"]
        next_frame = self._deps["next_frame"]
        tokenizer_caches_type = self._deps["tokenizer_caches_type"]
        normalize_latents = self._deps["normalize_latents"]

        dynamics_config = self._dynamics.cfg
        tokenizer_config = self._tokenizer.cfg
        self._schedule = self._deps["schedule_type"].init(
            num_steps=config.num_steps,
            k_max=dynamics_config.k_max,
            tau_ctx_target=config.tau_ctx_target,
        )

        n_latents = int(tokenizer_config.decoder.n_latents)
        d_bottleneck = int(tokenizer_config.encoder.d_bottleneck)
        height = int(tokenizer_config.decoder.H)
        width = int(tokenizer_config.decoder.W)
        self._latent_shape = (1, 1, n_latents, d_bottleneck)
        self._frame_shape = (height, width, 3)

        self._empty_dynamics_cache = self._dynamics.create_static_caches(
            batch_size=1,
            n_latents=n_latents,
            window_size=int(dynamics_config.context_length),
            n_agent=0,
            dtype=dynamics_config.dtype,
        )
        self._empty_tokenizer_cache = self._tokenizer.create_static_caches(
            batch_size=1,
            H=height,
            W=width,
            window_size=int(tokenizer_config.decoder.context_length),
            dtype=tokenizer_config.decoder.dtype,
        )
        schedule = self._schedule

        def compiled_next_frame(
            tokenizer: Any,
            dynamics: Any,
            action: Any,
            latent_shape: tuple[int, int, int, int],
            dynamics_cache: Any,
            tokenizer_cache: Any,
            rng: Any,
        ) -> tuple[Any, Any, Any, Any]:
            frame, _hidden, new_dynamics_cache, decoder_cache, new_rng = next_frame(
                tokenizer,
                dynamics,
                schedule,
                action,
                latent_shape,
                dynamics_cache,
                tokenizer_cache.decoder,
                rng,
            )
            new_tokenizer_cache = tokenizer_caches_type(
                encoder=tokenizer_cache.encoder,
                decoder=decoder_cache,
            )
            return frame, new_dynamics_cache, new_tokenizer_cache, new_rng

        def compiled_observe_frame(
            tokenizer: Any,
            dynamics: Any,
            frame: Any,
            action: Any,
            dynamics_cache: Any,
            tokenizer_cache: Any,
        ) -> tuple[Any, Any]:
            video = jnp.asarray(frame, dtype=jnp.float32)[None, None, ...]
            latent, _, encoder_cache = tokenizer.encode(
                video,
                deterministic=True,
                caches=tokenizer_cache.encoder,
            )
            normalized = normalize_latents(
                latent,
                dynamics.cfg.latent_mean,
                dynamics.cfg.latent_std,
            )
            action_with_time = action[:, None, ...]
            step_indices = jnp.full((1, 1), schedule.emax, dtype=jnp.int32)
            tau_indices = jnp.full((1, 1), schedule.k_max, dtype=jnp.int32)
            _, (_, new_dynamics_cache) = dynamics(
                action_with_time,
                step_indices,
                tau_indices,
                normalized,
                deterministic=True,
                caches=dynamics_cache,
            )
            _, decoder_cache = tokenizer.decode(
                latent,
                caches=tokenizer_cache.decoder,
                deterministic=True,
            )
            new_tokenizer_cache = tokenizer_caches_type(
                encoder=encoder_cache,
                decoder=decoder_cache,
            )
            return new_dynamics_cache, new_tokenizer_cache

        self._next_frame_jit = nnx.jit(
            compiled_next_frame,
            static_argnames=("latent_shape",),
        )
        self._observe_frame_jit = nnx.jit(compiled_observe_frame)

    def _warm_inference(self, config: OpenDreamerConfig) -> None:
        """Compile the generation and conditioning paths before serving."""
        if config.warmup_steps == 0:
            return
        assert self._latent_shape is not None
        assert self._frame_shape is not None
        assert self._next_frame_jit is not None
        assert self._observe_frame_jit is not None
        jax = self._deps["jax"]
        jnp = self._deps["jnp"]
        rng = jax.random.PRNGKey(config.seed)
        dynamics_cache = self._empty_dynamics_cache
        tokenizer_cache = self._empty_tokenizer_cache
        noop = self._noop_action()
        for _ in range(config.warmup_steps):
            rng, step_rng = jax.random.split(rng)
            frame, dynamics_cache, tokenizer_cache, rng = self._next_frame_jit(
                self._tokenizer,
                self._dynamics,
                noop,
                self._latent_shape,
                dynamics_cache,
                tokenizer_cache,
                step_rng,
            )
            jax.block_until_ready((frame, dynamics_cache, tokenizer_cache, rng))
        zero_frame = jnp.zeros(self._frame_shape, dtype=jnp.uint8)
        observed = self._observe_frame_jit(
            self._tokenizer,
            self._dynamics,
            zero_frame,
            noop,
            self._empty_dynamics_cache,
            self._empty_tokenizer_cache,
        )
        jax.block_until_ready(observed)

    def _validate_action_space(self) -> None:
        """Verify the loaded source and checkpoint use the expected VPT action space."""
        source_mapping = dict(self._deps["key_to_index"])
        if len(source_mapping) != int(self._deps["binary_actions"]):
            raise RuntimeError("OpenDreamer source has an inconsistent binary action space")
        required = set(_KEY_TO_VPT_NAME.values()) | set(_BUTTON_TO_VPT_NAME.values())
        required |= {"mouse.wheel_neg", "mouse.wheel_pos", "unknown"}
        if required.difference(source_mapping):
            raise RuntimeError("OpenDreamer source is missing required VPT actions")
        if int(self._dynamics.cfg.num_binary_actions) != len(source_mapping):
            raise RuntimeError("checkpoint binary action count does not match the source")
        if int(self._dynamics.cfg.categorical_action_dim) != int(self._deps["camera_classes"]):
            raise RuntimeError("checkpoint camera action count does not match the source")
        self._key_to_index = source_mapping

    # -- actions --------------------------------------------------------------

    def _build_action(
        self,
        keys: frozenset[str],
        buttons: frozenset[str],
        delta_x: float,
        delta_y: float,
        wheel_delta: int,
    ) -> Any:
        """Build one upstream ``Actions`` value from this step's controls.

        Names the action space does not cover are ignored, so a client sending a
        key this checkpoint never learned changes nothing rather than failing.
        """
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        binary = np.zeros((1, len(self._key_to_index)), dtype=np.int32)
        for key in keys:
            vpt_name = _KEY_TO_VPT_NAME.get(key)
            if vpt_name is not None:
                binary[0, self._key_to_index[vpt_name]] = 1
        for button in buttons:
            vpt_name = _BUTTON_TO_VPT_NAME.get(button)
            if vpt_name is not None:
                binary[0, self._key_to_index[vpt_name]] = 1
        if wheel_delta < 0:
            binary[0, self._key_to_index["mouse.wheel_neg"]] = 1
        elif wheel_delta > 0:
            binary[0, self._key_to_index["mouse.wheel_pos"]] = 1
        categorical = self._deps["mouse_to_categorical"](
            np.asarray([delta_x], dtype=np.float32),
            np.asarray([delta_y], dtype=np.float32),
        )
        return action_type(
            binary=jnp.asarray(binary, dtype=jnp.int32),
            categorical=jnp.asarray(categorical, dtype=jnp.int32),
            continuous=None,
        )

    def _action_at(self, actions: Any, index: int) -> Any:
        """Remove the time dimension from one batched conditioning action."""
        action_type = self._deps["action_type"]

        def take(value: Any) -> Any:
            return None if value is None else value[:, index]

        return action_type(
            binary=take(actions.binary),
            categorical=take(actions.categorical),
            continuous=take(actions.continuous),
        )

    def _noop_action(self) -> Any:
        """Return one neutral upstream ``Actions`` value."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        camera_classes = int(self._deps["camera_classes"])
        return action_type(
            binary=jnp.zeros((1, len(self._key_to_index) or 27), dtype=jnp.int32),
            categorical=jnp.full((1,), camera_classes // 2, dtype=jnp.int32),
            continuous=None,
        )

    def _repeated_noop_actions(self, frames: int) -> Any:
        """Return a batched neutral action history for static image conditioning."""
        jnp = self._deps["jnp"]
        action_type = self._deps["action_type"]
        noop = self._noop_action()

        def repeat(value: Any) -> Any:
            return None if value is None else jnp.repeat(value[:, None, ...], frames, axis=1)

        return action_type(
            binary=repeat(noop.binary),
            categorical=repeat(noop.categorical),
            continuous=repeat(noop.continuous),
        )
