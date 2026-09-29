"""The generic FlashDreams model half, :class:`FlashDreamsModel`.

Plain Python around a FlashDreams pipeline. ``load()`` resolves an application
slug through FlashDreams' own registry and builds the adapter's pipeline on the
GPU; ``generate()`` runs one pipeline step and tracks the rollout it belongs to;
``reset()`` forgets the rollout. Nothing here knows about clients, tracks,
commands, or the runtime's loop, and nothing imports
``reactor_runtime.interface``, so the class also runs behind a
:class:`~reactor_runtime.distributed.DistributedRunner` unchanged.

A family subclass writes one method, :meth:`FlashDreamsModel.initialize_cache`,
which starts a rollout from what the step input carries. Everything else on this
class is the same for every FlashDreams model.

FlashDreams and torch are imported inside the methods that need them, so this
module imports, and its tests run, on a machine without either.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reactor_runtime.flashdreams.contract import (
    FlashDreamsResult,
    PromptSwapUnsupported,
    RolloutExhausted,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "FlashDreams is not installed. The workspace's requirements.txt installs its "
    "three packages from one pinned commit of https://github.com/NVIDIA/flashdreams, "
    "each with `#subdirectory=`: `flashdreams` (core, `flashdreams/`), the family "
    "package such as `flashdreams-action2v` (`apps/action2v`), and the model package "
    "such as `flashdreams-waypoint` (`integrations_v2/waypoint`)."
)

_loguru_forwarded = False


class FlashDreamsModel:
    """A FlashDreams pipeline behind ``load`` / ``generate`` / ``reset``.

    Holds the FlashDreams application the slug resolved to, its pipeline, the
    pipeline's per-rollout cache, and the id of the rollout that cache belongs
    to. A step input carries ``rollout_id``; an id the model does not hold
    starts a new rollout through :meth:`initialize_cache`. It may also carry
    ``prompt``, which a new value in the same rollout swaps in place when the
    model class supports it, and ``control``, which reaches the pipeline as its
    per-step ``input``.

    Attributes:
        app: The FlashDreams application the slug resolved to, after ``load()``.
        desc: Its session description: frame size, rate, and layout.
        pipeline: The built pipeline, on :attr:`device`.
        device: The device the pipeline is built on. A
            :class:`~reactor_runtime.distributed.DistributedRunner` sets it to
            this rank's GPU before ``load()`` runs; ``load()`` resolves it
            otherwise.
        max_blocks: The adapter's ``total_blocks``; a rollout ends there.
        cache: The pipeline's cache for the rollout the model holds, else ``None``.
        rollout_id: The id of that rollout, else ``None``.
        prompt: The prompt the rollout was started or last swapped with.
    """

    def __init__(self) -> None:
        self.app: Any = None
        self.desc: Any = None
        self.pipeline: Any = None
        self.device: str | None = None
        self.max_blocks = 0
        self.cache: Any = None
        self.rollout_id: int | None = None
        self.prompt: str | None = None

    def load(
        self,
        application: str,
        weights_root: str,
        device: str | None = None,
        warmup_steps: int = 0,
    ) -> None:
        """Resolve the slug and build its pipeline on the device.

        Points the FlashDreams and Hugging Face caches at *weights_root* before
        FlashDreams is imported, so the checkpoints the pipeline loads are read
        from the model's weights bundle. Forwards FlashDreams' loguru records to
        the standard ``logging`` tree, where the runtime stamps them.

        Args:
            application: The FlashDreams application slug, such as
                ``action2v-waypoint-1-5-1b``.
            weights_root: The directory the caches live under. ``FLASHDREAMS_CACHE_DIR``
                is set to it and ``HF_HUB_CACHE`` to its ``huggingface`` subdirectory.
                ``HF_HUB_OFFLINE`` is set to ``1`` unless the environment already
                sets it, so a run that fills an empty bundle can allow downloads.
            device: The device the pipeline is built on. ``None`` takes the
                :attr:`device` a ``DistributedRunner`` set for this rank, and
                ``"cuda"`` when nothing set one. Pass a device to override
                either.
            warmup_steps: Pipeline steps to run before the first client waits.
                The generic class cannot build a step input, so a value above
                zero requires a model class that overrides ``_warmup()``.

        Raises:
            ModuleNotFoundError: FlashDreams is not installed. The message names
                the packages a workspace installs.
            LookupError: No installed application matches the slug.
            TypeError: The slug resolved to an application without the adapter
                surface this class reads (``defaults``, ``pipeline_config``,
                ``session_desc()``).
            NotImplementedError: ``warmup_steps`` is above zero on a model class
                that does not warm up.
        """
        _point_caches_at(weights_root)
        _forward_loguru_to_logging()
        create_application = _registry()
        app = create_application(application)
        _require_adapter_surface(app, application)
        self.app = app
        self.desc = app.session_desc()
        self.max_blocks = int(app.defaults.total_blocks)
        self.device = device if device is not None else self.device or "cuda"
        self.pipeline = app.pipeline_config.setup().to(self.device).eval()
        self.reset()
        self._warmup(int(warmup_steps))

    def generate(self, step: Any) -> FlashDreamsResult:
        """Run one pipeline step for *step*.

        A ``rollout_id`` the model does not hold drops the current cache and
        starts a new rollout with :meth:`initialize_cache`. A new ``prompt`` in
        the rollout the model holds is swapped in place, or raises
        :class:`PromptSwapUnsupported`. The step's index is the pipeline's own
        count plus one; at the adapter's ``total_blocks`` the step raises
        :class:`RolloutExhausted` and the rollout is kept, so the application
        decides what follows. The pipeline's ``generate()`` then ``finalize()``
        run with the step's ``control`` as the pipeline input; a failure in
        either drops the half-written cache before it propagates.

        Args:
            step: The step input. Reads ``rollout_id``, and ``prompt`` and
                ``control`` when present.

        Returns:
            The frames as uint8 on the CPU, with the index and the rollout id.

        Raises:
            RuntimeError: ``load()`` has not run.
            RolloutNotStarted: Raised by :meth:`initialize_cache` when a new
                rollout is asked for and the step carries nothing to start it.
            RolloutExhausted: The rollout reached ``total_blocks``.
            PromptSwapUnsupported: The prompt changed within the rollout on a
                model that cannot swap it.
            Exception: Whatever the pipeline raised, after the cache is dropped.
        """
        if self.pipeline is None:
            raise RuntimeError("load() has not run")

        if step.rollout_id != self.rollout_id:
            # Free the rollout the model holds before building the next one, so
            # two caches never sit on the GPU at once.
            self.reset()
            self.cache = self.initialize_cache(step)
            self.rollout_id = step.rollout_id
            self.prompt = getattr(step, "prompt", None)
        else:
            prompt = getattr(step, "prompt", None)
            if prompt is not None and prompt != self.prompt:
                self._replace_prompt(step)
                self.prompt = prompt

        last = self.cache.autoregressive_index
        index = 0 if last is None else last + 1
        if index >= self.max_blocks:
            raise RolloutExhausted(index)
        try:
            video = self.pipeline.generate(
                autoregressive_index=index,
                cache=self.cache,
                input=getattr(step, "control", None),
            )
            self.pipeline.finalize(autoregressive_index=index, cache=self.cache)
        except Exception:
            self.reset()
            raise
        return FlashDreamsResult(frames=_to_frames(video), index=index, rollout_id=step.rollout_id)

    def reset(self) -> None:
        """Forget the rollout: no cache, no rollout id, no prompt. The pipeline stays."""
        self.cache = None
        self.rollout_id = None
        self.prompt = None

    def initialize_cache(self, step: Any) -> Any:
        """Start a rollout from *step* and return the pipeline's cache for it.

        The one method a family writes. Runs only on the step that starts a
        rollout, with the step input that asked for it. Build the pipeline's
        cache from what the step carries (a seed image, a prompt) and return
        it; raise :class:`RolloutNotStarted` when the step carries nothing to
        start a rollout from.

        Args:
            step: The step input that names a rollout the model does not hold.

        Returns:
            The cache ``generate()`` and ``finalize()`` thread through the rollout.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define initialize_cache(); the generic "
            "FlashDreamsModel does not know what starts a rollout."
        )

    def _replace_prompt(self, step: Any) -> None:
        """Swap the prompt within the rollout the model holds.

        A model class that can change its prompt in place overrides this. The
        default says the model cannot.
        """
        raise PromptSwapUnsupported()

    def _warmup(self, steps: int) -> None:
        """Run *steps* pipeline steps at load so compiled kernels are ready.

        The generic class has no step input to run, so it warms up nothing. A
        model class that can start a rollout on its own overrides this and
        calls ``reset()`` when it is done.
        """
        if steps <= 0:
            return
        raise NotImplementedError(
            f"{type(self).__name__} does not warm up; set warmup_steps to 0 or override _warmup()."
        )


def _point_caches_at(weights_root: str) -> None:
    """Point the FlashDreams and Hugging Face caches at *weights_root*.

    ``huggingface_hub`` reads its cache location once, when it is imported, so
    this runs before FlashDreams is. ``HF_HUB_OFFLINE`` is left alone when the
    environment sets it: a run that fills an empty bundle exports it as ``0``.
    """
    root = Path(weights_root)
    if "huggingface_hub" in sys.modules:
        logger.warning(
            "huggingface_hub was imported before the caches were pointed at the weights "
            "root; its cache location is already fixed",
            extra={"weights_root": str(root)},
        )
    os.environ["FLASHDREAMS_CACHE_DIR"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


class _PropagateHandler(logging.Handler):
    """A loguru sink that hands each record to the standard logger of the same name."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


def _forward_loguru_to_logging() -> None:
    """Route FlashDreams' loguru records into the standard ``logging`` tree.

    loguru writes to its own stderr sink and skips the ``logging`` root where
    the runtime stamps the session id on every record. Replacing that sink with
    a handler that re-emits through ``logging`` puts FlashDreams' lines on the
    same path as the model's own. Done once per process; a missing loguru
    means FlashDreams is not installed, which ``load()`` reports on its own.
    """
    global _loguru_forwarded
    if _loguru_forwarded:
        return
    try:
        from loguru import logger as loguru_logger  # type: ignore[ty:unresolved-import]
    except ImportError:
        return
    loguru_logger.remove()
    loguru_logger.add(_PropagateHandler(), format="{message}")
    _loguru_forwarded = True


def _registry() -> Any:
    """Import FlashDreams' ``create_application``, or say how to install FlashDreams."""
    try:
        from flashdreams.runtime_v2.application_registry import (  # type: ignore[ty:unresolved-import]  # installed in the image only
            create_application,
        )
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.partition(".")[0] == "flashdreams":
            raise ModuleNotFoundError(_INSTALL_HINT) from exc
        raise
    return create_application


def _require_adapter_surface(app: Any, slug: str) -> None:
    """Check that *app* carries what this class reads off a FlashDreams family app.

    ``defaults`` (with ``total_blocks``), ``pipeline_config``, and a callable
    ``session_desc`` are on every FlashDreams family application, but not on
    the ``IApplication`` base, so they are checked here rather than by type.
    """
    missing: list[str] = []
    defaults = getattr(app, "defaults", None)
    if defaults is None:
        missing.append("defaults")
    elif not hasattr(defaults, "total_blocks"):
        missing.append("defaults.total_blocks")
    if not hasattr(app, "pipeline_config"):
        missing.append("pipeline_config")
    if not callable(getattr(app, "session_desc", None)):
        missing.append("session_desc()")
    if missing:
        raise TypeError(
            f"{slug!r} resolved to {type(app).__name__}, which lacks "
            f"{', '.join(missing)}. reactor_runtime.flashdreams serves FlashDreams family "
            "applications (action2v, t2v, cam2v, v2v), which carry these."
        )


def _to_frames(video: Any) -> np.ndarray:
    """Turn the pipeline's ``[-1, 1]`` video tensor into uint8 frames on the CPU.

    Accepts ``[T, C, H, W]`` or ``[1, T, C, H, W]``, as the action2v pipelines
    return, and keeps the first three channels. Returns ``(T, H, W, 3)``.
    """
    import torch  # type: ignore[ty:unresolved-import]  # installed in the image only

    if video.ndim == 5:
        video = video[0]
    frames = video[:, :3].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()
