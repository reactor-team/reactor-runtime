"""The generic FlashDreams application half, :class:`FlashDreamsApp`.

The :class:`~reactor_runtime.ReactorApp` every FlashDreams family builds on. It
writes the scaffold that is the same for every model: ``load()`` resolves the
slug, checks it belongs to the class's family, pins playout to the adapter's
frame rate, and builds the model half, in this process or behind a
:class:`~reactor_runtime.distributed.DistributedRunner`; ``generate()`` forwards
one step to it; ``process_output()`` turns a :class:`FlashDreamsResult` or one
of the model's three errors into what the client receives; the ``reset``
command and the session-end hook manage the rollout.

A family class adds the client's state and commands and writes
``process_input()``. A workspace names the family class in ``reactor.yaml``
and the slug in ``config.yml``, and writes no Python.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar

import yaml

from reactor_runtime.distributed import DistributedRunner
from reactor_runtime.flashdreams.contract import (
    FlashDreamsResult,
    PromptSwapUnsupported,
    RolloutExhausted,
)
from reactor_runtime.flashdreams.model import FlashDreamsModel, _point_caches_at, _registry
from reactor_runtime.interface import (
    InputField,
    InputState,
    MessageField,
    Metadata,
    ModelMessage,
    Output,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    Video,
    event,
    session_ended,
)
from reactor_runtime.paths import get_weights_path


class FlashDreamsState(InputState):
    """The client-settable state every FlashDreams family shares.

    A family extends it with its own fields. The two private fields are the
    rollout handshake between the two halves: ``_rollout_id`` is the rollout
    the application wants, and ``_applied_rollout_id`` is the one the model
    last reported holding. A command that starts over bumps the first; the
    default ``process_output()`` records the second off each result.
    """

    paused: bool = InputField(
        default=False, description="Hold generation. The stream freezes on the last frame."
    )
    seed: int = InputField(default=42, ge=0, description="Noise seed for the next rollout.")

    _rollout_id: int = 0
    _applied_rollout_id: int | None = None


class FlashDreamsOutput(Output):
    """The generated video, one chunk of frames per step."""

    main_video: Video


class RolloutRestarted(ModelMessage):
    """The rollout reached the model's limit and a new one starts from the same starting point."""

    steps: int = MessageField(description="How many steps the rollout ran before its limit.")


class FlashDreamsApp(ReactorApp):
    """Serve a FlashDreams model from its application slug.

    Subclass a family class such as ``Action2V``, or name it directly in
    ``reactor.yaml``, and set the slug on the class or in ``config.yml``. The
    family class supplies the state, the commands, and ``process_input()``;
    this class supplies everything else a step passes through.

    Class attributes:
        application: The FlashDreams application slug, such as
            ``action2v-waypoint-1-5-1b``. ``None`` reads ``application`` from
            ``config.yml``.
        family: The FlashDreams family this class serves: ``action2v``, ``t2v``,
            ``cam2v``, or ``v2v``. Set by the family class. ``load()`` fails
            when the slug resolves to another family's application.
        model_class: The model half ``load()`` constructs, a
            :class:`FlashDreamsModel` subclass. Set by the family class.
        world_size: GPUs to run the model half on. Above one, the model runs
            behind a ``DistributedRunner`` with one process per GPU.
        isolate: On one GPU, run the model half in its own process, so a hard
            crash in FlashDreams reaches ``process_output()`` as an error
            instead of ending this process.

    Config keys read from ``config.yml``: ``application`` when the class sets
    none, and ``warmup_steps`` (default ``0``), passed to the model half.

    Attributes:
        engine: The model half after ``load()``: a ``model_class`` instance, or
            the ``DistributedRunner`` around one.
    """

    state: FlashDreamsState

    application: ClassVar[str | None] = None
    family: ClassVar[str]
    model_class: ClassVar[type[FlashDreamsModel]]
    world_size: ClassVar[int] = 1
    isolate: ClassVar[bool] = False

    engine: Any

    def load(self, config_path: Path | None) -> None:
        """Resolve the slug, check its family, pin playout, and build the model half.

        The FlashDreams and Hugging Face caches are pointed at the weights root
        before FlashDreams is imported here, so the checkpoints its pipeline
        loads in this process come from the model's weights bundle. Playout is
        pinned to the adapter's ``frames_per_second_for_step``, so every model
        plays at the rate it was trained for.

        Args:
            config_path: The file ``runtime.config`` in ``reactor.yaml`` names,
                or ``None`` when the manifest names none.

        Raises:
            ValueError: Neither the class nor the config names an application.
            TypeError: The slug resolved to an application of another family.
            ModuleNotFoundError: FlashDreams, or the family's package, is not
                installed.
        """
        config = _read_config(config_path)
        slug = self.application or config.get("application")
        if not isinstance(slug, str) or not slug:
            raise ValueError(
                f"{type(self).__name__} names no FlashDreams application. Set `application` "
                "on the class, or `application:` in config.yml."
            )
        weights_root = str(get_weights_path())
        _point_caches_at(weights_root)

        fd_app = _registry()(slug)
        expected = _family_application_class(self.family)
        if not isinstance(fd_app, expected):
            raise TypeError(
                f"{slug!r} resolved to {type(fd_app).__name__}, not a {self.family} "
                f"application. {type(self).__name__} serves the {self.family} family; point "
                f"`runtime.import` at the reactor_runtime.flashdreams class for the family "
                f"{type(fd_app).__name__} belongs to."
            )
        type(self).fps = float(fd_app.session_desc().frames_per_second_for_step)
        self.configure(fd_app, config)

        load_kwargs: dict[str, Any] = {
            "application": slug,
            "weights_root": weights_root,
            "warmup_steps": int(config.get("warmup_steps", 0)),
        }
        if self.world_size > 1 or self.isolate:
            self.engine = DistributedRunner(
                self.model_class, world_size=self.world_size, load_kwargs=load_kwargs
            )
            self.engine.start()
        else:
            self.engine = self.model_class()
            self.engine.load(**load_kwargs)

    def configure(self, fd_app: Any, config: dict[str, Any]) -> None:
        """Read what the family needs off the adapter and the config, before the model loads.

        Runs once in ``load()``, after the slug resolved and before the model
        half is built. A family class overrides it to keep the adapter's hooks
        it uses in ``process_input()`` and to read its own config keys.

        Args:
            fd_app: The FlashDreams application the slug resolved to, not
                initialized: its ``defaults`` and ``session_desc()`` are usable.
            config: The parsed ``config.yml``, or an empty mapping.
        """

    def generate(self, step: Any) -> FlashDreamsResult:
        """One pipeline step. The model half does the work."""
        return self.engine.generate(step)

    async def process_output(self, outcome: StepOutcome) -> FlashDreamsOutput | None:
        """Turn the step's result or error into what the client receives.

        A :class:`RolloutExhausted` starts a new rollout from the same starting
        point and tells the client with ``rollout_restarted``. A
        :class:`PromptSwapUnsupported` starts a new rollout with the new prompt.
        Any other error is re-raised and ends the session. A result's frames
        go to ``main_video``, each tagged with its rollout id and index; the
        first frames of a new rollout flush what is still queued of the old one.

        A model whose rollout is one step long, such as a single-clip
        text-to-video model, must override this, or its clip loops.
        """
        if isinstance(outcome.error, RolloutExhausted):
            self.state._rollout_id += 1
            steps = outcome.error.args[0] if outcome.error.args else 0
            await self.send(RolloutRestarted(steps=int(steps)))
            return None
        if isinstance(outcome.error, PromptSwapUnsupported):
            self.state._rollout_id += 1
            return None
        if outcome.error is not None:
            # RolloutNotStarted means process_input() let a step through that it
            # should have refused, and anything from the pipeline is not repaired
            # by a reset. Either ends the session loudly.
            raise outcome.error
        result: FlashDreamsResult = outcome.result
        if result.rollout_id != self.state._applied_rollout_id:
            self.output.flush()
        self.state._applied_rollout_id = result.rollout_id
        tag: Metadata = {"rollout": result.rollout_id, "index": result.index}
        metadata: list[Metadata] = [tag] * len(result.frames)
        return FlashDreamsOutput(main_video=TrackPayload(result.frames, metadata=metadata))

    @event(name="reset", description="Start over from the same starting point.")
    def reset(self) -> None:
        """Ask for a new rollout. The next step starts it."""
        self.state._rollout_id += 1

    @session_ended
    def end(self) -> None:
        """Forget the rollout when the session ends, so the next one starts clean.

        A subclass extends this by overriding it and calling ``super().end()``.
        """
        self.engine.reset()


def _read_config(config_path: Path | None) -> dict[str, Any]:
    """Parse ``config.yml`` into a mapping; no file or an empty file is an empty mapping."""
    if config_path is None:
        return {}
    loaded = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must hold a mapping at the top level")
    return loaded


def _family_application_class(family: str) -> type:
    """Return the FlashDreams application class of *family*.

    Every family package is named after the family and exports one
    application class named after it: ``action2v.Action2VApplication``,
    ``t2v.T2VApplication``, ``cam2v.Cam2VApplication``, ``v2v.V2VApplication``.
    """
    try:
        module = importlib.import_module(family)
    except ModuleNotFoundError as exc:
        if exc.name == family:
            raise ModuleNotFoundError(
                f"The FlashDreams {family} family package is not installed. The workspace's "
                f"requirements.txt installs `flashdreams-{family}` from "
                f"https://github.com/NVIDIA/flashdreams with `#subdirectory=apps/{family}`."
            ) from exc
        raise
    return getattr(module, family.capitalize().replace("2v", "2V") + "Application")
