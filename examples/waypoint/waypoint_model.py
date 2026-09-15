"""Waypoint 1.5, the model half.

Plain Python around the upstream ``world_engine`` package. Nothing in this file
knows about clients, tracks, commands, or the runtime's loop, and nothing here
imports ``reactor_runtime``. :class:`WaypointModel` has three methods: ``load``
puts the weights on the GPU, ``generate`` runs one step, ``reset`` returns to
the default state. The application half (``waypoint.py``) constructs it, calls
``load`` once, and calls ``generate`` once per step.

The step input and the step result are the contract between the two halves.
Both are plain dataclasses the two files agree on; the runtime never reads
their fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class WaypointStepInput:
    """What one step needs.

    Attributes:
        buttons: Pressed Owl-Control VK keycodes.
        mouse: Mouse velocity for this step, as ``(dx, dy)``.
        scroll_wheel: Scroll direction for this step: ``-1`` down, ``0`` none, ``1`` up.
        seed: The frame the world starts from, uint8 ``(H, W, 3)``, or ``None``
            when the application has no new seed to offer.
        seed_id: Identifies the seed. A step whose ``seed_id`` differs from the
            one the model last applied starts a new world from ``seed``; the
            application sends ``seed`` on that step and leaves it ``None`` on
            every other.
    """

    buttons: frozenset[int]
    mouse: tuple[float, float]
    scroll_wheel: int
    seed: np.ndarray | None
    seed_id: int


@dataclass(frozen=True)
class WaypointStepResult:
    """What one step produced.

    Attributes:
        frames: The frames this step generated, uint8 ``(4, H, W, 3)``, on the CPU.
        index: The model's own count of this step within the current world,
            starting at 0 after a seed is applied.
        seed_id: The id of the seed the world was started from. The application
            reads it to know which seed the model holds.
    """

    frames: np.ndarray
    index: int
    seed_id: int


class NotSeeded(Exception):  # noqa: N818 (the model's own error, named for the state)
    """The model holds no world to step and the step input carries no seed to start one."""


class WaypointModel:
    """Waypoint 1.5 behind ``load`` / ``generate`` / ``reset``.

    Holds the engine, the id of the seed it applied, and its step index. The
    engine's KV cache is the world; ``reset`` clears it and forgets the seed,
    so the next ``generate`` applies whatever seed its input carries.
    """

    def __init__(self) -> None:
        self.engine: Any = None
        self.seed_id: int | None = None
        self.index = 0

    def load(self, config_path: Path | None) -> None:
        """Construct the engine on the GPU and warm up its compiled kernels.

        Reads ``model_uri``, ``quant``, ``device``, and ``warmup_steps`` from the
        YAML at *config_path*. The heavy imports happen here, so the module
        imports on a machine without a GPU.

        Args:
            config_path: The file ``runtime.config`` in ``reactor.yaml`` names,
                or ``None`` for the defaults.
        """
        _, world_engine_cls, _ = _backend()

        config: dict[str, Any] = {}
        if config_path is not None:
            config = yaml.safe_load(config_path.read_text()) or {}
        self.engine = world_engine_cls(
            config.get("model_uri", "Overworld/Waypoint-1.5-1B"),
            quant=config.get("quant"),
            device=config.get("device", "cuda"),
        )
        self.reset()
        self._warmup(int(config.get("warmup_steps", 0)))

    def generate(self, step: WaypointStepInput) -> WaypointStepResult:
        """Run one step of the world.

        Applies ``step.seed`` first when ``step.seed_id`` is not the seed this
        model last applied: the cache is cleared and the seed frame becomes the
        world's first frame. Then generates four frames from the controls. The
        result reports the seed the world holds, so the application knows when
        to send the next one.

        Args:
            step: The controls and the seed for this step.

        Returns:
            The four frames and the step's index within the current world.

        Raises:
            NotSeeded: A new world is asked for and the step carries no seed.
            RuntimeError: ``load`` has not run.
        """
        if self.engine is None:
            raise RuntimeError("load() has not run")
        torch, _, ctrl_input = _backend()

        if step.seed_id != self.seed_id:
            if step.seed is None:
                raise NotSeeded("no seed frame to start a world from")
            self.engine.reset()
            seed_x4 = torch.from_numpy(np.repeat(step.seed[None], 4, axis=0))
            self.engine.append_frame(seed_x4)
            self.seed_id = step.seed_id
            self.index = 0

        ctrl = ctrl_input(
            button=set(step.buttons), mouse=step.mouse, scroll_wheel=step.scroll_wheel
        )
        with torch.no_grad():
            frames = self.engine.gen_frame(ctrl=ctrl).cpu().numpy()
        index = self.index
        self.index += 1
        return WaypointStepResult(frames=frames, index=index, seed_id=step.seed_id)

    def reset(self) -> None:
        """Return to the default state: no world, no seed, index at zero."""
        if self.engine is not None:
            self.engine.reset()
        self.seed_id = None
        self.index = 0

    def _warmup(self, steps: int) -> None:
        """Run *steps* idle steps so the compiled kernels are built before a client waits."""
        if steps <= 0:
            return
        torch, _, ctrl_input = _backend()

        with torch.no_grad():
            for _ in range(steps):
                self.engine.gen_frame(ctrl=ctrl_input())
        self.engine.reset()


def _backend() -> tuple[Any, Any, Any]:
    """Import torch and the engine on first use.

    Both are GPU-side dependencies installed in the model image, not on a
    machine that only imports this module to read the contract or run tests.
    """
    import torch  # type: ignore[ty:unresolved-import]  # installed in the image only
    from world_engine import CtrlInput, WorldEngine  # type: ignore[ty:unresolved-import]  # same

    return torch, WorldEngine, CtrlInput
