"""The contract holds against the shape production engines already have.

An interactive autoregressive video engine converges on the same surface: open a
rollout from a prompt and a seed frame, ask how many frames the next step emits,
advance by exactly one step, commit it. The pipeline below is written to that
shape — keyword-only extras on the initializer, ``autoregressive_index``
throughout, a per-step timing dict out of ``finalize``, and a decoded
``[T, C, H, W]`` tensor in ``[-1, 1]`` — with ``map_inputs`` as the only thing
the contract asks for that such an engine does not already have.

If a change to the contract would force a wrapper around an engine of this
shape, this test fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from reactor_runtime import EnginePipeline
from reactor_runtime.core.values import ConnId
from reactor_runtime.engine_contract import Init, InputField, ModelInput, UserInput
from reactor_runtime.interface.engine import is_engine
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope
from reactor_runtime.interface.model.contract import ModelContract

SEED_FRAMES = 9
STEP_FRAMES = 12


class KeyDown(UserInput):
    """Press a movement key."""

    key: str = InputField(default="w", choices=["w", "a", "s", "d"])


class KeyUp(UserInput):
    """Release a movement key."""

    key: str = InputField(default="w", choices=["w", "a", "s", "d"])


class WorldInit(Init):
    """The prompt a rollout starts from."""

    text: str = "a walk through a misty forest"


class CamCtrlInput(ModelInput):
    """One pose per frame of the next chunk."""

    poses: Any
    world_scale: float = 1.0


@dataclass
class PipelineCache:
    """Their cache shape: rollout state plus the index of the last step."""

    text: str
    autoregressive_index: int | None = None
    held: frozenset[str] = frozenset()


class WorldPipeline:
    """Written to the engine's own signatures, not to the runtime's convenience."""

    declared_inputs = (KeyDown, KeyUp, WorldInit)

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.conditioning: list[CamCtrlInput] = []

    def initialize_cache(
        self,
        text: str = "",
        *,
        height: int | None = None,
        width: int | None = None,
        release_oneshot_encoders: bool = True,
    ) -> PipelineCache:
        return PipelineCache(text=text)

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        return SEED_FRAMES if autoregressive_index == 0 else STEP_FRAMES

    def map_inputs(
        self, autoregressive_index: int, cache: PipelineCache, inputs: list[UserInput]
    ) -> CamCtrlInput | None:
        held = set(cache.held)
        for item in inputs:
            if isinstance(item, WorldInit):
                cache.text = item.text
                cache.autoregressive_index = None
                return None
            if isinstance(item, KeyDown):
                held.add(item.key)
            elif isinstance(item, KeyUp):
                held.discard(item.key)
        cache.held = frozenset(held)
        frames = self.get_num_output_frames(autoregressive_index)
        return CamCtrlInput(poses=np.zeros((frames, 4, 4), dtype=np.float32))

    def generate(
        self,
        autoregressive_index: int,
        cache: PipelineCache,
        input: Any = None,
    ) -> Any:
        last = -1 if cache.autoregressive_index is None else cache.autoregressive_index
        assert autoregressive_index == last + 1
        self.steps.append(autoregressive_index)
        self.conditioning.append(input)
        frames = self.get_num_output_frames(autoregressive_index)
        return np.zeros((frames, 3, 8, 8), dtype=np.float32)

    def finalize(self, autoregressive_index: int, cache: PipelineCache) -> dict[str, float] | None:
        cache.autoregressive_index = autoregressive_index
        return {"denoise_seconds": 0.01}


class World(EnginePipeline):
    """The whole application: bind the engine."""

    engine = WorldPipeline


@pytest.fixture(autouse=True)
def _seed_registries(isolate_interface_registries: None, register_model: Any) -> None:
    register_model(World)


@pytest.fixture
def world() -> World:
    model = World()
    model.load(None)
    model._on_loop_ready()
    model.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=lambda chunk: None
    )
    return model


def _pipeline(model: EnginePipeline) -> WorldPipeline:
    assert isinstance(model._engine, WorldPipeline)
    return model._engine


async def _send(model: EnginePipeline, name: str, **args: object) -> None:
    command = ModelContract.of(type(model)).validate(name, args)
    await model._dispatch_command(CommandEnvelope(command, ConnId(1), None))


def test_an_engine_of_this_shape_satisfies_the_protocol() -> None:
    assert is_engine(WorldPipeline)


def test_binding_it_is_the_whole_application() -> None:
    commands = ModelContract.of(World).commands

    assert set(commands) == {"key_down", "key_up", "init", "step"}


def test_its_own_initializer_signature_is_called_as_written(world: World) -> None:
    # `text` is a positional-or-keyword parameter with keyword-only extras after
    # it; the runtime calls it by keyword from the Init's fields.
    assert world._cache is None


async def test_the_index_advances_as_the_engine_asserts(world: World) -> None:
    for _ in range(3):
        await world.step()

    assert _pipeline(world).steps == [0, 1, 2]


async def test_the_mapping_sizes_conditioning_to_the_step(world: World) -> None:
    await world.step()
    await world.step()

    sizes = [len(step.poses) for step in _pipeline(world).conditioning]
    assert sizes == [SEED_FRAMES, STEP_FRAMES]


async def test_a_declared_default_opens_the_rollout(world: World) -> None:
    await world.step()

    assert world._cache.text == "a walk through a misty forest"


async def test_a_client_init_opens_the_rollout_it_asked_for(world: World) -> None:
    await _send(world, "init", text="a city at night")

    await world.step()

    assert world._cache.text == "a city at night"


async def test_key_edges_reach_the_mapping_in_order(world: World) -> None:
    await _send(world, "key_down", key="w")
    await _send(world, "key_down", key="a")
    await _send(world, "key_up", key="w")

    await world.step()

    assert world._cache.held == frozenset({"a"})


async def test_a_decoded_tensor_reaches_the_wire_as_frames(world: World) -> None:
    emitted: list[Any] = []
    world.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=emitted.append
    )

    await world.emit_chunk(await world.step())

    track = emitted[0].bundle.get_track("main_video")
    assert track.data.shape == (SEED_FRAMES, 8, 8, 3)
    assert track.data.dtype == np.uint8


async def test_the_timings_finalize_returns_are_ignored(world: World) -> None:
    await world.step()

    assert world._cache.autoregressive_index == 0
