import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime import (
    OUTPUT_REGISTRY,
    InputField,
    InputState,
    Output,
    ReactorModel,
    SteppedModel,
    StepStats,
    Video,
    event,
)
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import ConnId
from reactor_runtime.interface.internal.reactor_core import ReactorCore
from reactor_runtime.interface.model.contract import ModelContract


class Frame(Output):
    main_video: Video


class State(InputState):
    prompt: str = InputField(default="", description="What to render.")
    guidance: float = InputField(default=1.0, ge=0.0, le=10.0)
    _scratch: Any = None


class Gen:
    """A model that satisfies the protocol and records how it was loaded."""

    def __init__(self) -> None:
        self.loaded_with: Path | None | str = "never"

    def load(self, config: Path | None) -> None:
        self.loaded_with = config

    def generate(self, **inputs: Any) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)


class Stepped(ReactorModel):
    """A model the runtime drives."""

    state: State
    model: Gen

    def load(self, config_path: Path | None) -> None:
        self.model = Gen()
        self.model.load(config_path)


@pytest.fixture(autouse=True)
def _seed_registries(isolate_interface_registries: None, register: Callable[..., None]) -> None:
    register(Frame)


def _ready(app: ReactorModel) -> None:
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None,
        addressed=lambda conn, message, request: None,
        media=lambda chunk: None,
    )


# -- state on a plain ReactorModel --------------------------------------------


def test_public_state_fields_become_commands() -> None:
    commands = ModelContract.of(Stepped).commands
    assert "set_prompt" in commands
    assert "set_guidance" in commands


def test_private_state_fields_are_not_exposed() -> None:
    assert "set__scratch" not in ModelContract.of(Stepped).commands


def test_a_generated_setter_carries_the_field_constraints() -> None:
    spec = ModelContract.of(Stepped).commands["set_guidance"]
    info = spec.command.__command_fields__["guidance"].info
    assert (info.ge, info.le) == (0.0, 10.0)


def test_a_hand_written_event_shadows_the_generated_setter() -> None:
    class Custom(ReactorModel):
        state: State
        model: Gen

        @event(name="set_prompt", description="hand written")
        def set_prompt(self, prompt: str = InputField(default="")) -> None:
            self.state.prompt = prompt

    assert ModelContract.of(Custom).commands["set_prompt"].description == "hand written"
    assert "set_guidance" in ModelContract.of(Custom).commands


def test_a_model_without_state_needs_no_state_annotation() -> None:
    class Stateless(ReactorModel):
        model: Gen

    app = Stateless()
    assert app.state is None
    assert "set_prompt" not in ModelContract.of(Stateless).commands


async def test_state_is_built_at_session_start_and_cleared_at_session_end() -> None:
    app = Stepped()
    _ready(app)
    assert app.state is None
    await app._dispatch_reactor_event(SessionStarted("s"))
    assert isinstance(app.state, State)
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert app.state is None


async def test_state_survives_a_client_leaving_and_rejoining_mid_session() -> None:
    app = Stepped()
    _ready(app)
    await app._dispatch_reactor_event(SessionStarted("s"))
    await app._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    app.state.prompt = "a forest"
    await app._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
    await app._dispatch_reactor_event(ClientConnected(ConnId(1002), 1))
    assert app.state.prompt == "a forest"


# -- binding the model --------------------------------------------------------


def test_a_class_level_default_is_built_for_the_author() -> None:
    class Declared(ReactorModel):
        model = Gen

    assert isinstance(Declared().model, Gen)


def test_the_default_load_loads_a_class_level_default() -> None:
    class Declared(ReactorModel):
        model = Gen

    app = Declared()
    app.load(Path("config.yml"))
    model = app.model
    assert isinstance(model, Gen)
    assert model.loaded_with == Path("config.yml")


def test_load_may_build_the_model_itself() -> None:
    app = Stepped()
    app.load(None)
    model = app.model
    assert isinstance(model, Gen)
    assert model.loaded_with is None


async def test_a_model_with_nothing_to_drive_says_what_is_missing() -> None:
    class Neither(ReactorModel):
        state: State

    app = Neither()
    _ready(app)
    with pytest.raises(TypeError, match=r"no model to drive"):
        await app.run()


def test_a_generator_satisfies_the_protocol_structurally() -> None:
    assert isinstance(Gen(), SteppedModel)
    assert not isinstance(object(), SteppedModel)


# -- the default mapping ------------------------------------------------------


def test_the_default_mapping_passes_every_public_state_field() -> None:
    app = Stepped()
    assert app.map_step(State(prompt="x", guidance=2.0), None) == {
        "prompt": "x",
        "guidance": 2.0,
    }


def test_the_default_mapping_of_a_stateless_model_is_empty() -> None:
    app = Stepped()
    assert app.map_step(None, None) == {}


def test_the_default_output_places_one_product_on_one_track() -> None:
    app = Stepped()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert app.to_output(frame).main_video is frame


def test_the_default_output_maps_a_mapping_onto_named_tracks() -> None:
    app = Stepped()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert app.to_output(main_video=frame).main_video is frame


def test_the_default_output_needs_an_output_class() -> None:
    class Messenger(ReactorModel):
        model: Gen

    # A message-only model declares no tracks at all, so it has to say what it
    # publishes. The registry is restored by the isolation fixture.
    OUTPUT_REGISTRY.clear()
    with pytest.raises(TypeError, match="declares no Output class"):
        Messenger().to_output(None)


def test_on_step_is_a_no_op_by_default() -> None:
    assert Stepped().on_step(StepStats(step=0, compute_time=0.1)) is None


# -- the surface an author reads ----------------------------------------------


def test_the_public_surface_is_exactly_this() -> None:
    # Frozen on purpose: machinery that creeps onto the class an author reads
    # fails here rather than in review. The default loop lives on StepDriver.
    assert sorted(name for name in dir(ReactorModel) if not name.startswith("_")) == [
        "bind_failure",
        "bind_output",
        "buffer_size",
        "emit",
        "fps",
        "load",
        "map_step",
        "on_state_changed",
        "on_step",
        "post_reactor_event",
        "push_media",
        "run",
        "send",
        "start_thread",
        "stop",
        "submit_command",
        "to_output",
    ]


def test_the_step_layer_adds_only_its_overridables() -> None:
    added = {
        name
        for name in dir(ReactorModel)
        if not name.startswith("_") and not hasattr(ReactorCore, name)
    }
    assert added == {"map_step", "on_state_changed", "on_step", "to_output"}


async def test_owning_the_loop_leaves_no_driver_behind() -> None:
    class OwnsLoop(ReactorModel):
        state: State

        def __init__(self) -> None:
            super().__init__()
            self.turns = 0

        async def run(self) -> None:
            while True:
                await self.connected.wait()
                while self.connected.is_set():
                    self.turns += 1
                    await asyncio.sleep(0)

    app = OwnsLoop()
    _ready(app)
    app.connected.set()
    task = asyncio.create_task(app.run())
    await asyncio.sleep(0.02)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert app.turns > 0
    # No driver was created, so nothing of the default loop is on the instance.
    assert not hasattr(app, "advance")
    assert "_step" not in vars(app)
