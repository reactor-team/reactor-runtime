import asyncio
import contextlib
from collections.abc import Callable

import numpy as np
import pytest
from fake_engine import (
    Camera,
    FakeCache,
    FakeEngine,
    FakeInit,
    FakeStepInput,
    MediaOnlyEngine,
    Move,
    StrictEngine,
    frame,
)

from reactor_runtime import EnginePipeline, event, override_input
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import ConnId, MediaChunk
from reactor_runtime.engine_contract import UserInput
from reactor_runtime.interface.engine.engine_pipeline import InitRequiredError
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope
from reactor_runtime.interface.model.contract import ModelContract


class Paint(EnginePipeline):
    """A model backed by the fake engine."""

    engine = FakeEngine

    emitted: list[MediaChunk]
    produced: asyncio.Event


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Paint)


@pytest.fixture
def paint() -> Paint:
    """A loaded model with its loop-bound state up and its output captured."""
    model = Paint()
    model.load(None)
    model._on_loop_ready()
    model.emitted = []
    model.produced = asyncio.Event()

    def collect(chunk: MediaChunk) -> None:
        model.emitted.append(chunk)
        model.produced.set()

    model.bind_output(broadcast=lambda message: None, addressed=lambda *args: None, media=collect)
    return model


def _engine(model: EnginePipeline) -> FakeEngine:
    assert isinstance(model._engine, FakeEngine)
    return model._engine


def _moves(window: list[UserInput]) -> list[Move]:
    return [item for item in window if isinstance(item, Move)]


async def _dispatch(model: EnginePipeline, name: str, **args: object) -> None:
    """Send a command the way the bridge does: validated, then dispatched."""
    command = ModelContract.of(type(model)).validate(name, args)
    await model._dispatch_command(CommandEnvelope(command, ConnId(1), None))


# -- the generated surface -----------------------------------------------------


def test_a_declared_event_becomes_a_command() -> None:
    assert "move" in ModelContract.of(Paint).commands


def test_a_generated_command_carries_the_declarations_constraints() -> None:
    info = ModelContract.of(Paint).commands["move"].command.__command_fields__["speed"].info

    assert (info.ge, info.le) == (0.0, 8.0)


def test_the_init_is_served_as_a_command() -> None:
    assert "init" in ModelContract.of(Paint).commands


def test_declared_media_becomes_an_input_track_not_a_command() -> None:
    contract = ModelContract.of(Paint)

    assert "camera" not in contract.commands
    assert contract.tracks["camera"].direction == "in"


def test_the_generated_surface_renders_into_the_schema() -> None:
    document = ModelContract.of(Paint).render_schema().to_openapi()

    assert "/events/move" in document["paths"]
    assert "/events/init" in document["paths"]


def test_a_built_in_step_command_exists_for_a_triggered_deployment() -> None:
    assert "step" in ModelContract.of(Paint).commands


def test_the_step_command_does_not_shadow_the_step_method() -> None:
    assert callable(Paint.step)


def test_an_engine_declaring_no_media_declares_no_track() -> None:
    class Bare(EnginePipeline):
        engine = StrictEngine

    assert Bare.__engine_tracks__ is None


# -- commands reach the window -------------------------------------------------


async def test_a_command_queues_the_input_it_declares(paint: Paint) -> None:
    await _dispatch(paint, "move", direction="right", speed=2.0)

    window = paint.inputs.drain(0)
    assert [(item.direction, item.speed) for item in _moves(window)] == [("right", 2.0)]


async def test_a_rejected_payload_never_reaches_the_window(paint: Paint) -> None:
    from reactor_runtime.interface.model.contract import ContractError

    with pytest.raises(ContractError):
        await _dispatch(paint, "move", direction="right", speed=99.0)

    assert paint.inputs.drain(0) == []


async def test_media_reaches_the_window_rather_than_a_track_buffer(paint: Paint) -> None:
    paint.push_media("camera", frame())
    paint.push_media("camera", frame())

    window = paint.inputs.drain(0)
    assert len(window) == 1
    assert isinstance(window[0], Camera)


# -- stepping ------------------------------------------------------------------


async def test_a_step_folds_the_window_and_advances_the_engine(paint: Paint) -> None:
    await _dispatch(paint, "move", direction="left")

    frames = await paint.step()

    assert frames is not None
    assert _moves(_engine(paint).windows[0])[0].direction == "left"
    assert _engine(paint).generated == [0]
    assert _engine(paint).finalized == [0]


async def test_the_step_index_advances_by_exactly_one(paint: Paint) -> None:
    for _ in range(3):
        await paint.step()

    assert _engine(paint).generated == [0, 1, 2]


async def test_a_mapping_that_returns_none_skips_the_step(paint: Paint) -> None:
    class Skipping(EnginePipeline):
        engine = MediaOnlyEngine

    model = Skipping()
    model.load(None)
    model._on_loop_ready()

    assert await model.step() is None
    assert _engine(model).generated == []


async def test_a_step_returns_its_chunk_rather_than_emitting_it(paint: Paint) -> None:
    chunk = await paint.step()

    assert chunk is not None
    assert paint.emitted == []


async def test_each_step_sees_only_what_arrived_since_the_last_one(paint: Paint) -> None:
    await _dispatch(paint, "move", direction="left")
    await paint.step()
    await paint.step()

    assert len(_engine(paint).windows[0]) == 1
    assert _engine(paint).windows[1] == []


# -- initialization ------------------------------------------------------------


async def test_a_rollout_opens_lazily_on_the_first_step(paint: Paint) -> None:
    assert paint._cache is None

    await paint.step()

    assert isinstance(paint._cache, FakeCache)


async def test_a_client_that_sends_nothing_gets_the_declared_defaults(paint: Paint) -> None:
    await paint.step()

    assert paint._cache.initialized_with == {"shade": 8}


async def test_a_leading_init_opens_the_rollout_the_client_asked_for(paint: Paint) -> None:
    await _dispatch(paint, "init", shade=200)

    await paint.step()

    assert paint._cache.initialized_with == {"shade": 200}


async def test_a_leading_init_is_consumed_rather_than_mapped(paint: Paint) -> None:
    await _dispatch(paint, "init", shade=200)
    await _dispatch(paint, "move", direction="left")

    await paint.step()

    window = _engine(paint).windows[0]
    assert not any(isinstance(item, FakeInit) for item in window)
    assert [type(item) for item in window] == [Move]


async def test_a_later_init_reaches_the_mapping_which_owns_the_restart(paint: Paint) -> None:
    await paint.step()
    await _dispatch(paint, "init", shade=99)

    assert await paint.step() is None
    assert paint._cache.restarts == 1
    assert paint._cache.shade == 99


async def test_a_later_init_restarts_the_step_index(paint: Paint) -> None:
    await paint.step()
    await paint.step()
    await _dispatch(paint, "init", shade=99)
    await paint.step()

    await paint.step()
    assert _engine(paint).generated == [0, 1, 0]


async def test_a_later_init_drops_what_was_scheduled_against_the_old_sequence(
    paint: Paint,
) -> None:
    await paint.step()
    paint.inputs.push(Move(direction="left"), at_step=4)
    await _dispatch(paint, "init", shade=99)
    await paint.step()

    for _ in range(5):
        await paint.step()
    assert not any(isinstance(item, Move) for window in _engine(paint).windows for item in window)


async def test_a_required_init_field_holds_the_rollout_back() -> None:
    class Strict(EnginePipeline):
        engine = StrictEngine

    model = Strict()
    model.load(None)
    model._on_loop_ready()

    with pytest.raises(InitRequiredError, match="prompt"):
        await model.step()


async def test_a_required_init_field_supplied_by_the_client_opens_the_rollout() -> None:
    class Strict(EnginePipeline):
        engine = StrictEngine

    model = Strict()
    model.load(None)
    model._on_loop_ready()
    contract = ModelContract.of(Strict)
    command = contract.validate("init", {"prompt": "a forest"})
    await model._dispatch_command(CommandEnvelope(command, ConnId(1), None))

    await model.step()

    assert model._cache.initialized_with == {"prompt": "a forest"}


# -- deferred injection --------------------------------------------------------


async def test_an_input_can_be_scheduled_for_a_later_step(paint: Paint) -> None:
    paint.inputs.push(Move(direction="right"), at_step=2)

    await paint.step()
    await paint.step()
    await paint.step()

    assert [type(item) for item in _engine(paint).windows[2]] == [Move]


# -- overrides -----------------------------------------------------------------


async def test_an_override_replaces_the_wire_payload() -> None:
    class Mirrored(EnginePipeline):
        engine = FakeEngine

        @override_input(Move)
        def move(self, direction: str) -> Move:
            return Move(direction="left" if direction == "right" else "right")

    fields = ModelContract.of(Mirrored).commands["move"].command.__command_fields__
    assert set(fields) == {"direction"}

    model = Mirrored()
    model.load(None)
    model._on_loop_ready()
    command = ModelContract.of(Mirrored).validate("move", {"direction": "right"})
    await model._dispatch_command(CommandEnvelope(command, ConnId(1), None))

    assert [item.direction for item in _moves(model.inputs.drain(0))] == ["left"]


async def test_an_override_returning_none_drops_the_input() -> None:
    class Filtered(EnginePipeline):
        engine = FakeEngine

        @override_input(Move)
        def move(self, direction: str) -> Move | None:
            return None

    model = Filtered()
    model.load(None)
    model._on_loop_ready()
    command = ModelContract.of(Filtered).validate("move", {"direction": "left"})
    await model._dispatch_command(CommandEnvelope(command, ConnId(1), None))

    assert model.inputs.drain(0) == []


async def test_an_added_event_feeds_the_same_window() -> None:
    class Dashing(EnginePipeline):
        engine = FakeEngine

        @event(name="dash")
        def dash(self, direction: str) -> None:
            self.inputs.push(Move(direction=direction, speed=8.0))

    model = Dashing()
    model.load(None)
    model._on_loop_ready()
    command = ModelContract.of(Dashing).validate("dash", {"direction": "right"})
    await model._dispatch_command(CommandEnvelope(command, ConnId(1), None))

    assert [item.speed for item in _moves(model.inputs.drain(0))] == [8.0]


def test_a_hand_written_event_wins_over_the_generated_one() -> None:
    class Custom(EnginePipeline):
        engine = FakeEngine

        @event(name="move", description="Move, this deployment's way.")
        def move(self, direction: str) -> None:
            self.inputs.push(Move(direction=direction))

    spec = ModelContract.of(Custom).commands["move"]
    assert spec.description == "Move, this deployment's way."


async def test_an_application_can_replace_the_mapping() -> None:
    class Reconditioned(EnginePipeline):
        engine = FakeEngine

        def map_inputs(self, autoregressive_index, cache, inputs):
            return FakeStepInput(shade=255)

    model = Reconditioned()
    model.load(None)
    model._on_loop_ready()

    chunk = await model.step()

    assert chunk is not None
    assert int(chunk.flat[0]) == 255
    assert _engine(model).windows == []


# -- the loop ------------------------------------------------------------------


async def _emit_at_least(model: Paint, count: int) -> None:
    """Run the loop until it has emitted *count* chunks, then stop it."""
    task = asyncio.create_task(model.run())
    try:
        async with asyncio.timeout(2.0):
            while len(model.emitted) < count:
                model.produced.clear()
                await model.produced.wait()
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _open_session(model: EnginePipeline) -> None:
    await model._dispatch_reactor_event(SessionStarted(session_id="s"))
    await model._dispatch_reactor_event(ClientConnected(conn_id=ConnId(1), total=1))


async def test_the_loop_emits_each_steps_frames(paint: Paint) -> None:
    await _open_session(paint)

    await _emit_at_least(paint, 3)

    assert isinstance(paint.emitted[0], MediaChunk)
    assert paint.emitted[0].bundle.get_track("main_video") is not None


async def test_the_loop_waits_for_a_session_and_an_audience(paint: Paint) -> None:
    task = asyncio.create_task(paint.run())
    await asyncio.sleep(0.05)
    assert paint.emitted == []

    await _open_session(paint)
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert paint.emitted


async def test_ending_the_session_discards_the_rollout(paint: Paint) -> None:
    await _open_session(paint)
    await _emit_at_least(paint, 2)
    await paint._dispatch_reactor_event(SessionEnded(session_id="s", reason=EndReason.STOPPED))
    await asyncio.sleep(0.02)

    assert paint._cache is None
    assert paint._index == 0


async def test_the_last_client_leaving_stops_the_loop(paint: Paint) -> None:
    await _open_session(paint)
    await _emit_at_least(paint, 2)
    await paint._dispatch_reactor_event(ClientDisconnected(conn_id=ConnId(1), total=0))
    await asyncio.sleep(0.02)
    settled = len(paint.emitted)
    await asyncio.sleep(0.05)

    assert len(paint.emitted) == settled


# -- stepping modes ------------------------------------------------------------


async def test_a_triggered_model_parks_its_loop(paint: Paint) -> None:
    paint.stepping = "triggered"
    await _open_session(paint)
    task = asyncio.create_task(paint.run())
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert paint.emitted == []


async def test_a_triggered_model_advances_one_step_per_command(paint: Paint) -> None:
    paint.stepping = "triggered"

    await _dispatch(paint, "step")
    await _dispatch(paint, "step")

    assert _engine(paint).generated == [0, 1]
    assert len(paint.emitted) == 2


async def test_the_step_command_is_inert_while_the_model_drives_itself(paint: Paint) -> None:
    await _dispatch(paint, "step")

    assert _engine(paint).generated == []


async def test_a_triggered_models_rollout_ends_with_its_session(paint: Paint) -> None:
    paint.stepping = "triggered"
    await _open_session(paint)
    await _dispatch(paint, "step")

    await paint._dispatch_reactor_event(SessionEnded(session_id="s", reason=EndReason.STOPPED))

    assert paint._cache is None


# -- emission ------------------------------------------------------------------


async def test_frames_are_emitted_on_the_track_the_runtime_declared(paint: Paint) -> None:
    await paint.emit_chunk(np.zeros((3, 2, 2), dtype=np.uint8))

    track = paint.emitted[0].bundle.get_track("main_video")
    assert track is not None
    assert track.data.shape == (2, 2, 3)


async def test_a_decoded_chunk_is_normalized_for_the_wire(paint: Paint) -> None:
    # What the fake engine returns is [T, C, H, W] uint8; a real decoder returns
    # floating point in its own range. Both leave as (N, H, W, 3) uint8.
    chunk = await paint.step()
    await paint.emit_chunk(chunk)

    track = paint.emitted[0].bundle.get_track("main_video")
    assert track is not None
    assert track.data.dtype == np.uint8
    assert track.data.shape[-1] == 3


async def test_a_pinned_fps_overrides_the_measured_pace() -> None:
    class Pinned(EnginePipeline):
        engine = FakeEngine
        fps = 12

    model = Pinned()
    model.load(None)
    model._on_loop_ready()
    chunks: list[MediaChunk] = []
    model.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=chunks.append
    )

    frames = await model.step()
    assert frames is not None
    await model.emit_chunk(frames, compute_time=1.0)

    assert chunks[0].fps == 12.0


# -- declaration errors --------------------------------------------------------


def test_binding_an_engine_is_the_whole_declaration() -> None:
    class Bare(EnginePipeline):
        engine = FakeEngine

    assert Bare().__engine_output__.__tracks__["main_video"].direction == "out"


def test_a_model_that_binds_no_engine_is_rejected() -> None:
    class NoEngine(EnginePipeline):
        pass

    with pytest.raises(TypeError, match="must bind an engine"):
        NoEngine()
