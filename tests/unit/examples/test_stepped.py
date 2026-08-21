"""Surface checks for the stepped example.

Guards the client-facing contract of ``examples/stepped`` — the commands its
state fields generate, the media tracks, the rendered schema, and that the
manifest still resolves to the model class. A thin behaviour smoke drives one
step through the mapping, the model, and the output shaping; the driver's own
suite covers the loop.

The workspace is imported by its own directory, which ``pyproject.toml`` puts on
the path, because that is how the runtime imports a model: ``stepped.py`` reaches
its sibling as ``kaleidoscope``, not through a package path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from kaleidoscope import Kaleidoscope

from reactor_runtime import NotReady, StepStats
from reactor_runtime.core.values import InputFrame
from reactor_runtime.interface.internal.input_buffer import InputBuffer
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config
from stepped import (
    BLOCK,
    CAMERA_FPS,
    StepDone,
    Stepped,
    SteppedInput,
    SteppedOutput,
    SteppedState,
)

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "stepped"


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None,
    register: Callable[..., None],
    register_model: Callable[[type], None],
) -> None:
    """Re-seed only this example's surface after the per-test registry clear."""
    register_model(Stepped)
    # StepDone is broadcast rather than returned by a handler, so the model walk
    # does not reach it.
    register(StepDone)


def _buffer(count: int, size: int = 4) -> InputBuffer:
    buffer = InputBuffer()
    for index in range(count):
        buffer.push(
            InputFrame(data=np.full((size, size, 3), 10 * index, dtype=np.uint8), pts=float(index))
        )
    return buffer


def _input(count: int) -> SteppedInput:
    """Build the input holder the runtime would bind, with one live buffer."""
    return SteppedInput(webcam=_buffer(count))


# -- the client-facing contract -----------------------------------------------


def test_commands_are_generated_from_the_state_fields() -> None:
    commands = ModelContract.of(Stepped).commands
    assert set(commands) == {"set_mirror", "set_drift", "restart"}


def test_the_mirror_command_carries_its_choices() -> None:
    command = ModelContract.of(Stepped).commands["set_mirror"].command
    assert command.__command_fields__["mirror"].info.choices == [
        "none",
        "vertical",
        "horizontal",
        "both",
    ]


def test_the_tracks_are_one_inbound_and_one_outbound() -> None:
    tracks = ModelContract.of(Stepped).tracks
    assert set(tracks) == {"webcam", "main_video"}


def test_the_step_message_is_published_in_the_schema() -> None:
    schema = ModelContract.of(Stepped).render_schema()
    assert "step_done" in schema.messages


def test_the_manifest_resolves_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert cfg.model_ref == "stepped:Stepped"
    monkeypatch.syspath_prepend(str(_EXAMPLE_DIR))
    assert import_model_class(cfg.model_ref).__qualname__ == "Stepped"


# -- the step -----------------------------------------------------------------


def test_the_mapping_declines_until_a_full_block_arrives() -> None:
    app = Stepped()
    with pytest.raises(NotReady, match="waiting for"):
        app.map_step(SteppedState(), _input(BLOCK - 1))


def test_the_mapping_resamples_a_block_that_straddles_a_resolution_change() -> None:
    # WebRTC rescales an inbound track mid-stream, so a block can hold two
    # sizes. They are resampled to the newest one rather than failing to stack.
    buffer = _buffer(BLOCK, size=4)
    for index in range(BLOCK - 1):
        buffer.push(InputFrame(data=np.zeros((8, 8, 3), dtype=np.uint8), pts=float(index)))
    app = Stepped()
    inputs = app.map_step(SteppedState(), SteppedInput(webcam=buffer))
    assert cast(np.ndarray, inputs["driving"]).shape == (BLOCK, 8, 8, 3)


def test_the_mapping_returns_the_model_arguments() -> None:
    app = Stepped()
    inputs = app.map_step(SteppedState(mirror="both", drift=0.25), _input(BLOCK))
    assert inputs["driving"].shape == (BLOCK, 4, 4, 3)
    assert inputs["mirror"] == "both"
    assert inputs["drift"] == 0.25


def test_load_builds_the_model_and_hands_it_the_config() -> None:
    app = Stepped()
    app.load(None)
    model = app.model
    assert isinstance(model, Kaleidoscope)
    assert model.step == 0


def test_one_step_produces_a_message_and_a_tagged_block() -> None:
    app = Stepped()
    app.load(None)
    inputs = app.map_step(SteppedState(), _input(BLOCK))
    produced = app.model.generate(**inputs)
    message, output = app.to_output(
        frames=cast(np.ndarray, produced["frames"]),
        hue=cast(float, produced["hue"]),
        chunk=cast(int, produced["chunk"]),
        stats=StepStats(step=3, compute_time=0.02),
    )

    assert isinstance(message, StepDone)
    # Two counts from two owners: the model's place in its rollout, and the
    # runtime's tally of what it has driven.
    assert (message.chunk, message.step) == (0, 3)
    assert isinstance(output, SteppedOutput)
    assert cast(np.ndarray, output.main_video).shape == (BLOCK, 4, 4, 3)
    # One metadata entry per frame, which is what a batch requires.
    assert len(output.__metadata__["main_video"]) == BLOCK
    assert output.fps == CAMERA_FPS


def test_a_restart_sets_the_model_back_to_the_start_of_its_rollout() -> None:
    app = Stepped()
    app.load(None)
    inputs = app.map_step(SteppedState(), _input(BLOCK))
    app.model.generate(**inputs)
    app.model.generate(**inputs)
    assert app.model.generate(**inputs)["chunk"] == 2

    app.model.restart()
    assert app.model.generate(**inputs)["chunk"] == 0


def test_the_model_restarts_its_own_rollout_when_conditioning_changes() -> None:
    model = Kaleidoscope()
    model.load(None)
    driving = np.zeros((BLOCK, 4, 4, 3), dtype=np.uint8)
    model.generate(driving=driving, mirror="none", drift=0.1)
    model.generate(driving=driving, mirror="none", drift=0.1)
    assert model.step == 2
    # A new mirror is new conditioning, so the rollout starts over.
    model.generate(driving=driving, mirror="both", drift=0.1)
    assert model.step == 1
