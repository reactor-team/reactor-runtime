"""Surface and behaviour checks for the starter example.

The model renders on the CPU, so it is tested as it is: loaded against a
temporary config and weights directory, and driven through ``generate`` with a
state instance the way the default ``process_input`` hands it one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from reactor_runtime.core.model import EndReason, SessionEnded, SessionStarted
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config
from starter import Starter, StarterOutput, StarterState

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "starter"


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Starter)


def _state(spin_speed: float = 1.0, paused: bool = False, static_interval: int = 8) -> StarterState:
    return StarterState(spin_speed=spin_speed, paused=paused, static_interval=static_interval)


def _loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, logo: bool = True) -> Starter:
    """A model loaded against a small frame and a weights directory of our own."""
    config = tmp_path / "config.yaml"
    config.write_text("width: 64\nheight: 40\nlogo: logo.png\n")
    weights = tmp_path / "weights"
    weights.mkdir()
    if logo:
        Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(weights / "logo.png")
    monkeypatch.setenv("REACTOR_WEIGHTS_PATH", str(weights))
    model = Starter()
    model.load(config)
    return model


# -- the client contract ------------------------------------------------------


def test_commands_are_exactly_the_state_setters() -> None:
    assert set(ModelContract.of(Starter).commands) == {
        "set_spin_speed",
        "set_paused",
        "set_static_interval",
    }


def test_the_setters_carry_the_field_bounds() -> None:
    commands = ModelContract.of(Starter).commands
    speed = commands["set_spin_speed"].command.__command_fields__["spin_speed"].info
    interval = commands["set_static_interval"].command.__command_fields__["static_interval"].info
    assert (speed.ge, speed.le) == (0.05, 5.0)
    assert (interval.ge, interval.le) == (1, 300)


def test_the_one_output_track_is_main_video() -> None:
    assert list(StarterOutput.__tracks__) == ["main_video"]


def test_the_step_is_generate_alone() -> None:
    # The example's point is the smallest model: the other two hooks are the
    # runtime's defaults, so the state reaches generate() as it is.
    assert "process_input" not in vars(Starter)
    assert "process_output" not in vars(Starter)


def test_manifest_resolves_to_the_app_class() -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert import_model_class(cfg.model_ref) is Starter
    assert cfg.config_path == _EXAMPLE_DIR / "config.yaml"


def test_the_workspace_ships_the_logo_the_config_names() -> None:
    assert (_EXAMPLE_DIR / "weights" / "logo.png").is_file()


# -- loading ------------------------------------------------------------------


def test_load_reads_the_frame_size_and_the_logo_from_the_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _loaded(tmp_path, monkeypatch)
    assert model.background.shape == (40, 64, 3)
    assert model.logo.size[0] <= 32  # fits half the frame width


def test_load_draws_a_placeholder_when_the_weights_hold_no_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _loaded(tmp_path, monkeypatch, logo=False)
    assert model.logo.mode == "RGBA"


def test_load_rejects_a_frame_size_that_is_not_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("width: 0\nheight: 40\n")
    monkeypatch.setenv("REACTOR_WEIGHTS_PATH", str(tmp_path))
    with pytest.raises(ValueError, match="positive"):
        Starter().load(config)


# -- generating ---------------------------------------------------------------


def test_generate_returns_a_frame_on_main_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _loaded(tmp_path, monkeypatch)
    output = model.generate(_state())
    assert isinstance(output, StarterOutput)
    frame = np.asarray(output.main_video)
    assert frame.shape == (40, 64, 3)
    assert frame.dtype == np.uint8


def test_the_angle_advances_by_the_speed_over_the_frame_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _loaded(tmp_path, monkeypatch)
    model.generate(_state(spin_speed=1.5))
    assert model.angle == pytest.approx(2 * np.pi * 1.5 / 30)


def test_paused_holds_the_angle_while_the_static_keeps_rolling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _loaded(tmp_path, monkeypatch)
    model.generate(_state(paused=True, static_interval=1))
    before = model.static
    model.generate(_state(paused=True, static_interval=1))
    assert model.angle == 0.0
    assert before is not model.static


def test_the_static_holds_for_the_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _loaded(tmp_path, monkeypatch)
    model.generate(_state(static_interval=3))
    first = model.static
    model.generate(_state(static_interval=3))
    model.generate(_state(static_interval=3))
    assert model.static is first
    model.generate(_state(static_interval=3))
    assert model.static is not first


def test_reset_returns_to_the_first_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _loaded(tmp_path, monkeypatch)
    for _ in range(3):
        model.generate(_state())
    assert model.index == 3
    model.reset()
    assert (model.angle, model.index, model.static) == (0.0, 0, None)


async def test_a_session_end_starts_the_animation_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _loaded(tmp_path, monkeypatch)
    model._on_loop_ready()
    model.bind_output(
        broadcast=lambda *args: None, addressed=lambda *args: None, media=lambda c: None
    )
    await model._dispatch_reactor_event(SessionStarted("s"))
    model.generate(model.state)
    assert model.index == 1
    await model._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert (model.angle, model.index, model.static) == (0.0, 0, None)
