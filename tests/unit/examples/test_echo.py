"""Surface checks for the echo example.

Guards the client-facing contract of ``examples/echo`` — its commands, their
constraints, the media tracks, command/reply wiring, the rendered schema, and
that the manifest still resolves to the model class. The runtime's own suites
cover the loopback behaviour; these tests fail if a rename or a signature change
would break a client or the generated SDK.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from examples.echo.echo import Echo, EchoInput, EchoOutput, EffectChanged
from reactor_runtime import serve
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.runner.runner import import_model_class

_EFFECTS = ["none", "grayscale", "sepia", "edges", "invert", "blur", "pixelate"]
_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "echo"


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    """Re-seed only echo's surface after the per-test registry clear."""
    register_model(Echo)


def test_commands_are_exactly_the_declared_set() -> None:
    commands = ModelContract.of(Echo).commands
    assert set(commands) == {"set_effect", "set_intensity", "set_caption", "set_overlay_image"}


def test_set_effect_offers_every_effect() -> None:
    spec = ModelContract.of(Echo).commands["set_effect"]
    assert spec.command.__command_fields__["effect"].info.choices == _EFFECTS


def test_set_intensity_is_bounded_zero_to_one() -> None:
    info = ModelContract.of(Echo).commands["set_intensity"].command.__command_fields__["intensity"]
    assert info.info.ge == 0.0
    assert info.info.le == 1.0


def test_set_caption_bounds_its_length() -> None:
    info = ModelContract.of(Echo).commands["set_caption"].command.__command_fields__["caption"]
    assert info.info.max_length == 200


def test_set_overlay_image_bounds_its_strength() -> None:
    fields = ModelContract.of(Echo).commands["set_overlay_image"].command.__command_fields__
    assert fields["overlay_strength"].info.ge == 0.0
    assert fields["overlay_strength"].info.le == 1.0


def test_tracks_are_bidirectional_audio_and_video() -> None:
    tracks = ModelContract.of(Echo).tracks
    assert {name: (t.kind.value, t.direction.value) for name, t in tracks.items()} == {
        "webcam": ("video", "in"),
        "mic": ("audio", "in"),
        "main_video": ("video", "out"),
        "main_audio": ("audio", "out"),
    }


def test_effect_commands_reply_with_effect_changed() -> None:
    commands = ModelContract.of(Echo).commands
    assert commands["set_effect"].response is EffectChanged
    assert commands["set_intensity"].response is EffectChanged
    assert commands["set_caption"].response is None
    assert commands["set_overlay_image"].response is None


def test_schema_renders_the_full_surface() -> None:
    doc = ModelContract.of(Echo).render_schema().to_openapi()
    assert set(doc["paths"]) == {
        "/events/set_effect",
        "/events/set_intensity",
        "/events/set_caption",
        "/events/set_overlay_image",
    }
    assert "effect_changed" in doc["webhooks"]
    assert "EffectChanged" in doc["components"]["schemas"]
    tracks = {t["name"]: (t["kind"], t["direction"]) for t in doc["x-reactor"]["tracks"]}
    assert tracks == {
        "webcam": ("video", "in"),
        "mic": ("audio", "in"),
        "main_video": ("video", "out"),
        "main_audio": ("audio", "out"),
    }


def test_manifest_resolves_to_the_model_class(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = serve._load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert cfg.model_ref == "echo:Echo"
    # The example runs from its own directory, so its module is a top-level
    # `echo`, imported under a second name here — compare by qualname, not identity.
    monkeypatch.syspath_prepend(str(_EXAMPLE_DIR))
    assert import_model_class(cfg.model_ref).__qualname__ == "Echo"


def test_model_constructs_with_input_buffers_and_loads() -> None:
    model = Echo()
    model.load(None)
    assert isinstance(model.input, EchoInput)
    assert model.effect == "none"


def test_output_carries_both_tracks() -> None:
    assert set(EchoOutput.__tracks__) == {"main_video", "main_audio"}
