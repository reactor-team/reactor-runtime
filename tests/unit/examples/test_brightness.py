"""Surface checks for the brightness example.

Guards the client-facing contract of ``examples/brightness`` — its commands and
constraints (including the 4K resolution and the text overlay), the media
tracks, command/reply wiring, the rendered schema, and that the manifest still
resolves to the model class. A thin behaviour smoke confirms the generator
honours resolution and the caption; the pipeline's own suites cover the run
loop.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from examples.brightness.brightness import (
    Brightness,
    BrightnessOutput,
    BrightnessSet,
    BrightnessSnapshot,
    BrightnessState,
    ImageSet,
)
from reactor_runtime import Idle
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config

_RESOLUTIONS = ["480p", "720p", "1080p", "2160p"]
_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "brightness"


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    """Re-seed only brightness's surface after the per-test registry clear."""
    register_model(Brightness)


def test_commands_are_exactly_the_declared_set() -> None:
    commands = ModelContract.of(Brightness).commands
    assert set(commands) == {
        "set_brightness",
        "set_paused",
        "set_resolution",
        "set_text",
        "set_image",
        "get_state",
    }


def test_set_brightness_is_bounded_zero_to_two() -> None:
    fields = ModelContract.of(Brightness).commands["set_brightness"].command.__command_fields__
    assert fields["brightness"].info.ge == 0.0
    assert fields["brightness"].info.le == 2.0


def test_set_resolution_offers_up_to_4k() -> None:
    fields = ModelContract.of(Brightness).commands["set_resolution"].command.__command_fields__
    assert fields["resolution"].info.choices == _RESOLUTIONS


def test_set_text_bounds_its_length() -> None:
    fields = ModelContract.of(Brightness).commands["set_text"].command.__command_fields__
    assert fields["text"].info.max_length == 200


def test_the_free_text_and_upload_fields_ask_for_moderation() -> None:
    commands = ModelContract.of(Brightness).commands
    text = commands["set_text"].command.__command_fields__
    image = commands["set_image"].command.__command_fields__
    assert text["text"].info.moderate is True
    assert image["image"].info.moderate is True
    # A bounded knob carries no free text, so it asks for nothing.
    brightness = commands["set_brightness"].command.__command_fields__
    assert brightness["brightness"].info.moderate is False


def test_tracks_are_video_and_audio_out() -> None:
    tracks = ModelContract.of(Brightness).tracks
    assert {name: (t.kind.value, t.direction.value) for name, t in tracks.items()} == {
        "main_video": ("video", "out"),
        "main_audio": ("audio", "out"),
    }


def test_typed_replies_are_wired() -> None:
    commands = ModelContract.of(Brightness).commands
    assert commands["set_brightness"].response is BrightnessSet
    assert commands["set_image"].response is ImageSet
    assert commands["get_state"].response is BrightnessSnapshot
    assert commands["set_paused"].response is None
    assert commands["set_resolution"].response is None
    assert commands["set_text"].response is None


def test_schema_renders_the_full_surface() -> None:
    doc = ModelContract.of(Brightness).render_schema().to_openapi()
    assert set(doc["paths"]) == {
        "/events/set_brightness",
        "/events/set_paused",
        "/events/set_resolution",
        "/events/set_text",
        "/events/set_image",
        "/events/get_state",
    }
    schemas = doc["components"]["schemas"]
    for message in ("BrightnessSet", "ImageSet", "BrightnessSnapshot"):
        assert message in schemas
    tracks = {t["name"]: (t["kind"], t["direction"]) for t in doc["x-reactor"]["tracks"]}
    assert tracks == {"main_video": ("video", "out"), "main_audio": ("audio", "out")}


def test_manifest_resolves_to_the_model_class(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert cfg.model_ref == "brightness:Brightness"
    monkeypatch.syspath_prepend(str(_EXAMPLE_DIR))
    assert import_model_class(cfg.model_ref).__qualname__ == "Brightness"


def _first_frame(model: Brightness) -> tuple[np.ndarray, np.ndarray]:
    """Drive one turn of the generator and return its (video, audio) arrays.

    The track attributes are typed as the ``Video`` / ``Audio`` markers, so cast
    back to the numpy arrays the generator actually yields.
    """
    output = next(model.inference())
    assert isinstance(output, BrightnessOutput)
    return cast(np.ndarray, output.main_video), cast(np.ndarray, output.main_audio)


def test_default_output_shape_and_audio() -> None:
    model = Brightness()
    model.state = BrightnessState()
    video, audio = _first_frame(model)
    assert video.shape == (480, 640, 3)
    assert video.dtype == np.uint8
    assert audio.shape == (1, 1600)
    assert audio.dtype == np.int16


def test_resolution_drives_the_frame_size() -> None:
    model = Brightness()
    model.state = BrightnessState()
    model.state.resolution = "2160p"
    video, _ = _first_frame(model)
    assert video.shape == (2160, 3840, 3)


def test_text_overlay_changes_the_frame() -> None:
    plain = Brightness()
    plain.state = BrightnessState()
    without, _ = _first_frame(plain)

    captioned = Brightness()
    captioned.state = BrightnessState()
    captioned.state.text = "hello world"
    with_text, _ = _first_frame(captioned)

    assert without.shape == with_text.shape
    assert not np.array_equal(without, with_text)


def test_paused_yields_idle() -> None:
    model = Brightness()
    model.state = BrightnessState()
    model.state.paused = True
    assert next(model.inference()) is Idle
