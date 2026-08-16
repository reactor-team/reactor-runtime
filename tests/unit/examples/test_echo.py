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
from typing import cast

import numpy as np
import pytest

from examples.echo.echo import Echo, EchoInput, EchoOutput, EffectChanged
from reactor_runtime import Output, TrackPayload
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config

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
    assert set(commands) == {
        "set_effect",
        "set_intensity",
        "set_burst",
        "set_caption",
        "set_overlay_image",
    }


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


def test_the_free_text_and_upload_fields_ask_for_moderation() -> None:
    commands = ModelContract.of(Echo).commands
    caption = commands["set_caption"].command.__command_fields__
    overlay = commands["set_overlay_image"].command.__command_fields__
    assert caption["caption"].info.moderate is True
    assert overlay["overlay_image"].info.moderate is True
    # A bounded knob carries no free text, so it asks for nothing.
    assert overlay["overlay_strength"].info.moderate is False


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
        "/events/set_burst",
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
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
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


async def test_session_start_resets_shared_state() -> None:
    """The session hook owns the shared state, so each session starts clean.

    A previous session's clients can leave the effect, caption, and overlay
    changed; ``on_session_start`` fires once at the top of the next session and
    restores the defaults, which no per-connection counter could guarantee
    across a server-side close.
    """
    model = Echo()
    model.load(None)
    model.effect = "sepia"
    model.intensity = 0.25
    model._caption = "left over"
    model._overlay = np.zeros((2, 2, 3), np.uint8)

    await model.on_session_start()

    assert model.effect == "none"
    assert model.intensity == 1.0
    assert model._caption == ""
    assert model._overlay is None


async def test_connect_leaves_shared_state_alone() -> None:
    """Connecting is per-client, so a later client never resets what a peer set."""
    model = Echo()
    model.load(None)
    await model.on_session_start()
    model.effect = "invert"

    await model.on_connect()

    assert model.effect == "invert"


def test_output_carries_both_tracks() -> None:
    assert set(EchoOutput.__tracks__) == {"main_video", "main_audio"}


def test_output_carries_the_metadata_a_frame_arrived_with() -> None:
    """A tagged inbound frame's metadata rides back out on what it produced."""
    output = EchoOutput(
        main_video=TrackPayload(np.zeros((2, 2, 3), np.uint8), metadata=b'{"seq":3}'),
        main_audio=np.zeros((1, 4), np.int16),
    )
    assert output.__metadata__["main_video"] == b'{"seq":3}'


def test_output_carries_no_metadata_for_an_untagged_frame() -> None:
    """A frame the client sent untagged goes back out untagged."""
    output = EchoOutput(
        main_video=np.zeros((2, 2, 3), np.uint8),
        main_audio=np.zeros((1, 4), np.int16),
    )
    assert "main_video" not in output.__metadata__


async def test_set_burst_records_the_batch_size() -> None:
    model = Echo()
    model.load(None)
    assert model.burst == 1  # a steady tick by default

    await model.set_burst(12)

    assert model.burst == 12


def test_set_burst_reaches_four_seconds_of_media() -> None:
    """The ceiling has to be past where a batching model would ever sit."""
    info = ModelContract.of(Echo).commands["set_burst"].command.__command_fields__["burst"]
    assert info.info.ge == 1
    assert info.info.le == 120  # 4s at the model's 30 fps


async def test_session_start_returns_the_burst_to_a_steady_tick() -> None:
    model = Echo()
    model.load(None)
    await model.set_burst(12)

    await model.on_session_start()

    assert model.burst == 1


def test_a_burst_pads_untagged_frames_with_an_empty_trailer() -> None:
    """A batch needs one metadata entry per frame, tagged or not."""
    out = EchoOutput(
        main_video=TrackPayload(
            np.zeros((3, 2, 2, 3), np.uint8),
            metadata=[b'{"seq":0}', b"", b'{"seq":2}'],
        ),
        main_audio=np.zeros((1, 1440), np.int16),
    )

    assert out.__metadata__["main_video"] == [b'{"seq":0}', b"", b'{"seq":2}']


def _gather(model: Echo, monkeypatch: pytest.MonkeyPatch) -> list[EchoOutput]:
    """Capture what a model emits instead of sending it downstream."""
    emitted: list[EchoOutput] = []

    async def _capture(output: Output, **_: object) -> None:
        emitted.append(cast(EchoOutput, output))

    monkeypatch.setattr(model, "emit", _capture)
    return emitted


async def test_a_resolution_change_ends_the_batch_instead_of_breaking_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WebRTC rescales inbound video mid-stream; a batch is one array."""
    model = Echo()
    model.load(None)
    emitted = _gather(model, monkeypatch)
    small = [np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8)]
    audio = [np.zeros((1, 480), np.int16) for _ in small]

    await model._emit_burst(small, audio, [None, None])

    assert len(emitted) == 1
    assert cast(np.ndarray, emitted[0].main_video).shape == (2, 2, 2, 3)
    assert small == []  # the burst is left empty for the next one


async def test_a_burst_of_one_emits_a_plain_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch of length one is a frame, not a batch."""
    model = Echo()
    model.load(None)
    emitted = _gather(model, monkeypatch)

    await model._emit_burst([np.zeros((2, 2, 3), np.uint8)], [np.zeros((1, 480), np.int16)], [None])

    assert cast(np.ndarray, emitted[0].main_video).shape == (2, 2, 3)


async def test_an_empty_burst_emits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Echo()
    model.load(None)
    emitted = _gather(model, monkeypatch)

    await model._emit_burst([], [], [])

    assert emitted == []
