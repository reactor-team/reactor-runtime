"""The client contract a director app reads from its ``generate()`` signature."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import numpy as np
import pytest

from reactor_runtime import InputField, Output, UploadedFile, Video
from reactor_runtime.apps.director import ReactorDirectorApp, Shot
from reactor_runtime.apps.director.contract import describe
from reactor_runtime.apps.director.shots import ShotEntry, ShotStatus
from reactor_runtime.interface.events.messages import MESSAGE_REGISTRY
from reactor_runtime.interface.model.contract import ContractError, ModelContract


class Frames(Output):
    main_video: Video


class Style(enum.Enum):
    SOFT = "soft"
    SHARP = "sharp"


class Reel(ReactorDirectorApp):
    """Shots of a lighthouse."""

    fps = 24

    def generate(
        self,
        prompt: str = InputField(moderate=True, max_length=200, description="What it shows."),
        seconds: float = InputField(default=2.0, ge=1.0, le=4.0),
        style: Style = Style.SOFT,
        starting_frame: UploadedFile | None = None,
        references: list[UploadedFile] | None = InputField(default=None, max_length=3),
        continue_from: Shot | None = None,
        ending_from: Shot | None = InputField(default=None, description="End on its frame."),
    ) -> Frames:
        return Frames(main_video=np.zeros((2, 2, 2, 3), dtype=np.uint8))


DIRECTOR_COMMANDS = {
    "cancel",
    "discard",
    "enqueue",
    "get_state",
    "move",
    "play",
    "reset",
    "set_autoplay",
    "set_build_paused",
    "set_hold_last_frame",
    "stop",
}


def test_every_director_app_serves_the_same_commands() -> None:
    assert set(ModelContract.of(Reel).commands) == DIRECTOR_COMMANDS


def test_enqueue_takes_every_generate_parameter_plus_metadata_and_position() -> None:
    fields = ModelContract.of(Reel).commands["enqueue"].command.__command_fields__
    assert list(fields) == [
        "prompt",
        "seconds",
        "style",
        "starting_frame",
        "references",
        "continue_from",
        "ending_from",
        "metadata",
        "position",
    ]


def test_enqueue_keeps_each_parameter_constraint() -> None:
    contract = ModelContract.of(Reel)
    with pytest.raises(ContractError, match="seconds"):
        contract.validate("enqueue", {"prompt": "x", "seconds": 9})
    with pytest.raises(ContractError, match="prompt"):
        contract.validate("enqueue", {"prompt": "x" * 201})
    with pytest.raises(ContractError, match="prompt"):
        contract.validate("enqueue", {})
    with pytest.raises(ContractError, match="position"):
        contract.validate("enqueue", {"prompt": "x", "position": -1})


def test_a_shot_parameter_takes_a_shot_id_on_the_wire() -> None:
    command = ModelContract.of(Reel).validate(
        "enqueue", {"prompt": "x", "continue_from": "3b1d", "ending_from": "3b1d"}
    )
    assert vars(command)["continue_from"] == "3b1d"
    with pytest.raises(ContractError, match="continue_from"):
        ModelContract.of(Reel).validate("enqueue", {"prompt": "x", "continue_from": 7})


def test_enqueue_replies_with_a_shot_and_get_state_with_the_state() -> None:
    commands = ModelContract.of(Reel).commands
    assert commands["enqueue"].response is not None
    assert commands["enqueue"].response.name == "shot_queued"
    assert commands["get_state"].response is not None
    assert commands["get_state"].response.name == "director_state"


def test_the_schema_types_the_inputs_a_shot_carries() -> None:
    spec = Reel.__director__
    assert spec is not None
    for message in dataclasses.astuple(spec.messages):
        MESSAGE_REGISTRY[message.name] = message
    document = ModelContract.of(Reel).render_schema().to_openapi()
    schemas = document["components"]["schemas"]
    shot = schemas["ShotQueued"]["properties"]["shot"]
    inputs = shot["properties"]["inputs"]["properties"]
    assert inputs["prompt"]["type"] == "string"
    assert inputs["starting_frame"]["anyOf"][0]["properties"].keys() == {"name", "mime_type"}
    assert inputs["continue_from"]["anyOf"][0]["type"] == "string"
    assert shot["properties"]["status"]["enum"] == [s.value for s in ShotStatus]
    assert {
        "DirectorState",
        "ShotQueued",
        "ShotGenerated",
        "ShotFailed",
        "ShotStarted",
        "ShotFinished",
        "ShotStopped",
    } <= set(schemas)
    state = schemas["DirectorState"]["properties"]
    assert state["generation"]["items"]["properties"]["inputs"] == shot["properties"]["inputs"]


def test_a_shot_description_carries_plain_inputs() -> None:
    upload = UploadedFile(name="a.png", mime_type="image/png", data=b"\x89PNG")
    entry = ShotEntry(
        id="s1",
        inputs={
            "prompt": "x",
            "style": Style.SHARP,
            "starting_frame": upload,
            "references": [upload],
        },
        sources={},
        continues=None,
        metadata="m",
        frames=48,
    )
    described = describe(entry, 24.0)
    assert described["inputs"] == {
        "prompt": "x",
        "style": "sharp",
        "starting_frame": {"name": "a.png", "mime_type": "image/png"},
        "references": [{"name": "a.png", "mime_type": "image/png"}],
    }
    assert described["seconds"] == 2.0
    assert described["status"] == "queued"


def test_a_class_without_generate_cannot_be_built() -> None:
    class Unfinished(ReactorDirectorApp):
        pass

    with pytest.raises(TypeError, match="generate"):
        Unfinished()


@pytest.mark.parametrize(
    ("signature", "match"),
    [
        ("def generate(self, prompt, /): ...", "named parameter"),
        ("def generate(self, *prompts): ...", "named parameter"),
        ("def generate(self, **inputs): ...", "named parameter"),
        ("def generate(self, metadata: str = ''): ...", "uses itself"),
        ("def generate(self, position: int = 0): ...", "uses itself"),
        ("def generate(self, client: str = ''): ...", "uses itself"),
        ("def generate(self, continue_from: Shot | None = 'x'): ...", "default to None"),
    ],
)
def test_a_generate_a_client_cannot_call_is_refused_when_the_class_is_defined(
    signature: str, match: str
) -> None:
    namespace: dict[str, Any] = {"Shot": Shot}
    exec(signature, namespace)
    with pytest.raises(TypeError, match=match):
        type("Bad", (ReactorDirectorApp,), {"generate": namespace["generate"]})


@pytest.mark.parametrize("name", ["max_queued", "max_ready", "history"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_a_limit_must_be_a_positive_integer(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=name):
        type("Bad", (Reel,), {name: value})


def test_plays_must_be_an_output_with_tracks() -> None:
    with pytest.raises(TypeError, match="plays"):
        type("Bad", (Reel,), {"plays": int})


def test_a_director_app_serves_stepped_sessions() -> None:
    assert Reel.supports_stepped()
