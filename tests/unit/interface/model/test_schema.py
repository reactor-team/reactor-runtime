import enum
from collections.abc import Callable
from typing import Any

import pytest

from reactor_runtime import (
    InputField,
    ModelMessage,
    Output,
    ReactorModel,
    UploadedFile,
    Video,
    event,
)
from reactor_runtime.interface.model import ModelContract


class Mode(enum.StrEnum):
    TURBO = "turbo"
    QUALITY = "quality"


class Status(ModelMessage):
    """How the run is going."""

    progress: float


class Out(Output):
    main_video: Video


class DemoModel(ReactorModel):
    """A demo model."""

    @event(name="set_level", description="Set the level")
    async def set_level(self, level: int = InputField(default=1, ge=0, le=10)) -> None: ...

    @event(name="set_mode")
    async def set_mode(self, mode: Mode = Mode.TURBO) -> Status:
        return Status(progress=0.0)

    @event(name="attach")
    async def attach(self, file: UploadedFile) -> None: ...

    @event(name="attach_marked")
    async def attach_marked(self, file: UploadedFile = InputField(moderate=True)) -> None: ...

    @event(name="set_prompt")
    async def set_prompt(self, prompt: str = InputField(default="", moderate=True)) -> None: ...


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(DemoModel)


def schema() -> dict[str, Any]:
    return ModelContract.of(DemoModel).render_schema(version="v2.0.1").to_openapi()


def test_document_is_openapi_3_1_with_model_identity() -> None:
    doc = schema()
    assert doc["openapi"] == "3.1.0"
    assert doc["info"]["title"] == "demo_model"
    assert doc["info"]["version"] == "v2.0.1"
    assert doc["info"]["description"] == "A demo model."


def test_commands_render_as_paths() -> None:
    paths = schema()["paths"]
    assert set(paths) == {
        "/events/set_level",
        "/events/set_mode",
        "/events/attach",
        "/events/attach_marked",
        "/events/set_prompt",
    }
    op = paths["/events/set_level"]["post"]
    assert op["operationId"] == "set_level"
    assert op["summary"] == "Set the level"


def test_field_constraints_reach_the_schema() -> None:
    props = schema()["paths"]["/events/set_level"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    assert props["level"]["minimum"] == 0
    assert props["level"]["maximum"] == 10
    assert props["level"]["default"] == 1


def test_a_field_renders_an_unmarked_moderation_preference_by_default() -> None:
    props = schema()["paths"]["/events/set_level"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    assert props["level"]["x-reactor-moderate"] is False


def test_a_field_that_opts_in_renders_the_moderation_mark() -> None:
    props = schema()["paths"]["/events/set_prompt"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    assert props["prompt"]["x-reactor-moderate"] is True


def test_an_upload_field_carries_its_preference_beside_the_reference() -> None:
    body = schema()["paths"]["/events/attach"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert body["properties"]["file"] == {
        "$ref": "#/components/schemas/ReactorUploadReference",
        "x-reactor-moderate": False,
    }


def test_a_marked_upload_field_renders_the_mark_beside_the_reference() -> None:
    body = schema()["paths"]["/events/attach_marked"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert body["properties"]["file"] == {
        "$ref": "#/components/schemas/ReactorUploadReference",
        "x-reactor-moderate": True,
    }
    # An InputField carrying no default leaves the upload required.
    assert body["required"] == ["file"]


def test_a_field_with_a_default_is_not_required() -> None:
    body = schema()["paths"]["/events/set_level"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert "required" not in body  # the only field has a default


def test_enum_field_renders_as_a_typed_enum() -> None:
    props = schema()["paths"]["/events/set_mode"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    assert props["mode"]["type"] == "string"
    assert props["mode"]["enum"] == ["turbo", "quality"]
    assert props["mode"]["default"] == "turbo"


def test_upload_field_renders_as_a_reference() -> None:
    props = schema()["paths"]["/events/attach"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    assert props["file"]["$ref"] == "#/components/schemas/ReactorUploadReference"


def test_messages_render_as_webhooks_referencing_their_component() -> None:
    webhooks = schema()["webhooks"]
    assert set(webhooks) == {"status"}
    op = webhooks["status"]["post"]
    assert op["operationId"] == "status"
    assert op["summary"] == "How the run is going."
    # The webhook body references the shared component rather than inlining it.
    assert op["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Status"
    }


def test_each_message_is_a_component_keyed_by_its_class_name() -> None:
    components = schema()["components"]["schemas"]
    assert "Status" in components
    assert components["Status"]["properties"]["progress"]["type"] == "number"


def test_a_command_with_a_return_type_references_the_message_component() -> None:
    # set_mode is annotated `-> Status`.
    responses = schema()["paths"]["/events/set_mode"]["post"]["responses"]
    assert responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Status"
    }


def test_a_command_returning_nothing_renders_accepted() -> None:
    responses = schema()["paths"]["/events/set_level"]["post"]["responses"]
    assert responses == {"202": {"description": "Command accepted"}}


def test_tracks_ride_on_the_x_reactor_extension() -> None:
    tracks = schema()["x-reactor"]["tracks"]
    assert tracks == [{"name": "main_video", "kind": "video", "direction": "out"}]


def test_upload_reference_is_a_shared_component() -> None:
    components = schema()["components"]["schemas"]
    assert "ReactorUploadReference" in components
    assert components["ReactorUploadReference"]["format"] == "reactor-upload-reference"
