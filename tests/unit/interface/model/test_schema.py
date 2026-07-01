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

    output: Out

    @event(name="set_level", description="Set the level")
    async def set_level(self, level: int = InputField(default=1, ge=0, le=10)) -> None: ...

    @event(name="set_mode")
    async def set_mode(self, mode: Mode = Mode.TURBO) -> Status:
        return Status(progress=0.0)

    @event(name="attach")
    async def attach(self, file: UploadedFile) -> None: ...


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
    assert set(paths) == {"/events/set_level", "/events/set_mode", "/events/attach"}
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
    assert props["file"] == {"$ref": "#/components/schemas/ReactorUploadReference"}


def test_messages_render_as_webhooks() -> None:
    webhooks = schema()["webhooks"]
    assert set(webhooks) == {"status"}
    op = webhooks["status"]["post"]
    assert op["operationId"] == "status"
    assert op["summary"] == "How the run is going."
    props = op["requestBody"]["content"]["application/json"]["schema"]["properties"]
    assert props["progress"]["type"] == "number"


def test_tracks_ride_on_the_x_reactor_extension() -> None:
    tracks = schema()["x-reactor"]["tracks"]
    assert tracks == [{"name": "main_video", "kind": "video", "direction": "out"}]


def test_upload_reference_is_a_shared_component() -> None:
    components = schema()["components"]["schemas"]
    assert "ReactorUploadReference" in components
    assert components["ReactorUploadReference"]["format"] == "reactor-upload-reference"
