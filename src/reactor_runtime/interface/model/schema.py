"""The published model schema — :class:`ModelSchema`.

The structured description of what a model exposes: its media tracks, the
commands a client can send, and the messages it sends back. The contract builds
one from a single class traversal, and :meth:`ModelSchema.to_openapi` renders it
as the OpenAPI 3.1 document downstream code generators consume — commands as
``paths``, messages as ``webhooks``, tracks under the ``x-reactor`` extension.
"""

from __future__ import annotations

import copy
import enum
from dataclasses import dataclass, field
from typing import Any

from reactor_runtime.core.fields import NO_DEFAULT, FieldInfo
from reactor_runtime.core.model import CommandField
from reactor_runtime.core.values import TrackInfo
from reactor_runtime.interface.events.messages import MessageFieldSpec

_UPLOAD_REFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "format": "reactor-upload-reference",
    "description": "Reference to a file uploaded via the Reactor upload protocol.",
    "properties": {
        "upload_id": {"type": "string", "format": "uuid"},
        "name": {"type": "string"},
        "mime_type": {"type": "string"},
        "size": {"type": "integer"},
    },
    "required": ["upload_id", "name", "mime_type", "size"],
}


@dataclass
class TrackSchema:
    """A single track's metadata in the schema.

    Attributes:
        kind: ``"video"`` or ``"audio"``.
        direction: ``"in"`` (client to model) or ``"out"`` (model to client).
        rate: Native rate in units per second, or ``0`` when not applicable.
    """

    kind: str
    direction: str
    rate: float = 0.0


@dataclass
class CommandSchema:
    """A single command's description and per-field schema."""

    description: str = ""
    schema: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class MessageSchema:
    """A single outbound message's description and per-field schema."""

    description: str = ""
    schema: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ModelSchema:
    """A versioned description of a model's tracks, commands, and messages.

    Attributes:
        version: Release tag, carrying a leading ``v`` (e.g. ``"v1.2.3"``).
        name: Model identifier, lowercase, doubling as the schema title.
        description: Human-readable model description.
        tracks: Track name to its metadata.
        commands: Command name to its schema.
        messages: Message name to its schema.
    """

    version: str = "v0.0.0"
    name: str = ""
    description: str = ""
    tracks: dict[str, TrackSchema] = field(default_factory=dict)
    commands: dict[str, CommandSchema] = field(default_factory=dict)
    messages: dict[str, MessageSchema] = field(default_factory=dict)

    def to_openapi(self) -> dict[str, Any]:
        """Render the schema as an OpenAPI 3.1 document.

        Commands become ``post`` operations under ``paths`` (a required field is
        one with no default); messages become ``webhooks``; tracks ride on the
        ``x-reactor`` extension. The upload reference is the one shared component.
        """
        paths: dict[str, Any] = {}
        for name, command in self.commands.items():
            paths[f"/events/{name}"] = {
                "post": {
                    "operationId": name,
                    "summary": command.description or f"Trigger {name}",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": _body(command.schema)}},
                    },
                    "responses": {"202": {"description": "Command accepted"}},
                }
            }

        webhooks: dict[str, Any] = {}
        for name, message in self.messages.items():
            operation: dict[str, Any] = {
                "operationId": name,
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": _body(message.schema)}},
                },
            }
            if message.description:
                operation["summary"] = message.description
            webhooks[name] = {"post": operation}

        doc: dict[str, Any] = {
            "openapi": "3.1.0",
            "info": {
                "title": self.name,
                "version": self.version,
                "description": self.description,
            },
            "x-reactor": {
                "tracks": [
                    {"name": name, "kind": track.kind, "direction": track.direction}
                    for name, track in self.tracks.items()
                ]
            },
        }
        if paths:
            doc["paths"] = paths
        if webhooks:
            doc["webhooks"] = webhooks
        doc["components"] = {
            "schemas": {"ReactorUploadReference": copy.deepcopy(_UPLOAD_REFERENCE_SCHEMA)}
        }
        return doc


def _body(properties: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Wrap per-field schemas into a request-body object schema."""
    required = [name for name, schema in properties.items() if "default" not in schema]
    body: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        body["required"] = required
    return body


def command_field_schema(command_field: CommandField) -> dict[str, Any]:
    """Render one command field as a JSON Schema fragment with its constraints."""
    schema = command_field.spec.to_json_schema()
    info = command_field.info
    if info.default is not NO_DEFAULT:
        schema["default"] = _coerce_default(info.default)
    _merge_constraints(schema, info)
    return schema


def message_field_schema(message_field: MessageFieldSpec) -> dict[str, Any]:
    """Render one message field as a JSON Schema fragment."""
    schema = message_field.spec.to_json_schema()
    if message_field.description:
        schema["description"] = message_field.description
    if message_field.default is not NO_DEFAULT:
        schema["default"] = _coerce_default(message_field.default)
    return schema


def track_schema(track: TrackInfo) -> TrackSchema:
    """Render a resolved track as its schema entry."""
    return TrackSchema(kind=track.kind.value, direction=track.direction.value, rate=track.rate)


def _merge_constraints(schema: dict[str, Any], info: FieldInfo) -> None:
    """Merge an :class:`FieldInfo`'s constraints into a JSON Schema fragment."""
    if info.description:
        schema["description"] = info.description
    if info.ge is not None:
        schema["minimum"] = info.ge
    if info.le is not None:
        schema["maximum"] = info.le
    if info.min_length is not None:
        schema["minLength"] = info.min_length
    if info.max_length is not None:
        schema["maxLength"] = info.max_length
    if info.choices is not None:
        schema["enum"] = info.choices
    if not info.moderate:
        schema["x-reactor-moderate"] = False


def _coerce_default(value: Any) -> Any:
    """Make a default JSON-serialisable, unwrapping enum members recursively."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, list):
        return [_coerce_default(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_default(item) for item in value]
    if isinstance(value, dict):
        return {key: _coerce_default(item) for key, item in value.items()}
    return value
