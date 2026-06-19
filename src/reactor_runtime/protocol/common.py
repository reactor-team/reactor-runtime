"""Version-agnostic adaptation primitives shared by the codecs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from reactor_wire.v1 import model_pb2


def struct_to_dict(value: Struct) -> dict[str, Any]:
    """Convert a protobuf Struct into a plain JSON-compatible dict."""
    return json_format.MessageToDict(value)


def dict_to_struct(value: Mapping[str, Any]) -> Struct:
    """Build a protobuf Struct from a plain mapping."""
    struct = Struct()
    struct.update(dict(value))
    return struct


def upload_reference_to_dict(ref: model_pb2.UploadReference) -> dict[str, Any]:
    """Render an UploadReference as the legacy upload-reference object."""
    return {
        "upload_id": ref.upload_id,
        "name": ref.name,
        "mime_type": ref.mime_type,
        "size": ref.size,
    }


def dict_to_upload_reference(value: Mapping[str, Any]) -> model_pb2.UploadReference:
    """Parse a legacy upload-reference object into an UploadReference."""
    return model_pb2.UploadReference(
        upload_id=str(value.get("upload_id", "")),
        name=str(value.get("name", "")),
        mime_type=str(value.get("mime_type", "")),
        size=int(value.get("size", 0)),
    )
