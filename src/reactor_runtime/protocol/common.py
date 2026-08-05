"""Version-agnostic adaptation primitives shared by the codecs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from reactor_wire.v1 import model_pb2


def struct_to_dict(value: Struct) -> dict[str, Any]:
    """Convert a protobuf Struct into a plain JSON-compatible dict."""
    return json_format.MessageToDict(value)


def dict_to_struct(value: Mapping[Any, Any]) -> Struct:
    """Build a protobuf Struct from a plain mapping.

    Struct fields are keyed by string, so mapping keys are coerced to the
    string ``json.dumps`` renders for them: ints and floats by their ``repr``
    (non-finite floats as ``Infinity`` / ``-Infinity`` / ``NaN``), ``True`` /
    ``False`` as ``true`` / ``false``, and ``None`` as ``null``. Coercion
    recurses through nested mappings, including mappings inside sequences.
    When two keys coerce to the same string, the entry iterated last wins.

    Args:
        value: The mapping to encode. Values must be JSON-representable.

    Returns:
        The equivalent Struct, with every field keyed by string.

    Raises:
        TypeError: If a key is not a str, int, float, bool, or None. The
            error names the offending key.
    """
    struct = Struct()
    struct.update({_coerce_key(key): _coerce_nested(item) for key, item in value.items()})
    return struct


def _coerce_key(key: Any) -> str:
    """Render a mapping key as the string ``json.dumps`` renders it."""
    if isinstance(key, str):
        return key
    if key is None or isinstance(key, bool | int | float):
        return json.dumps(key)
    raise TypeError(
        f"Struct keys must be str, int, float, bool or None, not {type(key).__name__}: {key!r}"
    )


def _coerce_nested(value: Any) -> Any:
    """Return *value* with every nested mapping key coerced to its string form."""
    if isinstance(value, Mapping):
        return {_coerce_key(key): _coerce_nested(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_coerce_nested(item) for item in value]
    return value


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
