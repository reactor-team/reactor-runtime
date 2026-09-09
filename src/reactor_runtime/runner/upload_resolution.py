"""Resolve the upload references a command's contract declares.

A client refers to an uploaded file by its ``upload_id`` and the runtime swaps
the reference for the bytes before the model sees the command. A single
top-level file travels beside the arguments, keyed by parameter name; a file
nested in a container has no such slot and travels inline, as a
``{"upload_id": ...}`` mapping inside the argument itself. Both forms are
resolved here.

The walk follows the command's declared :class:`~reactor_runtime.core.TypeSpec`
rather than the shape of the arguments: only a value the contract types as an
upload is fetched, so a mapping field of the model's own that happens to carry
an ``upload_id`` key is left exactly as the client sent it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from reactor_runtime.core import UploadedFile
from reactor_runtime.core.typespec import (
    DataclassSpec,
    DictSpec,
    ListSpec,
    OptionalSpec,
    TypeSpec,
    UploadSpec,
)

Fetch = Callable[[str], Awaitable[UploadedFile]]
"""Turn an ``upload_id`` into the uploaded file, or raise when it cannot."""


def declares_upload(spec: TypeSpec) -> bool:
    """Return whether a value of *spec* can carry an upload reference.

    True for an upload itself and for any container the contract supports
    around one — optional, list, dict, or dataclass field — however deep they
    nest. Every other type is False, so a caller can skip the walk for the
    fields that never need it.
    """
    if isinstance(spec, UploadSpec):
        return True
    if isinstance(spec, OptionalSpec):
        return declares_upload(spec.inner)
    if isinstance(spec, ListSpec):
        return declares_upload(spec.item)
    if isinstance(spec, DictSpec):
        return declares_upload(spec.value)
    if isinstance(spec, DataclassSpec):
        return any(declares_upload(field) for field in spec.fields.values())
    return False


async def resolve_uploads(spec: TypeSpec, value: Any, fetch: Fetch) -> Any:
    """Return *value* with every upload reference *spec* declares fetched.

    A reference is a mapping with a string ``upload_id`` in a position the spec
    types as an upload. An :class:`~reactor_runtime.core.UploadedFile` already
    in that position passes through, as does any value the spec does not type
    as an upload, so contract validation still sees whatever the client sent
    where it sent something else. The entries of a container are fetched
    together and returned under their original positions and keys, so the
    container as a whole waits no longer than its slowest entry.

    Args:
        spec: The declared type of the field holding *value*.
        value: The field's raw argument, as decoded from the wire.
        fetch: Resolves one ``upload_id`` to its file.

    Returns:
        The value with each declared reference replaced by its file.

    Raises:
        Exception: Whatever *fetch* raises for a reference it cannot resolve.
    """
    if isinstance(spec, OptionalSpec):
        if value is None:
            return None
        return await resolve_uploads(spec.inner, value, fetch)
    if isinstance(spec, ListSpec):
        if not isinstance(value, list) or not declares_upload(spec.item):
            return value
        return list(
            await asyncio.gather(*(resolve_uploads(spec.item, element, fetch) for element in value))
        )
    if isinstance(spec, DictSpec):
        if not isinstance(value, Mapping) or not declares_upload(spec.value):
            return value
        return await _resolve_entries(value, dict.fromkeys(value, spec.value), fetch)
    if isinstance(spec, DataclassSpec):
        if not isinstance(value, Mapping):
            return value
        # Fields the dataclass does not declare are left as sent; the contract
        # decides what to do with them.
        specs = {name: field for name, field in spec.fields.items() if name in value}
        if not any(declares_upload(field) for field in specs.values()):
            return value
        return await _resolve_entries(value, specs, fetch)
    if isinstance(spec, UploadSpec):
        if isinstance(value, Mapping) and isinstance(value.get("upload_id"), str):
            return await fetch(value["upload_id"])
        return value
    return value


async def _resolve_entries(
    value: Mapping[Any, Any], specs: Mapping[Any, TypeSpec], fetch: Fetch
) -> dict[Any, Any]:
    """Resolve the entries of *value* named in *specs* together, keeping every key in place."""
    resolved = dict(value)
    keys = list(specs)
    files = await asyncio.gather(*(resolve_uploads(specs[key], value[key], fetch) for key in keys))
    for key, file in zip(keys, files, strict=True):
        resolved[key] = file
    return resolved
