"""Resolve the upload references a command's contract declares.

A client refers to an uploaded file by its ``upload_id`` and the runtime swaps
the reference for the bytes before the model sees the command. A single
top-level file travels beside the arguments, keyed by parameter name; a list of
files has no such slot and travels inline, as ``[{"upload_id": ...}, ...]``
inside the argument itself. Both forms are resolved here.

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
from reactor_runtime.core.typespec import ListSpec, OptionalSpec, TypeSpec, UploadSpec

Fetch = Callable[[str], Awaitable[UploadedFile]]
"""Turn an ``upload_id`` into the uploaded file, or raise when it cannot."""


def declares_upload(spec: TypeSpec) -> bool:
    """Return whether a value of *spec* can carry an upload reference.

    True for an upload itself and for an optional or list wrapper around one,
    however deep the wrappers nest. Every other type is False, so a caller can
    skip the walk for the fields that never need it.
    """
    if isinstance(spec, UploadSpec):
        return True
    if isinstance(spec, OptionalSpec):
        return declares_upload(spec.inner)
    if isinstance(spec, ListSpec):
        return declares_upload(spec.item)
    return False


async def resolve_uploads(spec: TypeSpec, value: Any, fetch: Fetch) -> Any:
    """Return *value* with every upload reference *spec* declares fetched.

    A reference is a mapping with a string ``upload_id`` in a position the spec
    types as an upload. An :class:`~reactor_runtime.core.UploadedFile` already
    in that position passes through, as does any value the spec does not type
    as an upload, so contract validation still sees whatever the client sent
    where it sent something else. The elements of a list are fetched together
    and returned in the client's order, so the list as a whole waits no longer
    than its slowest entry.

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
    if isinstance(spec, UploadSpec):
        if isinstance(value, Mapping) and isinstance(value.get("upload_id"), str):
            return await fetch(value["upload_id"])
        return value
    return value
