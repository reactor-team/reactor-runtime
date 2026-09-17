"""The contract resolved from an application class and the schema it renders.

Declaring a :class:`reactor_runtime.ReactorApp` subclass assembles its
:class:`ModelContract` from one traversal of the class and renders the
:class:`ModelSchema` a client reads. The contract and schema types are the
integration surface the runtime consumes, not part of the curated authoring API.
"""

from typing import Any

from reactor_runtime.interface.internal.aliases import deprecated_alias
from reactor_runtime.interface.model.contract import (
    CommandSpec,
    ContractError,
    LifecycleHooks,
    ModelContract,
)
from reactor_runtime.interface.model.schema import (
    CommandSchema,
    MessageSchema,
    ModelSchema,
    TrackSchema,
)

__all__ = [
    "CommandSchema",
    "CommandSpec",
    "ContractError",
    "LifecycleHooks",
    "MessageSchema",
    "ModelContract",
    "ModelSchema",
    "TrackSchema",
]


def __getattr__(name: str) -> Any:
    if name == "ReactorModel":
        from reactor_runtime.interface.app.reactor_app import ReactorApp

        return deprecated_alias("ReactorModel", "ReactorApp", ReactorApp)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
