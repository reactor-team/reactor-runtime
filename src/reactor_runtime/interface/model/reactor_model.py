"""The previous module path of :class:`ReactorApp`.

The class lives in :mod:`reactor_runtime.interface.app.reactor_app`. This
module keeps ``from reactor_runtime.interface.model.reactor_model import
ReactorModel`` importable. The name resolves to :class:`ReactorApp`, with a
:class:`DeprecationWarning`, and is removed in the next major.
"""

from __future__ import annotations

from typing import Any

from reactor_runtime.interface.app.reactor_app import ReactorApp
from reactor_runtime.interface.internal.aliases import deprecated_alias

__all__: list[str] = []


def __getattr__(name: str) -> Any:
    if name == "ReactorModel":
        return deprecated_alias("ReactorModel", "ReactorApp", ReactorApp)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
