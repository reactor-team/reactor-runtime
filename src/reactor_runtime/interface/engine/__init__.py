"""Serving an inference engine — :class:`EnginePipeline` and the layer under it.

The runtime half of the engine/runtime contract. An engine declares its inputs
and satisfies a four-method protocol; this package reads those declarations into
the serving surface the rest of the runtime already understands — commands,
input tracks, a schema — and drives the engine's rollout from the ordered window
of client input.

An author sees three things: :class:`EnginePipeline` to subclass,
:func:`override_input` to replace one generated event, and ``self.inputs`` to
push into.
"""

from reactor_runtime.interface.engine.application import application_for, is_engine
from reactor_runtime.interface.engine.engine_pipeline import (
    STEPPING_MODES,
    EnginePipeline,
    InitRequiredError,
    Stepping,
)
from reactor_runtime.interface.engine.overrides import override_input
from reactor_runtime.interface.engine.reflection import VIDEO_TRACK
from reactor_runtime.interface.engine.store import InputStore, MediaSpec

__all__ = [
    "STEPPING_MODES",
    "VIDEO_TRACK",
    "EnginePipeline",
    "InitRequiredError",
    "InputStore",
    "MediaSpec",
    "Stepping",
    "application_for",
    "is_engine",
    "override_input",
]
