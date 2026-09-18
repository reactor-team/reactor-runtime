"""Reactor Runtime — a Python framework for building real-time, interactive video models.

The authoring surface is re-exported here so a model author imports from one
obvious place::

    from reactor_runtime import ReactorApp, Output, Video, event

The same names are available under :mod:`reactor_runtime.interface`; importing
from the top-level package is the preferred path.

Four names are deprecated and import with a :class:`DeprecationWarning`.
``ReactorModel`` is :class:`ReactorApp` and ``Input`` is :class:`MediaInput`,
renames that resolve to the new class. ``ReactorPipeline`` and ``Idle`` still
resolve to themselves, so a model on the generator pattern keeps working, but
the pattern is retired in favour of :class:`ReactorApp` and ``generate()``.
All four are removed in the next major.
"""

from importlib.metadata import version
from typing import Any

from reactor_runtime.interface import (
    EVENT_REGISTRY,
    INPUT_REGISTRY,
    MESSAGE_REGISTRY,
    OUTPUT_REGISTRY,
    ApplicationError,
    Audio,
    BufferClosed,
    ClientInfo,
    Command,
    CommandError,
    FieldInfo,
    InputBuffer,
    InputField,
    InputFrame,
    InputState,
    MediaInput,
    MessageField,
    Metadata,
    ModelMessage,
    Output,
    OutputStream,
    ReactorApp,
    ReadMode,
    StepOutcome,
    Track,
    TrackPayload,
    UploadedFile,
    Video,
    all_input_tracks,
    all_output_tracks,
    connected,
    disconnected,
    event,
    file_uploaded,
    session_ended,
    session_started,
)
from reactor_runtime.interface.internal.aliases import (
    IDLE_DEPRECATION,
    PIPELINE_DEPRECATION,
    deprecated_alias,
)

# Bound under private names so the public ones stay out of the module's
# namespace and resolve through __getattr__, which is where the warning lives.
from reactor_runtime.interface.pipeline import Idle as _Idle
from reactor_runtime.interface.pipeline import ReactorPipeline as _ReactorPipeline
from reactor_runtime.log import get_logger
from reactor_runtime.paths import get_weights_path

__version__ = version("reactor-runtime")

__all__ = [
    "EVENT_REGISTRY",
    "INPUT_REGISTRY",
    "MESSAGE_REGISTRY",
    "OUTPUT_REGISTRY",
    "ApplicationError",
    "Audio",
    "BufferClosed",
    "ClientInfo",
    "Command",
    "CommandError",
    "FieldInfo",
    "InputBuffer",
    "InputField",
    "InputFrame",
    "InputState",
    "MediaInput",
    "MessageField",
    "Metadata",
    "ModelMessage",
    "Output",
    "OutputStream",
    "ReactorApp",
    "ReadMode",
    "StepOutcome",
    "Track",
    "TrackPayload",
    "UploadedFile",
    "Video",
    "__version__",
    "all_input_tracks",
    "all_output_tracks",
    "connected",
    "disconnected",
    "event",
    "file_uploaded",
    "get_logger",
    "get_weights_path",
    "session_ended",
    "session_started",
]

_DEPRECATED: dict[str, tuple[str, object, str | None]] = {
    "ReactorModel": ("ReactorApp", ReactorApp, None),
    "Input": ("MediaInput", MediaInput, None),
    "ReactorPipeline": ("ReactorApp", _ReactorPipeline, PIPELINE_DEPRECATION),
    "Idle": ("ApplicationError", _Idle, IDLE_DEPRECATION),
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED:
        new, target, message = _DEPRECATED[name]
        return deprecated_alias(name, new, target, message)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
