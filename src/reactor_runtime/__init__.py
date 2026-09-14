"""Reactor Runtime — a Python framework for building real-time, interactive video models.

The authoring surface is re-exported here so a model author imports from one
obvious place::

    from reactor_runtime import ReactorApp, Output, Video, event

The same names are available under :mod:`reactor_runtime.interface`; importing
from the top-level package is the preferred path.

Two names are deprecated aliases. ``ReactorModel`` is :class:`ReactorApp` and
``Input`` is :class:`MediaInput`; both import with a :class:`DeprecationWarning`
and are removed in the next major.
"""

from importlib.metadata import version
from typing import Any

from reactor_runtime.interface import (
    EVENT_REGISTRY,
    INPUT_REGISTRY,
    MESSAGE_REGISTRY,
    OUTPUT_REGISTRY,
    Audio,
    BufferClosed,
    ClientInfo,
    Command,
    CommandError,
    FieldInfo,
    Idle,
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
    ReactorPipeline,
    ReadMode,
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
from reactor_runtime.interface.internal.aliases import deprecated_alias
from reactor_runtime.log import get_logger
from reactor_runtime.paths import get_weights_path

__version__ = version("reactor-runtime")

__all__ = [
    "EVENT_REGISTRY",
    "INPUT_REGISTRY",
    "MESSAGE_REGISTRY",
    "OUTPUT_REGISTRY",
    "Audio",
    "BufferClosed",
    "ClientInfo",
    "Command",
    "CommandError",
    "FieldInfo",
    "Idle",
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
    "ReactorPipeline",
    "ReadMode",
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

_DEPRECATED = {
    "ReactorModel": ("ReactorApp", ReactorApp),
    "Input": ("MediaInput", MediaInput),
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED:
        new, target = _DEPRECATED[name]
        return deprecated_alias(name, new, target)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
