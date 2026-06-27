"""Reactor Runtime — a Python framework for building real-time, interactive video models.

The authoring surface is re-exported here so a model author imports from one
obvious place::

    from reactor_runtime import ReactorModel, Output, Video, event

The same names are available under :mod:`reactor_runtime.interface`; importing
from the top-level package is the preferred path.
"""

from importlib.metadata import version

from reactor_runtime.interface import (
    Audio,
    BufferClosed,
    ClientInfo,
    Command,
    FieldInfo,
    Idle,
    Input,
    InputBuffer,
    InputField,
    InputFrame,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    ReactorModel,
    ReactorPipeline,
    ReadMode,
    Track,
    UploadedFile,
    Video,
    connected,
    disconnected,
    event,
    file_uploaded,
    session_ended,
    session_started,
)
from reactor_runtime.paths import get_weights_path

__version__ = version("reactor-runtime")

__all__ = [
    "Audio",
    "BufferClosed",
    "ClientInfo",
    "Command",
    "FieldInfo",
    "Idle",
    "Input",
    "InputBuffer",
    "InputField",
    "InputFrame",
    "InputState",
    "MessageField",
    "ModelMessage",
    "Output",
    "ReactorModel",
    "ReactorPipeline",
    "ReadMode",
    "Track",
    "UploadedFile",
    "Video",
    "__version__",
    "connected",
    "disconnected",
    "event",
    "file_uploaded",
    "get_weights_path",
    "session_ended",
    "session_started",
]
