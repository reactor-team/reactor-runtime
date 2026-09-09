"""Reactor Runtime — a Python framework for building real-time, interactive video models.

The authoring surface is re-exported here so a model author imports from one
obvious place::

    from reactor_runtime import ReactorModel, Output, Video, event

The same names are available under :mod:`reactor_runtime.interface`; importing
from the top-level package is the preferred path.
"""

from importlib.metadata import version

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
    Input,
    InputBuffer,
    InputField,
    InputFrame,
    InputState,
    MessageField,
    Metadata,
    ModelMessage,
    Output,
    OutputStream,
    ReactorModel,
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
    "Input",
    "InputBuffer",
    "InputField",
    "InputFrame",
    "InputState",
    "MessageField",
    "Metadata",
    "ModelMessage",
    "Output",
    "OutputStream",
    "ReactorModel",
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
