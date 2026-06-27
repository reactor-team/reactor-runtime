"""Model-authoring surface — the public API a model author writes against.

One obvious place to import from: the tracks a model exchanges, the ``@event``
handlers and lifecycle hooks that shape its command set, the typed messages it
sends back, and the field metadata that constrains them. These build on the
spine's neutral vocabulary and carry no engine — declaring them is enough to
resolve a model's contract.

Everything re-exported here is also available directly on the top-level
``reactor_runtime`` package, which is the preferred import path.
"""

from reactor_runtime.core import (
    Command,
    FieldInfo,
    InputField,
    InputFrame,
    UploadedFile,
)
from reactor_runtime.interface.client import ClientInfo
from reactor_runtime.interface.events import (
    MessageField,
    ModelMessage,
    connected,
    disconnected,
    event,
    file_uploaded,
    session_ended,
    session_started,
)
from reactor_runtime.interface.internal.input_buffer import (
    BufferClosed,
    InputBuffer,
    ReadMode,
)
from reactor_runtime.interface.model import ReactorModel
from reactor_runtime.interface.pipeline import Idle, InputState, ReactorPipeline
from reactor_runtime.interface.tracks import Audio, Input, Output, Track, Video

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
    "connected",
    "disconnected",
    "event",
    "file_uploaded",
    "session_ended",
    "session_started",
]
