"""Model-authoring surface — commands, messages, tracks, and handler decorators.

What a model author reaches for to declare a model's client-facing contract: the
command set (via ``@event`` handlers), typed outbound messages, media tracks, and
the lifecycle hooks. These build on the spine's neutral vocabulary and carry no
engine — declaring them is enough to resolve the contract.
"""

from reactor_runtime.core import Command, FieldInfo, InputField, UploadedFile
from reactor_runtime.model.decorators import (
    EventHandler,
    connected,
    disconnected,
    event,
    file_uploaded,
    make_command,
    session_ended,
    session_started,
)
from reactor_runtime.model.message import (
    MessageField,
    MessageFieldInfo,
    MessageFieldSpec,
    ModelMessage,
)
from reactor_runtime.model.tracks import Audio, Input, Output, Track, Video

__all__ = [
    "Audio",
    "Command",
    "EventHandler",
    "FieldInfo",
    "Input",
    "InputField",
    "MessageField",
    "MessageFieldInfo",
    "MessageFieldSpec",
    "ModelMessage",
    "Output",
    "Track",
    "UploadedFile",
    "Video",
    "connected",
    "disconnected",
    "event",
    "file_uploaded",
    "make_command",
    "session_ended",
    "session_started",
]
