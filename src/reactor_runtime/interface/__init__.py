"""Model-authoring surface — the public API a model author writes against.

One obvious place to import from: the tracks a model exchanges, the ``@event``
handlers and lifecycle hooks that shape its command set, the typed messages it
sends back, and the field metadata that constrains them. These build on the
spine's neutral vocabulary and carry no engine — declaring them is enough to
resolve a model's contract.

Everything re-exported here is also available directly on the top-level
``reactor_runtime`` package, which is the preferred import path.

Two names are deprecated aliases. ``ReactorModel`` is :class:`ReactorApp` and
``Input`` is :class:`MediaInput`; both import with a :class:`DeprecationWarning`
and are removed in the next major.
"""

from typing import Any

from reactor_runtime.core import (
    Command,
    FieldInfo,
    InputField,
    InputFrame,
    UploadedFile,
)
from reactor_runtime.interface.app import ReactorApp
from reactor_runtime.interface.client import ClientInfo
from reactor_runtime.interface.events import (
    EVENT_REGISTRY,
    MESSAGE_REGISTRY,
    CommandError,
    MessageField,
    ModelMessage,
    connected,
    disconnected,
    event,
    file_uploaded,
    session_ended,
    session_started,
)
from reactor_runtime.interface.internal.aliases import deprecated_alias
from reactor_runtime.interface.internal.input_buffer import (
    BufferClosed,
    InputBuffer,
    ReadMode,
)
from reactor_runtime.interface.internal.reactor_core import OutputStream
from reactor_runtime.interface.pipeline import Idle, InputState, ReactorPipeline
from reactor_runtime.interface.tracks import (
    INPUT_REGISTRY,
    OUTPUT_REGISTRY,
    Audio,
    MediaInput,
    Metadata,
    Output,
    Track,
    TrackPayload,
    Video,
    all_input_tracks,
    all_output_tracks,
)

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
    "all_input_tracks",
    "all_output_tracks",
    "connected",
    "disconnected",
    "event",
    "file_uploaded",
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
