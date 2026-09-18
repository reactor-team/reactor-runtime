"""Model-authoring surface — the public API a model author writes against.

One obvious place to import from: the tracks a model exchanges, the ``@event``
handlers and lifecycle hooks that shape its command set, the typed messages it
sends back, and the field metadata that constrains them. These build on the
spine's neutral vocabulary and carry no engine — declaring them is enough to
resolve a model's contract.

Everything re-exported here is also available directly on the top-level
``reactor_runtime`` package, which is the preferred import path.

Four names are deprecated and import with a :class:`DeprecationWarning`.
``ReactorModel`` is :class:`ReactorApp` and ``Input`` is :class:`MediaInput`,
renames that resolve to the new class. ``ReactorPipeline`` and ``Idle`` still
resolve to themselves, so a model on the generator pattern keeps working, but
the pattern is retired in favour of :class:`ReactorApp` and ``generate()``.
All four are removed in the next major.
"""

from typing import Any

from reactor_runtime.core import (
    Command,
    FieldInfo,
    InputField,
    InputFrame,
    UploadedFile,
)
from reactor_runtime.interface.app import InputState, ReactorApp, StepOutcome
from reactor_runtime.interface.client import ClientInfo
from reactor_runtime.interface.events import (
    EVENT_REGISTRY,
    MESSAGE_REGISTRY,
    ApplicationError,
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
from reactor_runtime.interface.internal.aliases import (
    IDLE_DEPRECATION,
    PIPELINE_DEPRECATION,
    deprecated_alias,
)
from reactor_runtime.interface.internal.input_buffer import (
    BufferClosed,
    InputBuffer,
    ReadMode,
)
from reactor_runtime.interface.internal.reactor_core import OutputStream

# Bound under private names so the public ones stay out of the module's
# namespace and resolve through __getattr__, which is where the warning lives.
from reactor_runtime.interface.pipeline import Idle as _Idle
from reactor_runtime.interface.pipeline import ReactorPipeline as _ReactorPipeline
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
    "all_input_tracks",
    "all_output_tracks",
    "connected",
    "disconnected",
    "event",
    "file_uploaded",
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
