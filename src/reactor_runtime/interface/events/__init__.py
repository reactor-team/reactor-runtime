"""Client-facing events and messages — handler decorators and typed messages.

The ``@event`` decorator and the lifecycle hooks mark the methods a client can
drive; :class:`ModelMessage` is the typed payload a model sends back. Together
they are the conversational half of a model's contract, alongside its tracks.
"""

from reactor_runtime.interface.events.decorators import (
    EVENT_REGISTRY,
    EventHandler,
    connected,
    disconnected,
    event,
    file_uploaded,
    make_command,
    session_ended,
    session_started,
)
from reactor_runtime.interface.events.errors import CommandError
from reactor_runtime.interface.events.messages import (
    MESSAGE_REGISTRY,
    MessageField,
    MessageFieldInfo,
    MessageFieldSpec,
    ModelMessage,
)

__all__ = [
    "EVENT_REGISTRY",
    "MESSAGE_REGISTRY",
    "CommandError",
    "EventHandler",
    "MessageField",
    "MessageFieldInfo",
    "MessageFieldSpec",
    "ModelMessage",
    "connected",
    "disconnected",
    "event",
    "file_uploaded",
    "make_command",
    "session_ended",
    "session_started",
]
