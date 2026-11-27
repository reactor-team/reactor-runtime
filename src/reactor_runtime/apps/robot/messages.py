"""What a robot policy tells every client after a step: ``actions`` and ``actions_failed``.

Both are broadcast to every connection of the session and carry the step and
the capture stamp of the observation the step ran on, so a robot can tell
which of its captures the result belongs to::

    {"type": "actions", "data": {"step": 7, "capture_time_us": 1712345678901,
     "chunk": [[0.1, 0.2], [0.15, 0.25]], "dt_ms": 20.0, "names": ["j0", "j1"],
     "extra": {}, "generate_ms": 41.3, "observation_age_ms": 44.0}}
    {"type": "actions_failed", "data": {"step": 8, "capture_time_us": 1712345678934,
     "reason": "RuntimeError: CUDA out of memory"}}
"""

from __future__ import annotations

from typing import Any

from reactor_runtime.interface.events.messages import MessageField, ModelMessage


class ActionsMessage(ModelMessage):
    """The chunk of actions a policy computed from one observation.

    Sent once for every step whose ``generate()`` returned
    :class:`~reactor_runtime.apps.robot.Actions`, after ``process_output()``
    and before the step's media.
    """

    name = "actions"

    step: int = MessageField(
        description="The step that computed the chunk, counted from 0 in the session."
    )
    capture_time_us: int = MessageField(
        description=(
            "The stamp of the capture the step observed, in microseconds on the robot's "
            "monotonic clock, as the robot sent it."
        )
    )
    chunk: list[list[float]] = MessageField(
        description=(
            "The actions, one row per action and one column per actuator value. Row i is "
            "meant to play i * dt_ms after the capture."
        )
    )
    dt_ms: float = MessageField(description="Milliseconds between two consecutive rows.")
    names: list[str] | None = MessageField(
        description="The name of each column, or null when the columns are unnamed."
    )
    extra: dict[str, Any] = MessageField(
        description="Anything else the policy returned with the chunk, empty when nothing."
    )
    generate_ms: float = MessageField(
        description="Milliseconds the model's generate() took for this step."
    )
    observation_age_ms: float = MessageField(
        description=(
            "Milliseconds from the runtime receiving the first frame or state of the capture "
            "to this message being sent, on the runtime's clock."
        ),
    )


class ActionsFailed(ModelMessage):
    """A step whose ``generate()`` raised, so its observation has no actions.

    The policy loop goes on with the next observation.
    """

    step: int = MessageField(description="The step that failed, counted from 0 in the session.")
    capture_time_us: int = MessageField(
        description=(
            "The stamp of the capture the step observed, in microseconds on the robot's "
            "monotonic clock, as the robot sent it."
        )
    )
    reason: str = MessageField(description="The error generate() raised, as its type and message.")
