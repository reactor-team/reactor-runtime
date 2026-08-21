"""The model base, the vocabulary it steps with, and the contract resolved from it.

:class:`ReactorModel` is what an author subclasses; declaring the subclass
assembles its :class:`ModelContract` from one traversal of the class and renders
the :class:`ModelSchema` a client reads. :class:`InputState` declares what a
client may change mid-session, and :class:`SteppedModel`, :class:`StepStats`,
and :class:`NotReady` are the vocabulary of a runtime-driven step. The contract
and schema types are the integration surface the runtime consumes, not part of
the curated authoring API.
"""

from reactor_runtime.interface.model.contract import (
    CommandSpec,
    ContractError,
    LifecycleHooks,
    ModelContract,
)
from reactor_runtime.interface.model.input_state import InputState
from reactor_runtime.interface.model.reactor_model import ReactorModel
from reactor_runtime.interface.model.schema import (
    CommandSchema,
    MessageSchema,
    ModelSchema,
    TrackSchema,
)
from reactor_runtime.interface.model.stepping import NotReady, SteppedModel, StepStats

__all__ = [
    "CommandSchema",
    "CommandSpec",
    "ContractError",
    "InputState",
    "LifecycleHooks",
    "MessageSchema",
    "ModelContract",
    "ModelSchema",
    "NotReady",
    "ReactorModel",
    "StepStats",
    "SteppedModel",
    "TrackSchema",
]
