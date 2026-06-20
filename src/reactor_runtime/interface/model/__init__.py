"""The model base and the contract resolved from it.

:class:`ReactorModel` is what an author subclasses; declaring the subclass
assembles its :class:`ModelContract` from one traversal of the class and renders
the :class:`ModelSchema` a client reads. The contract and schema types are the
integration surface the runtime consumes, not part of the curated authoring API.
"""

from reactor_runtime.interface.model.contract import (
    CommandSpec,
    ContractError,
    LifecycleHooks,
    ModelContract,
)
from reactor_runtime.interface.model.reactor_model import ReactorModel
from reactor_runtime.interface.model.schema import (
    CommandSchema,
    MessageSchema,
    ModelSchema,
    TrackSchema,
)

__all__ = [
    "CommandSchema",
    "CommandSpec",
    "ContractError",
    "LifecycleHooks",
    "MessageSchema",
    "ModelContract",
    "ModelSchema",
    "ReactorModel",
    "TrackSchema",
]
