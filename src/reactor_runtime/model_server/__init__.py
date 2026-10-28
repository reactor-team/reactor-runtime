"""Serve several sessions from one copy of a model's weights.

A model split in two runs its model half in one process, the model server, and
its application half in ordinary one-session runtimes, one per session, that
all call it. What the halves exchange is pickled, written together in one
workspace::

    class PlasmaModel(ModelServer[Tick, Frame]):   # the model half, in the model server
        def step(self, requests): ...

The rules of the split are in ``skills/application-model-isolation``. A model
that does not split is served exactly as before: the runtime does not import
this package.
"""

from reactor_runtime.model_server.errors import (
    BatchFailed,
    StepExpired,
    StepUnanswered,
    WorkerError,
    WorkerLost,
    WorkerRejected,
    WorkerStopping,
)
from reactor_runtime.model_server.server import (
    ModelServer,
    ServerRequest,
    ServerSettings,
)

__all__ = [
    "BatchFailed",
    "ModelServer",
    "ServerRequest",
    "ServerSettings",
    "StepExpired",
    "StepUnanswered",
    "WorkerError",
    "WorkerLost",
    "WorkerRejected",
    "WorkerStopping",
]
