"""Serve several sessions from one copy of a model's weights.

A model split in two runs its model half in one process, the model server, and
its application half in ordinary one-session runtimes, one per session, that
all call it. The two halves are written together, in one workspace, and what
they exchange is pickled::

    class PlasmaModel(ModelServer[Tick, Frame]):   # the model half, in the model server
        def step(self, requests): ...

    self.model = ModelClient.connect()             # the application half, in each session
    frame = self.model.step(Tick(tick=n))

The rules of the split are in ``skills/application-model-isolation``. A model
that does not split is served exactly as before: the runtime does not import
this package.
"""

from reactor_runtime.model_server.client import LinkState, ModelClient
from reactor_runtime.model_server.config import LinkConfig, link_config
from reactor_runtime.model_server.errors import (
    BatchFailed,
    ClientClosed,
    PayloadTooLarge,
    ResetRequired,
    StepExpired,
    StepStillRunning,
    StepTimedOut,
    StepUnanswered,
    WorkerError,
    WorkerLost,
    WorkerRejected,
    WorkerStopping,
    WorkerUnreachable,
)
from reactor_runtime.model_server.link import (
    LinkListener,
    LinkTransport,
    NoListener,
    StreamLink,
    WorkerLink,
)
from reactor_runtime.model_server.server import (
    ModelServer,
    ServerRequest,
    ServerSettings,
)
from reactor_runtime.model_server.shm_link import ShmLink, UnixShmListener, UnixShmTransport
from reactor_runtime.model_server.worker import WorkerServer

__all__ = [
    "BatchFailed",
    "ClientClosed",
    "LinkConfig",
    "LinkListener",
    "LinkState",
    "LinkTransport",
    "ModelClient",
    "ModelServer",
    "NoListener",
    "PayloadTooLarge",
    "ResetRequired",
    "ServerRequest",
    "ServerSettings",
    "ShmLink",
    "StepExpired",
    "StepStillRunning",
    "StepTimedOut",
    "StepUnanswered",
    "StreamLink",
    "UnixShmListener",
    "UnixShmTransport",
    "WorkerError",
    "WorkerLink",
    "WorkerLost",
    "WorkerRejected",
    "WorkerServer",
    "WorkerStopping",
    "WorkerUnreachable",
    "link_config",
]
