"""Serve several sessions from one copy of a model's weights.

A model split in two runs its model half in one process, the model server, and
its application half in ordinary one-session runtimes, one per session, that
all call it. The two halves are written together, in one workspace, and what
they exchange is pickled::

    class PlasmaModel(ModelServer[Tick, Frame]):   # the model half, in the model server
        def step(self, requests): ...

    # The application half, in each session; a lost model server ends it.
    self.model = ModelClient.connect(on_lost=self.crash)
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
    ModelServerError,
    ModelServerLost,
    ModelServerRejected,
    ModelServerStopping,
    ModelServerTimeout,
    ModelServerUnreachable,
    PayloadTooLarge,
    ResetRequired,
    StepExpired,
    StepStillRunning,
    StepTimedOut,
    StepUnanswered,
)
from reactor_runtime.model_server.link import (
    LinkListener,
    LinkTransport,
    ModelServerLink,
    NoListener,
    StreamLink,
)
from reactor_runtime.model_server.server import (
    DEFERRED,
    Deferred,
    ModelServer,
    ServerRequest,
    ServerSettings,
)
from reactor_runtime.model_server.server_core import ModelServerCore
from reactor_runtime.model_server.shm_link import ShmLink, UnixShmListener, UnixShmTransport

__all__ = [
    "DEFERRED",
    "BatchFailed",
    "ClientClosed",
    "Deferred",
    "LinkConfig",
    "LinkListener",
    "LinkState",
    "LinkTransport",
    "ModelClient",
    "ModelServer",
    "ModelServerCore",
    "ModelServerError",
    "ModelServerLink",
    "ModelServerLost",
    "ModelServerRejected",
    "ModelServerStopping",
    "ModelServerTimeout",
    "ModelServerUnreachable",
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
    "link_config",
]
