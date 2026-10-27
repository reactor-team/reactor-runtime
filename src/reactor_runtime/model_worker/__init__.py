"""Serve several sessions from one copy of a model's weights.

A model split in two runs its worker half in one process, ``reactor-runtime
serve-worker``, which owns the weights and the GPU, and its session half in
ordinary one-session runtimes, ``reactor-runtime serve``, one per session. The
sessions reach the worker through a Unix socket, with the arrays in shared
memory, so they run on one host: in Kubernetes, one pod with the worker as a
native sidecar and the session runtimes as its containers.

The manifest names both halves::

    runtime:
      import: plasma:Plasma               # the session half, a ReactorApp
      worker: plasma_worker:PlasmaWorker  # the worker half, a ModelWorker

The worker half owns the weights and steps every session's request::

    class PlasmaWorker(ModelWorker):
        max_batch = 4

        def load(self, config_path):
            self.palette = ...

        def step(self, requests):
            return [render(self.palette, r.payload) for r in requests]

The session half is an ordinary ``ReactorApp``; its ``generate()`` sends what
the worker needs and returns what comes back::

    class Plasma(ReactorApp):
        def load(self, config_path):
            self.worker = WorkerClient.connect(self)

        def generate(self, input):
            return PlasmaOutput(main_video=self.worker.step({"tick": self.tick}))

Passing the app to :meth:`WorkerClient.connect` lets the client tell the
runtime when the worker goes and comes back: the session ends with
``worker_died: <why>``, the runtime reports not ready, and it serves again once
the worker is back, without the process exiting. A model that does not split
is served exactly as before.
"""

from reactor_runtime.distributed.errors import WorkerCrashed, WorkerTimeout
from reactor_runtime.model_worker.client import (
    WorkerClient,
    WorkerError,
    WorkerLost,
    WorkerRejected,
)
from reactor_runtime.model_worker.contract import ModelWorker, WorkerRequest
from reactor_runtime.model_worker.transport import Transport, UnixShmTransport, WorkerUnreachable

__all__ = [
    "ModelWorker",
    "Transport",
    "UnixShmTransport",
    "WorkerClient",
    "WorkerCrashed",
    "WorkerError",
    "WorkerLost",
    "WorkerRejected",
    "WorkerRequest",
    "WorkerTimeout",
    "WorkerUnreachable",
]
