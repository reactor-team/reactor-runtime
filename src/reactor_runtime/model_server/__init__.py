"""Serve several sessions from one copy of a model's weights.

A model split in two runs its model half in one process, ``python -m
reactor_runtime.model_server``, which owns the weights and the GPU, and its
application half in ordinary one-session runtimes, ``python -m
reactor_runtime.serve``, one per session. The split is the one
``skills/application-model-isolation`` describes, with the model half in its
own process: the skill owns its rules. The model half refuses what it cannot
serve and never recovers on its own, and only the application calls
:meth:`ModelClient.reset`. The sessions reach the worker through a Unix
socket, with the arrays in shared memory, so they run on one host: in
Kubernetes, one pod with the worker as a native sidecar and the session
runtimes as its containers.

:class:`ModelServer` is the model half that serves many sessions from one
process; it is not :class:`~reactor_runtime.distributed.DistributedWorker`,
which is one rank of a model split over several GPUs. The two compose: a
``ModelServer`` whose ``step()`` drives distributed ranks serves every session
from weights spread over those GPUs. The client is a :class:`ModelClient`.
"Worker" names the process that runs a ``ModelServer``: its environment
variables, its socket, and the errors that describe it (:class:`WorkerLost`,
:class:`WorkerError`).

The manifest names both halves; the worker block also holds the serving
settings, within the bounds the worker class sets::

    runtime:
      import: plasma:Plasma                 # the application half, a ReactorApp
      worker:
        import: plasma_worker:PlasmaWorker  # the model half, a ModelServer
        max_batch: 4

The types the halves exchange are frozen dataclasses, declared once in the
model's code and imported by both::

    @dataclass(frozen=True)
    class Tick:
        tick: int

    class PlasmaWorker(ModelServer):
        request_type = Tick
        max_batch = 4

        def step(self, requests):
            return [render(self.palette, r.payload.tick) for r in requests]

The application half connects in ``load()`` (blocking until the worker
welcomes it), resets its slot when a session starts, and calls the worker from
``generate()``::

    class Plasma(ReactorApp):
        def load(self, config_path):
            self.worker = ModelClient.connect(request_type=Tick)

        @session_started
        def begin(self):
            self.worker.reset()

        def generate(self, input):
            return PlasmaOutput(main_video=self.worker.step(Tick(self.tick)))

A lost or a stuck worker reaches the application on its own call path: the
step raises :class:`WorkerLost` or :class:`StepTimedOut` (a
:class:`StepUnanswered`), and the error arrives in ``process_output()`` as
``outcome.error``. The application decides, in one of two ways.

**Option 1, re-raise.** The session ends with an error and the session process
exits; its restart blocks in ``load()`` until the worker is back. No recovery
code::

    async def process_output(self, outcome):
        if isinstance(outcome.error, (WorkerLost, StepUnanswered)):
            raise outcome.error

**Option 2, report and recover.** The process stays up and warm: the session
ends with the reason, the runtime refuses new sessions until the worker is
back, and a worker that keeps dying backs off alone. The application owns one
watcher, started in ``load()``, that also sees a worker lost between steps or
while idle. Every wait in it is bounded, so shutdown is never held::

    class Plasma(ReactorApp):
        def load(self, config_path):
            self.worker = ModelClient.connect(request_type=Tick)
            self._give_up = threading.Event()
            threading.Thread(target=self._watch, daemon=True).start()

        def _watch(self):
            while True:
                try:
                    reason = self.worker.wait_lost(timeout=1.0)
                except TimeoutError:
                    if not self._give_up.is_set():
                        continue
                    reason = self.unavailable or "worker_unavailable"
                except ClientClosed:
                    return
                self.report_unavailable(reason)          # ends the session, refuses new ones
                while True:
                    try:
                        self.worker.reconnect(timeout=5.0)
                        break
                    except (WorkerTimeout, WorkerRejected, WorkerUnreachable):
                        continue
                    except ClientClosed:
                        return
                self._give_up.clear()
                self.report_available()                  # /health 200, sessions accepted

        def give_up(self, reason):
            self.report_unavailable(reason)
            self._give_up.set()

        @session_started
        def begin(self):
            try:
                self.worker.reset()
            except (WorkerLost, StepUnanswered, WorkerRejected) as exc:
                self.give_up(f"worker_unavailable: reset failed: {exc}")

        async def process_input(self):
            if self.unavailable is not None:
                raise ApplicationError(self.unavailable)  # refuse steps until recovered
            ...

        async def process_output(self, outcome):
            if isinstance(outcome.error, (WorkerLost, StepUnanswered)):
                self.give_up(str(outcome.error))
                return None
            ...

:meth:`~reactor_runtime.ReactorApp.report_unavailable` is non-blocking and
safe from any thread: its first call journals an ``error`` fact and stops the
live session with end reason ``error`` and the reason as its close reason;
``/health`` then answers 503 with the reason and ``/start_session`` answers
503 with ``Retry-After`` until :meth:`~reactor_runtime.ReactorApp.report_available`.
``examples/shared_worker`` uses option 2.

A model that does not split is served exactly as before: the runtime does not
import this package.
"""

from reactor_runtime.model_server.contract import (
    ModelServer,
    ServerRequest,
    ServerSettings,
    cuda_health,
)
from reactor_runtime.model_server.schema import SchemaError

__all__ = [
    "ModelServer",
    "SchemaError",
    "ServerRequest",
    "ServerSettings",
    "cuda_health",
]
