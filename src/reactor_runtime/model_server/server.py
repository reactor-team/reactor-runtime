"""The model half that serves several sessions from one process, and its serving settings.

A :class:`ModelServer` subclass is the model half of the split
``skills/application-model-isolation`` describes, constructed once in the model
server's process. Each session runtime's application half reaches it with a
:class:`~reactor_runtime.model_server.ModelClient`. Two rules of the skill shape
this class: the model half refuses what it cannot serve and leaves the
recovery to the application, and only the application resets a slot.

It is not :class:`~reactor_runtime.distributed.DistributedWorker`, which is one
rank of a model split over several GPUs; a ``ModelServer`` whose ``step()``
drives such ranks serves every session from weights spread over those GPUs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from reactor_runtime.core import (
    DEFAULT_SHM_INPUT_BYTES,
    DEFAULT_SHM_OUTPUT_BYTES,
    Health,
    WorkerConfig,
)

_MANIFEST_ONLY = ("shm_input_bytes", "shm_output_bytes")


@dataclass(frozen=True)
class ServerRequest[T]:
    """One session's request in a batch.

    Attributes:
        slot: The session runtime's slot, ``0`` to the deployment's session
            count minus one. A session runtime that restarts keeps its slot.
        generation: How many times this slot has been opened, from ``1``. It
            moves on when a session runtime connects in the slot and at every
            reset, so state kept per slot can tell a new session from the one
            before.
        payload: The request the session's application sent. Its arrays are
            its own: keep them past ``step()`` as they are.
        session_id: The id of the session this request belongs to, as the
            session's last reset named it, for the model's own log lines;
            ``None`` before the first reset.
    """

    slot: int
    generation: int
    payload: T
    session_id: str | None = None


class ModelServer[RequestT, ResultT]:
    """Base class for the model half: weights loaded once, one batch stepped at a time.

    A subclass declares what it can serve and writes :meth:`step`::

        class PlasmaModel(ModelServer[Tick, Frame]):
            max_batch = 4             # the most requests one step can take

            def load(self, config_path):
                self.palette = load_palette(config_path)

            def step(self, requests):
                return [Frame(pixels=render(self.palette, r.payload.tick)) for r in requests]

    The model server's process calls :meth:`load` once, then calls :meth:`step`
    with the requests waiting from its sessions, one call at a time, on one
    thread. :meth:`session_opened` and :meth:`session_closed` run on that same
    thread between steps, so state kept per slot needs no lock against
    ``step()``. :meth:`signal` runs on the thread that reads that session's
    connection, and :meth:`health` on the health check's thread, both possibly
    while a step runs.

    How requests are batched:

    * A session has at most one request waiting or running.
    * Requests are taken in arrival order. When the worker is free and a
      request waits, it waits up to ``batch_window`` more seconds for others,
      stopping early once ``max_batch`` wait or every open session has one,
      then steps up to ``max_batch`` of them together.
    * A session that connects, or a request that arrives, while a step runs
      joins the next step, never the running one.
    * A request whose timeout passed while it waited is answered as expired and
      never reaches ``step()``.

    What crosses between the halves is pickled, so a request, a result and a
    signal can be any value pickle takes. Plain data, such as frozen
    dataclasses of numbers, strings and numpy arrays, is the convention: the
    session processes hold no GPU, so a CUDA tensor cannot be rebuilt there,
    and a callable runs in whichever process calls it. The type parameters
    are for type checkers alone; nothing checks them at run time.

    Class attributes:
        max_batch: The most requests one step can take, 1 by default.
            ``runtime.worker.max_batch`` in ``reactor.yaml`` may lower it, never
            raise it.
        batch_window: Seconds to wait for a batch to fill, 0 by default;
            ``runtime.worker.batch_window`` overrides it.

    The sizes of the shared memory each session holds are set in
    ``reactor.yaml`` (``runtime.worker``) and nowhere else: a deployment sizes
    each session's memory from the manifest, so a class that declares them is
    refused.

    Attributes:
        device: ``"cuda"`` when torch sees a GPU in the model server's process,
            else ``"cpu"``. Set before :meth:`load` runs.

    Raises:
        TypeError: When a subclass is defined with a setting the manifest owns.
        ValueError: When a subclass is defined with ``max_batch`` or
            ``batch_window`` out of range.
    """

    max_batch: ClassVar[int] = 1
    batch_window: ClassVar[float] = 0.0

    device: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check the subclass's declarations as it is defined."""
        super().__init_subclass__(**kwargs)
        for key in _MANIFEST_ONLY:
            if hasattr(cls, key):
                raise TypeError(
                    f"{cls.__name__}.{key}: set it in reactor.yaml as runtime.worker.{key}, "
                    "where a deployment reads it"
                )
        _positive_int(f"{cls.__name__}.max_batch", cls.max_batch)
        _seconds(f"{cls.__name__}.batch_window", cls.batch_window)

    def load(self, config_path: Path | None) -> None:
        """Load the weights, once, before any session connects.

        Args:
            config_path: The file ``runtime.config`` in ``reactor.yaml`` names,
                resolved to an absolute path, or ``None``.

        A raise here fails the model server's start, and its process exits.
        """

    def step(self, requests: list[ServerRequest[RequestT]]) -> Sequence[ResultT | Exception]:
        """Run one step for every request and return one output per request, in order.

        An output that is an exception instance fails that request alone, and
        reaches that session's application as itself. This is how a model
        fails one bad request without failing the others in its batch.

        A raise from ``step()`` fails every request of the batch: each session
        sees a :class:`~reactor_runtime.model_server.BatchFailed` carrying the
        error, and its application decides what to do. Nothing is stepped
        again, and the worker goes on serving; :meth:`health` says whether it
        can.

        Args:
            requests: Between one and the effective ``max_batch`` requests, at
                most one per session, in arrival order.

        Returns:
            One output per request, in the order of *requests*: its result,
            or an exception.
        """
        raise NotImplementedError(f"{type(self).__name__} must define step()")

    def session_opened(self, slot: int, generation: int) -> None:
        """Set up the state of the session that just opened *slot*.

        A raise refuses the session: its application sees a
        :class:`~reactor_runtime.model_server.WorkerRejected`.
        """

    def session_closed(self, slot: int, generation: int) -> None:
        """Drop the state of the session that held *slot*.

        Runs when the session runtime disconnects or dies, and before a reset
        reopens the slot; for any slot it runs before the next
        :meth:`session_opened`. A raise is logged and ignored.
        """

    def signal(self, slot: int, signal: Any) -> None:
        """Take a signal a session sent outside a step, such as a request to stop early.

        *signal* is what the application passed to ``signal()``. It runs
        on that session's reader thread, possibly while :meth:`step` runs, so
        it should only record the signal for ``step()`` to read. Signals from
        one session arrive in the order sent, and one sent before a request is
        taken before that request can join a step. A raise is logged and
        ignored.
        """

    def health(self) -> Health:
        """Report whether the model can serve, as every runtime component does.

        The model server's health check calls it on every probe and reports
        the worker unhealthy, with the detail, when it is, so the deployment
        can restart the worker. The default reports healthy; override it with
        the model's own signals of trouble. It runs on the health check's
        thread, concurrently with :meth:`step`: read state the step writes
        without blocking on it, and keep it quick. A raise is reported as
        unhealthy. For example, a model that latches a CUDA error it cannot
        recover from::

            def step(self, requests):
                try:
                    return self.render(requests)
                except torch.cuda.CudaError as exc:
                    self.broken = str(exc)    # the model's own latch
                    raise

            def health(self) -> Health:
                if self.broken:
                    return Health(HealthStatus.UNHEALTHY, self.broken)
                return Health.healthy()
        """
        return Health.healthy()


@dataclass(frozen=True)
class ServerSettings:
    """The settings a model server serves with: its class's, as ``reactor.yaml`` adjusts them.

    Attributes:
        max_batch: The most requests one step receives.
        batch_window: Seconds to wait for a batch to fill.
        shm_input_bytes: Each session's request block.
        shm_output_bytes: Each session's output block.
    """

    max_batch: int
    batch_window: float
    shm_input_bytes: int
    shm_output_bytes: int

    @property
    def shm_slot_bytes(self) -> int:
        """The shared memory one session holds: its request and output blocks together."""
        return self.shm_input_bytes + self.shm_output_bytes

    @classmethod
    def resolve(cls, server: type[ModelServer], config: WorkerConfig | None) -> ServerSettings:
        """Return the settings *server* serves with under the manifest's *config*.

        The class is the authority on ``max_batch``: the manifest may lower it,
        not raise it. The manifest sets the rest, with the runtime's defaults
        for what it leaves out.

        Raises:
            ValueError: The manifest raises ``max_batch`` above the class's.
        """
        max_batch = server.max_batch
        if config is not None and config.max_batch is not None:
            if config.max_batch > server.max_batch:
                raise ValueError(
                    f"runtime.worker.max_batch is {config.max_batch}, above "
                    f"{server.__name__}.max_batch = {server.max_batch}: the manifest may only "
                    "lower it"
                )
            max_batch = config.max_batch
        if config is None:
            return cls(
                max_batch=max_batch,
                batch_window=float(server.batch_window),
                shm_input_bytes=DEFAULT_SHM_INPUT_BYTES,
                shm_output_bytes=DEFAULT_SHM_OUTPUT_BYTES,
            )
        return cls(
            max_batch=max_batch,
            batch_window=float(
                server.batch_window if config.batch_window is None else config.batch_window
            ),
            shm_input_bytes=config.input_bytes,
            shm_output_bytes=config.output_bytes,
        )


def _positive_int(where: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{where} must be a positive integer, got {value!r}")


def _seconds(where: str, value: Any) -> None:
    if type(value) not in (int, float) or not 0 <= value < math.inf:
        raise ValueError(f"{where} must be a number of seconds, at least 0, got {value!r}")
