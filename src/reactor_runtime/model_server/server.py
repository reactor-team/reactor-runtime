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
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from reactor_runtime.core import (
    Health,
    ModelServerConfig,
    SharedMemoryConfig,
    shm_slot_bytes,
)

_MANIFEST_ONLY = {
    "shm_input_bytes": "model.resources.sessions.shared_memory.input",
    "shm_output_bytes": "model.resources.sessions.shared_memory.output",
}


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
    * Requests are taken in arrival order. When the model server is free and a
      request waits, it waits up to ``batch_window`` more seconds for others,
      stopping early once ``max_batch`` wait or every live session has one
      (one that opened, reset or sent a request in the last two seconds; a
      session runtime with no client never holds a batch back for long),
      then steps up to ``max_batch`` of them together.
    * A session that connects, or a request that arrives, while a step runs
      joins the next step, never the running one.
    * A request whose timeout passed while it waited is answered as expired and
      never reaches ``step()``.
    * A request whose output is a :class:`concurrent.futures.Future` keeps
      running after ``step()`` returns, while other sessions' requests are
      stepped, and its session stays busy until the future is done. See
      :meth:`step`.

    A pipeline hands each request to its first stage and returns a future
    that its last stage completes::

        def step(self, requests):
            futures = []
            for request in requests:
                future = Future()
                future.set_running_or_notify_cancel()   # its work starts now
                self.pipeline.submit(request, future)
                futures.append(future)
            return futures

        # on the pipeline's last stage, once per request
        future.set_result(frame)          # or future.set_exception(error)

        def session_closed(self, slot, generation):
            # Stop that generation's work and wait for any stage still
            # writing the slot's state, then resolve its futures.
            self.pipeline.discard(slot, generation)

    ``reactor.yaml`` sets how many requests one step can take::

        runtime:
          model_server:
            import: plasma_model:PlasmaModel
            max_batch: 4      # the most requests in one step

    How many sessions share the model server is a separate setting,
    ``model.resources.sessions.count``. When the file sets no count, the
    ``reactor`` CLI sets it to ``max_batch``; an explicit count stays
    independent of ``max_batch``. With more sessions
    than ``max_batch``, the requests past ``max_batch`` wait for the next
    step, so a model that runs one request per step keeps ``max_batch`` at 1
    and serves its sessions in turn.

    What crosses between the halves is pickled, so a request, a result and a
    signal can be any value pickle takes. Plain data, such as frozen
    dataclasses of numbers, strings and numpy arrays, is the convention: the
    session processes hold no GPU, so a CUDA tensor cannot be rebuilt there,
    and a callable runs in whichever process calls it. The type parameters
    are for type checkers alone; nothing checks them at run time.

    Class attributes:
        batch_window: Seconds to wait for a batch to fill, 0 by default;
            ``runtime.model_server.batch_window`` overrides it.

    The sizes of the shared memory each session holds are set in
    ``reactor.yaml`` (``model.resources.sessions.shared_memory``) and nowhere
    else: a deployment sizes each worker's memory from the manifest, so a class
    that declares them is refused.

    Attributes:
        device: ``"cuda"`` when torch sees a GPU in the model server's process,
            else ``"cpu"``. Set before :meth:`load` runs.
        max_batch: The most requests one step receives, from
            ``runtime.model_server.max_batch`` in ``reactor.yaml``; 1 when
            neither the manifest nor a deprecated class declaration sets it.
            Set after the constructor, over any value the constructor
            sets, and before :meth:`load` runs, so ``load()`` can warm the
            model up at the full batch.

    A subclass that declares ``max_batch`` itself is deprecated, and the
    model server logs a warning that names the key to move it to. It serves
    with the class value when ``reactor.yaml`` sets none, with the manifest's
    when it is at most the class value, and refuses to start when the
    manifest's is above it.

    Raises:
        TypeError: When a subclass is defined with a setting the manifest owns.
        ValueError: When a subclass is defined with ``max_batch`` or
            ``batch_window`` out of range.
    """

    batch_window: ClassVar[float] = 0.0

    device: str
    max_batch: int = 1

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check the subclass's declarations as it is defined."""
        super().__init_subclass__(**kwargs)
        for attribute, key in _MANIFEST_ONLY.items():
            if hasattr(cls, attribute):
                raise TypeError(
                    f"{cls.__name__}.{attribute}: set it in reactor.yaml as {key}, "
                    "where a deployment reads it"
                )
        declared = _declared_max_batch(cls)
        if declared is not None:
            _positive_int(f"{cls.__name__}.max_batch", declared)
        _seconds(f"{cls.__name__}.batch_window", cls.batch_window)

    def load(self, config_path: Path | None) -> None:
        """Load the weights, once, before any session connects.

        Args:
            config_path: The file ``runtime.config`` in ``reactor.yaml`` names,
                resolved to an absolute path, or ``None``.

        A raise here fails the model server's start, and its process exits.
        """

    def step(
        self, requests: list[ServerRequest[RequestT]]
    ) -> Sequence[ResultT | Exception | Future[ResultT]]:
        """Run one step for every request and return one output per request, in order.

        An output that is an exception instance fails that request alone, and
        reaches that session's application as itself. This is how a model
        fails one bad request without failing the others in its batch.

        A raise from ``step()`` fails every request of the batch: each session
        sees a :class:`~reactor_runtime.model_server.BatchFailed` carrying the
        error, and its application decides what to do. Nothing is stepped
        again, and the model server goes on serving; :meth:`health` says
        whether it can. Requests whose futures earlier steps returned are not
        part of the batch, and keep running.

        An output that is a :class:`concurrent.futures.Future` completes its
        request later, from any thread, as a stdlib executor completes one:

        * Call ``set_running_or_notify_cancel()`` when the request's work
          starts, before anything it does is visible, and skip the work if it
          returns ``False``. Then call ``set_result()`` or ``set_exception()``
          exactly once. An exception fails that request alone, as an
          exception output does.
        * The model server cancels the future when the request's timeout
          passes, when its session gives up on it, resets or leaves, and when
          the model server stops. ``cancel()`` only reaches work that has not
          started: once running, the work goes on, and its result is dropped
          if the session no longer waits for it. ``set_result()`` on a future
          cancelled before it ran raises ``InvalidStateError``, which is why
          the work starts with ``set_running_or_notify_cancel()``.
        * Until the future is done, its session stays busy: a step it sends is
          answered with :class:`~reactor_runtime.model_server.StepStillRunning`,
          even after the request's timeout passed and the session got
          :class:`~reactor_runtime.model_server.StepExpired`, so the model
          never has two requests of one session running. A reset or a leave
          frees the session at once instead, and :meth:`session_closed` keeps
          the closed generation's work away from the next one's state.
        * Every future must be done eventually, its session closed or not: one
          that is not done ``stuck_after`` seconds after ``step()`` returned it
          makes the model server report itself unhealthy, so a pipeline that
          stops making progress is restarted.
        * The future's done-callbacks run on the thread that completes it,
          and are short: they finish the request and queue its answer.

        Args:
            requests: Between one and ``max_batch`` requests, at
                most one per session, in arrival order.

        Returns:
            One output per request, in the order of *requests*: its result,
            an exception, or a future that resolves to one of them.
        """
        raise NotImplementedError(f"{type(self).__name__} must define step()")

    def session_opened(self, slot: int, generation: int) -> None:
        """Set up the state of the session that just opened *slot*.

        A raise refuses the session: its application sees a
        :class:`~reactor_runtime.model_server.ModelServerRejected`.
        """

    def session_closed(self, slot: int, generation: int) -> None:
        """Drop the state of the session that held *slot*.

        Runs when the session runtime disconnects or dies, and before a reset
        reopens the slot. For any slot it runs before the next
        :meth:`session_opened`. A raise is logged and ignored.

        A model that returns futures stops this generation's work here: work
        that already started is out of ``cancel()``'s reach (see :meth:`step`),
        so return only once no thread will write state the next session on
        *slot* uses, and resolve the generation's futures, whose results the
        model server drops. Its time counts toward ``stuck_after``.
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
        the model server unhealthy, with the detail, when it is, so the
        deployment can restart it. The default reports healthy; override it with
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
    """The settings a model server serves with, from ``reactor.yaml`` and its class.

    Attributes:
        max_batch: The most requests one step receives.
        batch_window: Seconds to wait for a batch to fill.
        shm_input_bytes: Each session's request block, from
            ``model.resources.sessions.shared_memory.input``.
        shm_output_bytes: Each session's output block, from
            ``model.resources.sessions.shared_memory.output``.
    """

    max_batch: int
    batch_window: float
    shm_input_bytes: int
    shm_output_bytes: int

    @property
    def shm_slot_bytes(self) -> int:
        """The most shared memory one session's slot holds, which its volume must fit.

        :func:`~reactor_runtime.core.shm_slot_bytes` of the two block sizes.
        """
        return shm_slot_bytes(self.shm_input_bytes, self.shm_output_bytes)

    @classmethod
    def resolve(
        cls,
        server: type[ModelServer],
        config: ModelServerConfig | None,
        shared_memory: SharedMemoryConfig | None = None,
    ) -> ServerSettings:
        """Return the settings *server* serves with under the manifest.

        ``max_batch`` is ``runtime.model_server.max_batch``; a class that
        declares ``max_batch`` (deprecated) supplies it when the manifest sets
        none and caps what the manifest may set, and 1 applies when neither
        does. ``batch_window`` is the
        manifest's, else the class's. The shared memory sizes are the
        manifest's, with the runtime's defaults for what it leaves out.

        Args:
            server: The model half's class.
            config: The manifest's ``runtime.model_server`` block, or ``None``
                when the manifest has none.
            shared_memory: The manifest's ``model.resources.sessions.shared_memory``
                block; the default sizes when omitted.

        Raises:
            ValueError: The manifest sets ``max_batch`` above the one the class
                declares.
        """
        blocks = shared_memory or SharedMemoryConfig()
        declared = _declared_max_batch(server)
        configured = config.max_batch if config is not None else None
        if declared is not None and configured is not None and configured > declared:
            raise ValueError(
                f"runtime.model_server.max_batch is {configured}, above "
                f"{server.__name__}.max_batch = {declared}: remove max_batch from the class, "
                "reactor.yaml sets it"
            )
        if configured is not None:
            max_batch = configured
        elif declared is not None:
            _positive_int(f"{server.__name__}.max_batch", declared)
            max_batch = declared
        else:
            max_batch = 1
        batch_window = float(server.batch_window)
        if config is not None and config.batch_window is not None:
            batch_window = float(config.batch_window)
        return cls(
            max_batch=max_batch,
            batch_window=batch_window,
            shm_input_bytes=blocks.input_bytes,
            shm_output_bytes=blocks.output_bytes,
        )


def is_max_batch_declared(server: type[ModelServer]) -> bool:
    """Return whether *server*, or a base below :class:`ModelServer`, declares ``max_batch``.

    Such a declaration is deprecated: ``runtime.model_server.max_batch`` in
    ``reactor.yaml`` sets it.
    """
    return _declared_max_batch(server) is not None


def _declared_max_batch(server: type[ModelServer]) -> Any:
    for klass in server.__mro__:
        if klass is ModelServer:
            return None
        if "max_batch" in vars(klass):
            return vars(klass)["max_batch"]
    return None


def _positive_int(where: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{where} must be a positive integer, got {value!r}")


def _seconds(where: str, value: Any) -> None:
    if type(value) not in (int, float) or not 0 <= value < math.inf:
        raise ValueError(f"{where} must be a number of seconds, at least 0, got {value!r}")
