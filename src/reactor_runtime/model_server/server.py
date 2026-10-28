"""The model half that serves several sessions from one process, and what it is declared with.

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

import dataclasses
import enum
import math
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args, get_origin

import numpy as np

from reactor_runtime.core import (
    DEFAULT_SHM_INPUT_BYTES,
    DEFAULT_SHM_OUTPUT_BYTES,
    Health,
    WorkerConfig,
)

_SCALARS: tuple[Any, ...] = (bool, int, float, str, bytes, type(None))
_MANIFEST_ONLY = ("shm_input_bytes", "shm_output_bytes")


@dataclass(frozen=True)
class ModelServerTypes:
    """The types a model's two halves exchange, declared once and imported by both.

    Each is a frozen dataclass whose fields are data: ``bool``, ``int``,
    ``float``, ``str``, ``bytes``, ``None``, ``numpy.ndarray``, an enum, a
    ``Literal``, another such dataclass, and ``list``, ``tuple``, ``dict`` (with
    ``str`` or ``int`` keys) and ``T | None`` of those. Only data crosses
    between the halves: a field that holds a function, a lock, or a tensor on a
    GPU is refused when the declaration is made, so the session processes,
    which hold no GPU, only ever receive arrays::

        @dataclass(frozen=True)
        class Tick:
            tick: int
            audio: np.ndarray

        @dataclass(frozen=True)
        class Frame:
            pixels: np.ndarray

        PLASMA = ModelServerTypes(request=Tick, result=Frame)

    If the other half must react to something, it is a declared signal it
    handles itself, never code that travels.

    Attributes:
        request: What a session sends to the model's ``step()``.
        result: What ``step()`` returns for one request.
        signals: What a session may send outside a step, such as a request to
            stop early.

    Raises:
        TypeError: A type is not a frozen dataclass of data fields, or a signal
            type is listed twice.
    """

    request: type
    result: type
    signals: tuple[type, ...] = ()

    def __post_init__(self) -> None:
        """Check every declared type once, when the declaration is made."""
        check_data_type(self.request, "request")
        check_data_type(self.result, "result")
        if not isinstance(self.signals, tuple):
            raise TypeError(f"signals must be a tuple of frozen dataclasses, got {self.signals!r}")
        if len(set(self.signals)) != len(self.signals):
            raise TypeError("signals names a type twice")
        for declared in self.signals:
            check_data_type(declared, "signals")


def check_data_type(cls: Any, where: str) -> None:
    """Refuse *cls* unless it is a frozen dataclass whose fields are data.

    Raises:
        TypeError: Naming the first field that is not data, or why *cls* is not
            such a dataclass.
    """
    _check_dataclass(cls, where, set())


def _check_dataclass(cls: Any, where: str, seen: set[type]) -> None:
    if not isinstance(cls, type) or "__dataclass_fields__" not in cls.__dict__:
        raise TypeError(f"{where}: {cls!r} is not a dataclass declared with @dataclass")
    params = cls.__dict__.get("__dataclass_params__")
    if params is None or not params.frozen:
        raise TypeError(f"{where}: {cls.__qualname__} is not frozen; declare it frozen=True")
    if cls in seen:
        return
    seen.add(cls)
    hints = typing.get_type_hints(cls)
    for field in dataclasses.fields(cls):
        _check_annotation(hints[field.name], f"{where}: {cls.__qualname__}.{field.name}", seen)


def _check_annotation(hint: Any, where: str, seen: set[type]) -> None:
    origin, args = get_origin(hint), get_args(hint)
    if hint in _SCALARS or hint is np.ndarray or origin is np.ndarray:
        return
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return
    if origin is Literal:
        return
    if origin in (typing.Union, types.UnionType):
        for arg in args:
            _check_annotation(arg, where, seen)
        return
    if origin in (list, tuple):
        for arg in args:
            if arg is not Ellipsis:
                _check_annotation(arg, where, seen)
        return
    if origin is dict and len(args) == 2:
        if args[0] not in (str, int):
            raise TypeError(f"{where}: a dict's keys must be str or int, got {args[0]!r}")
        _check_annotation(args[1], where, seen)
        return
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        _check_dataclass(hint, where, seen)
        return
    raise TypeError(
        f"{where} is typed {hint!r}, which is not data: use plain values, numpy arrays, "
        "enums, frozen dataclasses, and list, tuple, dict or optional of those"
    )


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
        payload: The request, an instance of the declared request type. Its
            arrays are its own: keep them past ``step()`` as they are.
        session_id: The id of the session this request belongs to, as the
            session's last reset named it, for the model's own log lines;
            ``None`` before the first reset.
    """

    slot: int
    generation: int
    payload: T
    session_id: str | None = None


class ModelServer:
    """Base class for the model half: weights loaded once, one batch stepped at a time.

    A subclass declares the types it exchanges and what it can serve, and
    writes :meth:`step`::

        class PlasmaModel(ModelServer):
            types = PLASMA            # a ModelServerTypes both halves import
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

    Class attributes:
        types: The :class:`ModelServerTypes` both halves import. Required.
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
        TypeError: When a subclass is defined without ``types``, or with a
            setting the manifest owns.
        ValueError: When a subclass is defined with ``max_batch`` or
            ``batch_window`` out of range.
    """

    types: ClassVar[ModelServerTypes]
    max_batch: ClassVar[int] = 1
    batch_window: ClassVar[float] = 0.0

    device: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check the subclass's declarations as it is defined."""
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("types", getattr(cls, "types", None))
        if not isinstance(declared, ModelServerTypes):
            raise TypeError(
                f"{cls.__name__}.types must be a ModelServerTypes naming the request, result "
                f"and signal types, got {declared!r}"
            )
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

    def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
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
            One output per request, in the order of *requests*: an instance of
            the declared result type, or an exception.
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

        *signal* is an instance of one of the declared signal types. It runs
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
        manifest = config or WorkerConfig(import_ref=f"{server.__module__}:{server.__name__}")
        return cls(
            max_batch=max_batch,
            batch_window=float(
                server.batch_window if manifest.batch_window is None else manifest.batch_window
            ),
            shm_input_bytes=manifest.shm_input_bytes or DEFAULT_SHM_INPUT_BYTES,
            shm_output_bytes=manifest.shm_output_bytes or DEFAULT_SHM_OUTPUT_BYTES,
        )


def _positive_int(where: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{where} must be a positive integer, got {value!r}")


def _seconds(where: str, value: Any) -> None:
    if type(value) not in (int, float) or not 0 <= value < math.inf:
        raise ValueError(f"{where} must be a number of seconds, at least 0, got {value!r}")
