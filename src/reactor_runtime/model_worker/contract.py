"""The worker half of a model that serves several sessions from one copy of its weights.

This is the application/model split of ``skills/application-model-isolation``
with the model half in a process of its own: a :class:`ModelWorker` subclass is
the model half, constructed once in the worker's process, and each session
process's :class:`~reactor_runtime.ReactorApp` is the application half, which
reaches it through a :class:`~reactor_runtime.model_worker.WorkerClient`. The
skill owns the rules of the split. Two of them shape this contract:

* The model half refuses what it cannot serve and leaves the recovery to the
  application: a request it cannot run fails (return an exception for it), a
  session it cannot open is refused, and the worker never retries, resets, or
  repairs a session on its own.
* Only the application calls
  :meth:`~reactor_runtime.model_worker.WorkerClient.reset`; neither the worker
  nor the client resets a slot by itself.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from reactor_runtime.core import WORKER_SETTINGS
from reactor_runtime.model_worker import schema


@dataclass(frozen=True)
class WorkerRequest[T]:
    """One session's request in a batch.

    Attributes:
        slot: The session process's slot, ``0`` to ``sessions - 1``. Stable for
            the life of the launcher: a session process restarted in a slot
            gets the same number.
        generation: Counts the times this slot was opened, starting at ``1``.
            It moves on when a new process connects in the slot and when the
            application calls :meth:`~reactor_runtime.model_worker.WorkerClient.reset`,
            so state kept per slot can tell a new session from the one before.
        payload: What the session half passed to ``step()``: an instance of
            :attr:`ModelWorker.request_type` built from its fields, or, for a
            worker that declares none, plain values and numpy arrays. Its
            arrays are read-only views of shared memory, valid until this
            request is answered: copy whatever is kept past ``step()``.
    """

    slot: int
    generation: int
    payload: T


class ModelWorker:
    """Base class for the worker half: weights loaded once, a batch stepped at a time.

    Not to be confused with :class:`~reactor_runtime.distributed.DistributedWorker`,
    one rank of a model split over several GPUs: a ``ModelWorker`` is one model
    half that serves many sessions, and its ``step()`` may drive such ranks.

    The worker's process calls :meth:`load` once, then accepts session
    processes and calls :meth:`step` with the requests waiting from them, one
    call at a time, on one thread. :meth:`session_opened` and
    :meth:`session_closed` run on that same thread between steps, so per-slot
    state needs no lock against ``step()``. :meth:`signal` runs on the thread
    reading that session's socket, and :meth:`health` on the health server's
    thread, both possibly while a step runs.

    The batching and join policy:

    * A session has at most one request waiting or running. A second one sent
      while the first waits replaces it, and the first is answered as expired.
    * Requests are taken in arrival order. When the worker is free and a request
      waits, it waits up to :attr:`batch_window` more seconds for others to
      arrive, stopping early once :attr:`max_batch` wait or every open session
      has one waiting, then steps up to :attr:`max_batch` of them together.
    * A session that connects, or a request that arrives, while a step runs
      joins the next step, never the running one. A session that leaves while
      its request runs has the answer dropped and is closed after the step.
    * A request whose timeout passed while it waited is answered as expired
      and never reaches ``step()``.

    The types the halves exchange are declared once, in the model's code, and
    imported by both halves (see :mod:`~reactor_runtime.model_worker.schema`
    for what a declaration may hold). Each is checked when the subclass is
    defined, again at load, and the session half's are checked at connect;
    the worker refuses a session whose declared types differ from its own.

    Class attributes:
        request_type: The frozen dataclass every request is built as, from the
            fields the session sent; a request that does not match it is
            rejected for that session alone. ``None`` (the default) takes plain
            values and numpy arrays.
        result_type: The frozen dataclass every output of :meth:`step` must
            be; an output that is not one fails that request. ``None`` (the
            default) sends plain values and numpy arrays.
        signals: The frozen dataclasses a session may send with
            :meth:`~reactor_runtime.model_worker.WorkerClient.signal`. Their
            fields are plain values (no arrays or bytes). A signal of another
            type is refused: the client raises before it sends it, and a
            session that sends one anyway is disconnected. None by default.
        max_batch: The most requests one :meth:`step` call receives, and the
            ceiling ``runtime.worker.max_batch`` in ``reactor.yaml`` may lower
            (batch sizes can be tied to captured graphs). ``1`` (the default)
            steps sessions one at a time in arrival order.
        batch_window: Seconds the worker waits after the first request for
            others to batch with it; ``runtime.worker.batch_window`` overrides
            it. ``0`` (the default) steps whatever is waiting at once.
        max_request_bytes: The most bytes one request may carry, its header and
            its arrays together; ``runtime.worker.max_request_bytes``
            overrides it. A larger request is refused for that session alone.
            256 MiB by default.
        shm_input_bytes: The size of the block a session's requests are
            written into, 1 MiB by default: a floor that
            ``runtime.worker.shm_input_bytes`` may only raise.
        shm_output_bytes: The size of the block its outputs are written into,
            8 MiB by default: a floor ``runtime.worker.shm_output_bytes`` may
            only raise. The worker makes and zero-fills both blocks when the
            session connects, before it is welcomed, so ``/dev/shm`` is charged
            to the worker and one session too many is refused at once. A
            request or output that outgrows its block still travels, in a
            larger block made for it.

    Attributes:
        device: ``"cuda"`` when torch sees a GPU in the worker's process,
            else ``"cpu"``. Set before :meth:`load` runs.
    """

    request_type: ClassVar[type | None] = None
    result_type: ClassVar[type | None] = None
    signals: ClassVar[tuple[type, ...]] = ()
    max_batch: ClassVar[int] = 1
    batch_window: ClassVar[float] = 0.0
    max_request_bytes: ClassVar[int] = 256 << 20
    shm_input_bytes: ClassVar[int] = 1 << 20
    shm_output_bytes: ClassVar[int] = 8 << 20

    device: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check the subclass's declarations as it is defined."""
        super().__init_subclass__(**kwargs)
        check_declarations(cls)

    @classmethod
    def shm_slot_size(cls) -> int:
        """Return the bytes of ``/dev/shm`` one session holds with the class's block sizes."""
        return cls.shm_input_bytes + cls.shm_output_bytes

    def load(self, config_path: Path | None) -> None:
        """Load the weights, once, before any session connects.

        Args:
            config_path: The file ``runtime.config`` in ``reactor.yaml`` names,
                resolved to an absolute path, or ``None``.

        A raise here fails the worker's start, and the launcher exits with it.
        """

    def step(self, requests: list[WorkerRequest[Any]]) -> list[Any]:
        """Run one step for every request and return one output per request, in order.

        An output that is an exception instance fails that request alone: the
        session that sent it sees a
        :class:`~reactor_runtime.model_worker.WorkerError` naming it. This is
        how a model fails one bad request without failing its batch-mates.

        A raise from ``step()`` fails every request of the batch: each session
        sees a :class:`~reactor_runtime.model_worker.BatchFailed` carrying the
        error, and its application decides what to do. Nothing is stepped
        again. The worker goes on serving, and its ``/health`` reports what
        :meth:`health` and :meth:`mark_unhealthy` say.

        An output, like a request, is an instance of :attr:`result_type`, or
        plain values, lists, tuples, dicts, bytes, and numpy arrays when no
        type is declared (see :mod:`reactor_runtime.model_worker.codec`); the
        arrays travel through shared memory. Nothing is pickled.

        Args:
            requests: Between one and the effective ``max_batch`` requests, at
                most one per session, in arrival order.

        Returns:
            One output per request, in the order of *requests*.
        """
        raise NotImplementedError(f"{type(self).__name__} must define step()")

    def session_opened(self, slot: int, generation: int) -> None:
        """Set up the state of the session that just opened *slot*.

        A raise refuses the session: its ``WorkerClient.connect()`` or
        ``reset()`` raises :class:`~reactor_runtime.model_worker.WorkerRejected`.
        """

    def session_closed(self, slot: int, generation: int) -> None:
        """Drop the state of the session that held *slot*.

        Runs when the session process disconnects or dies, and before a reset
        reopens the slot. For any slot it runs before the next
        :meth:`session_opened` of the same slot. A raise is logged and ignored.
        """

    def signal(self, slot: int, signal: Any) -> None:
        """Take an out-of-band signal a session sent, such as a request to stop early.

        *signal* is an instance of one of the types :attr:`signals` declares,
        built from the fields the session sent.

        Runs on that session's reader thread, possibly while :meth:`step` runs
        on the step thread, so it should only record the signal (set a flag
        under a lock) for ``step()`` to read. Signals from one session arrive in
        the order sent, and one sent before a request is taken before that
        request can join a step. A raise is logged and ignored.
        """

    def health(self) -> str | None:
        """Return ``None`` while the worker can serve, else why it cannot.

        The worker's ``GET /health`` calls it on every probe and answers 503
        ``unhealthy`` with the reason when it returns one, so a liveness probe
        restarts the worker. The default reports healthy. Override it with the
        model's own signals of trouble; :func:`cuda_health` is a ready-made
        check for a broken CUDA context.

        It runs on the health server's thread, concurrently with :meth:`step`:
        read state the step thread writes without blocking on it, and keep it
        quick (a probe has a timeout). A raise is reported as unhealthy, with
        the exception as the reason.
        """
        return None

    def mark_unhealthy(self, reason: str) -> None:
        """Latch the worker unhealthy: ``/health`` answers 503 with *reason* from now on.

        Call it from :meth:`step` (or any hook) on an error the model cannot
        recover from, such as a sticky CUDA error, so the deployment restarts
        the worker. It is never cleared; the first reason is kept. Safe from
        any thread.
        """
        with _UNHEALTHY_LOCK:
            if getattr(self, "_reactor_unhealthy", None) is None:
                self._reactor_unhealthy = str(reason) or "marked unhealthy"

    @property
    def unhealthy(self) -> str | None:
        """The reason :meth:`mark_unhealthy` latched, or ``None``."""
        return getattr(self, "_reactor_unhealthy", None)


_UNHEALTHY_LOCK = threading.Lock()


def cuda_health() -> str | None:
    """Return why CUDA cannot run work in this process, or ``None`` when it can.

    For a :meth:`ModelWorker.health` override: ``return cuda_health()``. It
    reads torch only when the process already imported it, so it adds no
    dependency, and reports healthy without torch. It runs a one-element
    kernel and synchronizes the device, which waits for the work already
    queued on it (a running step's kernels included): give the probe a
    timeout longer than a step.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return "the GPU is not visible"
        torch.ones(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:
        return f"CUDA error: {type(exc).__name__}: {exc}"
    return None


def check_declarations(cls: type[ModelWorker]) -> None:
    """Check a worker class's serving settings and its declared types.

    Raises:
        ValueError: A setting has the wrong type or range.
        TypeError: A declared type
            is not a data-only frozen dataclass (see
            :mod:`~reactor_runtime.model_worker.schema`).
    """
    _positive_int(cls.__name__, "max_batch", cls.max_batch)
    _positive_int(cls.__name__, "max_request_bytes", cls.max_request_bytes)
    _positive_int(cls.__name__, "shm_input_bytes", cls.shm_input_bytes)
    _positive_int(cls.__name__, "shm_output_bytes", cls.shm_output_bytes)
    _window(cls.__name__, "batch_window", cls.batch_window)
    for name in ("request_type", "result_type"):
        declared = getattr(cls, name)
        if declared is not None:
            try:
                schema.validate(declared)
            except schema.SchemaError as exc:
                raise TypeError(f"{cls.__name__}.{name}: {exc}") from None
    signals = cls.signals
    if not isinstance(signals, tuple):
        raise TypeError(
            f"{cls.__name__}.signals must be a tuple of frozen dataclasses, got {signals!r}"
        )
    check_signal_types(signals, f"{cls.__name__}.signals")


def check_signal_types(signals: tuple[Any, ...], where: str) -> None:
    """Check a tuple of signal types: distinct, data-only frozen dataclasses with no arrays.

    Raises:
        TypeError: One of them breaks a rule.
    """
    if len(set(signals)) != len(signals):
        raise TypeError(f"{where} names a type twice")
    names = [getattr(s, "__qualname__", repr(s)) for s in signals]
    if len(set(names)) != len(names):
        raise TypeError(f"{where} holds two types of one name")
    for declared in signals:
        try:
            buffers = schema.holds_buffers(declared)
        except schema.SchemaError as exc:
            raise TypeError(f"{where}: {exc}") from None
        if buffers:
            raise TypeError(
                f"{where}: {declared.__qualname__} holds arrays or bytes; a signal carries "
                "plain values only"
            )


@dataclass(frozen=True)
class WorkerSettings:
    """The serving settings a worker runs with: its class's, as ``reactor.yaml`` adjusted them.

    Attributes:
        max_batch: The most requests one step receives.
        batch_window: Seconds to wait for a batch to fill.
        max_request_bytes: The most bytes one request may carry.
        shm_input_bytes: Each session's request block.
        shm_output_bytes: Each session's output block.
    """

    max_batch: int
    batch_window: float
    max_request_bytes: int
    shm_input_bytes: int
    shm_output_bytes: int

    def check(self) -> None:
        """Check every value's type and range.

        Raises:
            ValueError: A value is not a positive integer (``batch_window``:
                a finite number of seconds, at least 0).
        """
        for key in WORKER_SETTINGS:
            value = getattr(self, key)
            if key == "batch_window":
                _window("settings", key, value)
            else:
                _positive_int("settings", key, value)

    @property
    def shm_slot_bytes(self) -> int:
        """The ``/dev/shm`` one session holds: its input and output blocks together."""
        return self.shm_input_bytes + self.shm_output_bytes

    @classmethod
    def resolve(cls, worker: type[ModelWorker], overrides: Mapping[str, Any]) -> WorkerSettings:
        """Return the settings *worker* serves with once *overrides* from ``runtime.worker`` apply.

        The class is the authority: an override may lower ``max_batch`` but
        not raise it, may raise ``shm_input_bytes`` and ``shm_output_bytes``
        but not lower them, and may set ``batch_window`` and
        ``max_request_bytes`` to any valid value.

        Raises:
            ValueError: An override is unknown, of the wrong type, or outside
                what the class allows.
        """
        unknown = sorted(set(overrides) - set(WORKER_SETTINGS))
        if unknown:
            raise ValueError(
                f"runtime.worker has unknown settings {unknown}; it takes import and "
                f"{', '.join(WORKER_SETTINGS)}"
            )
        name = worker.__name__
        values: dict[str, Any] = {k: getattr(worker, k) for k in WORKER_SETTINGS}
        for key, value in overrides.items():
            where = f"runtime.worker.{key}"
            if key == "batch_window":
                values[key] = _window("reactor.yaml", where, value)
                continue
            _positive_int("reactor.yaml", where, value)
            ceiling_or_floor = getattr(worker, key)
            if key == "max_batch" and value > ceiling_or_floor:
                raise ValueError(
                    f"{where} is {value}, above {name}.max_batch = {ceiling_or_floor}: the "
                    "class's max_batch is a ceiling the manifest may only lower"
                )
            if key.startswith("shm_") and value < ceiling_or_floor:
                raise ValueError(
                    f"{where} is {value}, below {name}.{key} = {ceiling_or_floor}: the class's "
                    "block sizes are a floor the manifest may only raise"
                )
            values[key] = value
        settings = cls(**values)
        settings.check()
        return settings


def _positive_int(owner: str, name: str, value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{owner}: {name} must be a positive integer, got {value!r}")
    return value


def _window(owner: str, name: str, value: Any) -> float:
    if type(value) not in (int, float) or not value >= 0 or value == float("inf"):
        raise ValueError(f"{owner}: {name} must be a number of seconds, at least 0, got {value!r}")
    return float(value)
