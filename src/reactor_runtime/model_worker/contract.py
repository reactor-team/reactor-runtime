"""The worker half of a model that serves several sessions from one copy of its weights.

A model split this way has two halves. The worker half, a :class:`ModelWorker`
subclass, is constructed once in its own process: it loads the weights, owns the
GPU, and answers :meth:`ModelWorker.step` with the requests that sessions sent.
The session half is an ordinary :class:`~reactor_runtime.ReactorApp`, one per
session process, whose ``generate()`` sends its input to the worker through a
:class:`~reactor_runtime.model_worker.WorkerClient` and returns the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class WorkerRequest:
    """One session's request in a batch.

    Attributes:
        slot: The session process's slot, ``0`` to ``sessions - 1``. Stable for
            the life of the launcher: a session process restarted in a slot
            gets the same number.
        generation: Counts the times this slot was opened, starting at ``1``.
            It moves on when a new process connects in the slot and when the
            session half calls :meth:`~reactor_runtime.model_worker.WorkerClient.reset`,
            so state kept per slot can tell a new session from the one before.
        payload: What the session half passed to ``step()``, rebuilt from plain
            values and numpy arrays. Its arrays are read-only views of shared
            memory, valid until this request is answered: copy whatever is
            kept past ``step()``.
    """

    slot: int
    generation: int
    payload: Any


class ModelWorker:
    """Base class for the worker half: weights loaded once, a batch stepped at a time.

    The worker's process calls :meth:`load` once, then accepts session
    processes and calls :meth:`step` with the requests waiting from them, one
    call at a time, on one thread. :meth:`session_opened` and
    :meth:`session_closed` run on that same thread between steps, so per-slot
    state needs no lock against ``step()``. :meth:`signal` is the exception: it
    runs on the thread reading that session's socket, while a step may be
    running.

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

    Class attributes:
        max_batch: The most requests one :meth:`step` call receives. ``1``
            (the default) steps sessions one at a time in arrival order.
        batch_window: Seconds the worker waits after the first request for
            others to batch with it. ``0`` (the default) steps whatever is
            waiting at once.
        max_request_bytes: The most bytes one request may carry, its header and
            its arrays together. A larger request is refused for that session
            alone. 256 MiB by default.
        signals: The names a session may pass to
            :meth:`~reactor_runtime.model_worker.WorkerClient.signal`. A signal
            of another name is dropped, for that session alone. None by default.
        shm_input_bytes: The size of the block a session's requests are
            written into, 1 MiB by default.
        shm_output_bytes: The size of the block its outputs are written into,
            8 MiB by default. The worker makes and zero-fills both blocks when
            the session connects, before it is welcomed, so ``/dev/shm`` is
            charged to the worker and one session too many is refused at once.
            A request or output that outgrows its block still travels, in a
            larger block made for it. :meth:`shm_slot_size` is the two together:
            what one session holds in ``/dev/shm``.

    Attributes:
        device: ``"cuda"`` when torch sees a GPU in the worker's process,
            else ``"cpu"``. Set before :meth:`load` runs.
    """

    max_batch: ClassVar[int] = 1
    batch_window: ClassVar[float] = 0.0
    max_request_bytes: ClassVar[int] = 256 << 20
    signals: ClassVar[frozenset[str]] = frozenset()
    shm_input_bytes: ClassVar[int] = 1 << 20
    shm_output_bytes: ClassVar[int] = 8 << 20

    device: str

    @classmethod
    def shm_slot_size(cls) -> int:
        """Return the bytes of ``/dev/shm`` one session holds: its input and output blocks.

        Size ``/dev/shm`` for the deployment from it: this times the number of
        sessions, plus room for a request or an output that outgrows its block.
        """
        return cls.shm_input_bytes + cls.shm_output_bytes

    def load(self, config_path: Path | None) -> None:
        """Load the weights, once, before any session connects.

        Args:
            config_path: The file ``runtime.config`` in ``reactor.yaml`` names,
                resolved to an absolute path, or ``None``.

        A raise here fails the worker's start, and the launcher exits with it.
        """

    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        """Run one step for every request and return one output per request, in order.

        An output that is an exception instance fails that request alone: the
        session that sent it sees a
        :class:`~reactor_runtime.model_worker.WorkerError` naming it.
        When a step of several requests raises, the worker calls ``step()``
        again for each of them alone, so the error reaches only the requests
        that raise on their own; ``step()`` therefore sees those requests a
        second time, and must leave a session's state as it was when it
        raises. A raise for one request fails that request. The worker goes on
        either way; its ``/health`` reports ``failing`` once
        ``REACTOR_MODEL_WORKER_FAILING_STEPS`` steps in a row (5 by default)
        have raised. An output, like a
        request, is plain values, lists, tuples, dicts, bytes, and numpy
        arrays (see :mod:`reactor_runtime.model_worker.codec`); the arrays
        travel through shared memory. Nothing is pickled.

        Args:
            requests: Between one and :attr:`max_batch` requests, at most one
                per session, in arrival order.

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

    def signal(self, slot: int, name: str, payload: Any) -> None:
        """Take an out-of-band signal a session sent, such as a request to stop early.

        Only the names :attr:`signals` declares reach it, with a JSON payload.

        Runs on that session's reader thread, possibly while :meth:`step` runs
        on the step thread, so it should only record the signal (set a flag
        under a lock) for ``step()`` to read. Signals from one session arrive in
        the order sent, and one sent before a request is taken before that
        request can join a step. A raise is logged and ignored.
        """
