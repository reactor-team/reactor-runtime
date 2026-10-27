"""The application half of the split echo: examples/echo with its engine in the worker.

It takes option 2 of :mod:`reactor_runtime.model_worker` for a lost or stuck
worker: the session ends with the reason, the process stays up and refuses
sessions, and one watcher thread reconnects. Any other model error is re-raised,
as in examples/echo.

``FAULT_STAGE=input`` or ``output`` with ``FAULT_SLOT`` makes that slot's
session raise once in ``process_input`` or ``process_output``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from echo_model import EchoInput, EchoResult
from split_echo_worker import fault_due

from echo import Echo, EchoOutput
from reactor_runtime import ApplicationError, StepOutcome, session_started
from reactor_runtime.model_worker import (
    ClientClosed,
    StepUnanswered,
    WorkerClient,
    WorkerLost,
    WorkerRejected,
    WorkerTimeout,
    WorkerUnreachable,
)

_SLOT = int(os.getenv("REACTOR_SESSION_SLOT", "0"))


class RemoteEngine:
    """The echo model's surface, answered by the worker."""

    def __init__(self) -> None:
        self.client = WorkerClient.connect(request_type=EchoInput, result_type=EchoResult)

    def generate(self, input: EchoInput) -> EchoResult:
        return self.client.step(input)

    def reset(self) -> None:
        """Nothing here: the application resets its slot when the next session starts."""


class SplitEcho(Echo):
    """examples/echo, its model half behind the shared worker."""

    def load(self, config_path: Path | None) -> None:
        super().load(config_path)
        self.remote = RemoteEngine()
        self.engine = self.remote  # type: ignore[assignment]
        self._give_up = threading.Event()
        threading.Thread(target=self._watch, name="echo-watch", daemon=True).start()

    @session_started
    def on_session_started(self) -> None:
        try:
            self.remote.client.reset()
        except (WorkerLost, StepUnanswered, WorkerRejected) as exc:
            self._lose_worker(f"worker_unavailable: the reset failed: {exc}")

    async def process_input(self) -> EchoInput:
        if self.unavailable is not None:
            raise ApplicationError(self.unavailable)
        if fault_due("input", _SLOT):
            raise RuntimeError(f"injected input fault in slot {_SLOT}")
        return await super().process_input()

    async def process_output(self, outcome: StepOutcome) -> EchoOutput | None:
        if isinstance(outcome.error, (WorkerLost, StepUnanswered)):
            self._lose_worker(str(outcome.error))
            return None
        if fault_due("output", _SLOT):
            raise RuntimeError(f"injected output fault in slot {_SLOT}")
        result: Any = await super().process_output(outcome)
        return result

    def _lose_worker(self, reason: str) -> None:
        self.report_unavailable(reason)
        self._give_up.set()

    def _watch(self) -> None:
        client = self.remote.client
        while True:
            try:
                reason = client.wait_lost(timeout=1.0)
            except TimeoutError:
                if not self._give_up.is_set():
                    continue
                reason = self.unavailable or "worker_unavailable"
            except ClientClosed:
                return
            self.report_unavailable(reason)
            while True:
                try:
                    client.reconnect(timeout=5.0)
                    break
                except (WorkerTimeout, WorkerRejected, WorkerUnreachable):
                    continue
                except ClientClosed:
                    return
            self._give_up.clear()
            self.report_available()
