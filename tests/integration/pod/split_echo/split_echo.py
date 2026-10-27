"""The session half of the split echo: examples/echo with its engine in the worker.

``FAULT_STAGE=input`` or ``output`` with ``FAULT_SLOT`` makes that slot's
session raise once in ``process_input`` or ``process_output``.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from echo_model import EchoInput, EchoResult
from split_echo_worker import fault_due

from echo import Echo, EchoOutput
from reactor_runtime import ModelUnavailable, StepOutcome
from reactor_runtime.model_worker import WorkerClient

_SLOT = int(os.getenv("REACTOR_SESSION_SLOT", "0"))


class RemoteEngine:
    """The echo model's surface, answered by the worker."""

    def __init__(self, app: Echo) -> None:
        self.client = WorkerClient.connect(app)

    def generate(self, input: EchoInput) -> EchoResult:
        return self.client.step(input)

    def reset(self) -> None:
        with contextlib.suppress(ModelUnavailable):
            self.client.reset()


class SplitEcho(Echo):
    """examples/echo, its model half behind the shared worker."""

    def load(self, config_path: Path | None) -> None:
        super().load(config_path)
        self.engine = RemoteEngine(self)  # type: ignore[assignment]

    async def process_input(self) -> EchoInput:
        if fault_due("input", _SLOT):
            raise RuntimeError(f"injected input fault in slot {_SLOT}")
        return await super().process_input()

    async def process_output(self, outcome: StepOutcome) -> EchoOutput | None:
        if fault_due("output", _SLOT):
            raise RuntimeError(f"injected output fault in slot {_SLOT}")
        result: Any = await super().process_output(outcome)
        return result
