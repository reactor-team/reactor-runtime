"""The application half of the split echo: examples/echo with its engine in the model server.

A lost or stuck model server ends this runtime: mid-session, echo's own
``process_output()`` re-raises the step's error; between sessions, the
client's ``on_lost`` calls ``self.crash()``.

``FAULT_STAGE=input`` or ``output`` with ``FAULT_SLOT`` makes that slot's
session raise once in ``process_input`` or ``process_output``.
"""

from __future__ import annotations

import os
from pathlib import Path

from echo_model import EchoInput, EchoResult
from faults import fault_due

from echo import Echo, EchoOutput
from reactor_runtime import StepOutcome, session_started
from reactor_runtime.model_server import ModelClient

_SLOT = int(os.getenv("REACTOR_SESSION_SLOT", "0"))


class RemoteEngine:
    """The echo model's surface, answered by the model server."""

    def __init__(self, client: ModelClient[EchoInput, EchoResult]) -> None:
        self.client = client

    def generate(self, input: EchoInput) -> EchoResult:
        return self.client.step(input)

    def reset(self) -> None:
        """Nothing here: the application resets its slot when the next session starts."""


class SplitEcho(Echo):
    """examples/echo, its model half behind the shared model server."""

    def load(self, config_path: Path | None) -> None:
        super().load(config_path)
        self.remote = RemoteEngine(ModelClient.connect(on_lost=self.crash))
        self.engine = self.remote

    @session_started
    def on_session_started(self) -> None:
        self.remote.client.reset()

    async def process_input(self) -> EchoInput:
        if fault_due("input", _SLOT):
            raise RuntimeError(f"injected input fault in slot {_SLOT}")
        return await super().process_input()

    async def process_output(self, outcome: StepOutcome) -> EchoOutput | None:
        if fault_due("output", _SLOT):
            raise RuntimeError(f"injected output fault in slot {_SLOT}")
        return await super().process_output(outcome)
