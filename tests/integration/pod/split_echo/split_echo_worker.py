"""The worker half of the split echo: examples/echo's model, stepped for every session.

``FAULT_STAGE=worker`` with ``FAULT_SLOT`` fails that slot's next request once,
the first time ``FAULT_MARK`` does not exist yet.
"""

from __future__ import annotations

import os
from pathlib import Path

from echo_model import EchoInput, EchoModel, EchoResult

from reactor_runtime.model_worker import ModelWorker, WorkerRequest


def fault_due(stage: str, slot: int) -> bool:
    """Whether the injected fault for *stage* fires now for *slot*; it fires once."""
    if os.getenv("FAULT_STAGE") != stage or os.getenv("FAULT_SLOT") != str(slot):
        return False
    mark = Path(os.environ["FAULT_MARK"])
    if mark.exists():
        return False
    mark.write_text(stage)
    return True


class EchoWorker(ModelWorker):
    """Apply each session's effect to its frame, up to four sessions a step."""

    request_type = EchoInput
    result_type = EchoResult
    max_batch = 4
    batch_window = 0.002

    def load(self, config_path: Path | None) -> None:
        """Construct the shared model."""
        self.model = EchoModel()
        self.model.load()

    def step(self, requests: list[WorkerRequest[EchoInput]]) -> list[EchoResult | Exception]:
        """Run the effect for every request."""
        outputs: list[EchoResult | Exception] = []
        for request in requests:
            if fault_due("worker", request.slot):
                outputs.append(RuntimeError(f"injected worker fault in slot {request.slot}"))
            else:
                outputs.append(self.model.generate(request.payload))
        return outputs
