"""The model half of the split echo: examples/echo's model, stepped for every session."""

from __future__ import annotations

from pathlib import Path

from echo_model import EchoInput, EchoModel, EchoResult
from faults import fault_due

from reactor_runtime.model_server import ModelServer, ServerRequest


class SplitEchoModel(ModelServer[EchoInput, EchoResult]):
    """Apply each session's effect to its frame, up to four sessions a step."""

    max_batch = 4
    batch_window = 0.002

    def load(self, config_path: Path | None) -> None:
        """Construct the shared model."""
        self.model = EchoModel()
        self.model.load()

    def step(self, requests: list[ServerRequest[EchoInput]]) -> list[EchoResult | Exception]:
        """Run the effect for every request; an injected fault fails its request alone."""
        outputs: list[EchoResult | Exception] = []
        for request in requests:
            if fault_due("model", request.slot):
                outputs.append(RuntimeError(f"injected model fault in slot {request.slot}"))
            else:
                outputs.append(self.model.generate(request.payload))
        return outputs
