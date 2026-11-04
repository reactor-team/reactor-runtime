"""The model half: examples/echo's model, stepped for the sessions that connect."""

from __future__ import annotations

from pathlib import Path

from echo_model import EchoInput, EchoModel, EchoResult

from reactor_runtime.model_server import ModelServer, ServerRequest


class EchoServer(ModelServer[EchoInput, EchoResult]):
    """Apply the effect for each request, one at a time."""

    def load(self, config_path: Path | None) -> None:
        """Construct the echo model."""
        self.model = EchoModel()
        self.model.load()

    def step(self, requests: list[ServerRequest[EchoInput]]) -> list[EchoResult]:
        """Run the echo model for every request."""
        return [self.model.generate(r.payload) for r in requests]
