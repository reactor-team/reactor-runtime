"""The worker half: examples/echo's model, stepped for the sessions that connect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from echo_model import EchoInput, EchoModel

from reactor_runtime.model_worker import ModelWorker, WorkerRequest


class EchoWorker(ModelWorker):
    """Apply the effect for each request, one at a time."""

    def load(self, config_path: Path | None) -> None:
        """Construct the echo model."""
        self.model = EchoModel()
        self.model.load()

    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        """Run the echo model for every request."""
        return [
            self.model.generate(
                EchoInput(
                    frame=r.payload["frame"],
                    effect=r.payload["effect"],
                    intensity=r.payload["intensity"],
                    caption=r.payload["caption"],
                )
            ).frame
            for r in requests
        ]
