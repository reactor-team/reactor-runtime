"""The session half: the tracked echo, its engine answered by the model worker."""

from __future__ import annotations

from pathlib import Path

from echo_model import EchoInput, EchoResult
from tracked_echo import TrackedEcho

from reactor_runtime.model_worker import WorkerClient


class _RemoteEngine:
    """The echo model's three calls, answered by the worker."""

    def __init__(self, client: WorkerClient) -> None:
        self._client = client

    def load(self) -> None:
        """Nothing to load: the worker holds the model."""

    def generate(self, input: EchoInput) -> EchoResult:
        """Send the step to the worker and wrap its frame."""
        sent = {
            "frame": input.frame,
            "effect": input.effect,
            "intensity": input.intensity,
            "caption": input.caption,
        }
        return EchoResult(frame=self._client.step(sent))

    def reset(self) -> None:
        """Start the slot over on the worker."""
        self._client.reset()


class SharedWorkerEcho(TrackedEcho):
    """The tracked echo with ``EchoModel`` in the shared model worker."""

    def load(self, config_path: Path | None) -> None:
        """Load as echo does, then answer its engine's calls from the worker."""
        super().load(config_path)
        self.engine = _RemoteEngine(WorkerClient.connect(self))
