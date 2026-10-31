"""The application half: the tracked echo, its engine answered by the model server."""

from __future__ import annotations

from pathlib import Path

from echo_model import EchoInput, EchoResult
from tracked_echo import TrackedEcho

from reactor_runtime import session_started
from reactor_runtime.model_server import ModelClient


class _RemoteEngine:
    """The echo model's three calls, answered by the model server."""

    def __init__(self, client: ModelClient[EchoInput, EchoResult]) -> None:
        self._client = client

    def load(self) -> None:
        """Nothing to load: the model server holds the model."""

    def generate(self, input: EchoInput) -> EchoResult:
        """Send the step to the model server; the frame travels through shared memory."""
        return self._client.step(input)

    def reset(self) -> None:
        """Start the slot over on the model server."""
        self._client.reset()


class SharedWorkerEcho(TrackedEcho):
    """The tracked echo with ``EchoModel`` in the shared model server."""

    def load(self, config_path: Path | None) -> None:
        """Load as echo does, then answer its engine's calls from the model server."""
        super().load(config_path)
        self.engine = _RemoteEngine(ModelClient.connect(on_lost=self.crash))

    @session_started
    def on_session_started(self) -> None:
        """Reset this slot on the model server, so the session starts on fresh state."""
        self.engine.reset()
