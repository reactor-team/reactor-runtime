"""The model half: examples/echo's model, stepped for the sessions that connect.

``ECHO_ANSWER_MODE`` picks how a step answers: ``immediate`` (the default)
returns each result from ``step()``; ``future`` hands each request to a
completion thread and returns a future it completes.
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future
from pathlib import Path

from echo_model import EchoInput, EchoModel, EchoResult

from reactor_runtime.model_server import ModelServer, ServerRequest


class EchoServer(ModelServer[EchoInput, EchoResult]):
    """Apply the effect for each request, in ``step()`` or on a completion thread."""

    def load(self, config_path: Path | None) -> None:
        """Construct the echo model, and its completion thread in future mode."""
        self.model = EchoModel()
        self.model.load()
        self.mode = os.environ.get("ECHO_ANSWER_MODE", "immediate")
        if self.mode not in ("immediate", "future"):
            raise ValueError(f"ECHO_ANSWER_MODE must be immediate or future, got {self.mode!r}")
        self._work: queue.Queue[tuple[EchoInput, Future[EchoResult]]] = queue.Queue()
        if self.mode == "future":
            threading.Thread(target=self._complete, name="echo-completion", daemon=True).start()

    def step(
        self, requests: list[ServerRequest[EchoInput]]
    ) -> list[EchoResult] | list[Future[EchoResult]]:
        """Run the echo model for every request, or queue each for the completion thread."""
        if self.mode == "immediate":
            return [self.model.generate(r.payload) for r in requests]
        futures: list[Future[EchoResult]] = []
        for request in requests:
            future: Future[EchoResult] = Future()
            self._work.put((request.payload, future))
            futures.append(future)
        return futures

    def _complete(self) -> None:
        while True:
            payload, future = self._work.get()
            # A request cancelled before its work starts is skipped.
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(self.model.generate(payload))
                except Exception as exc:
                    future.set_exception(exc)
            del payload, future
