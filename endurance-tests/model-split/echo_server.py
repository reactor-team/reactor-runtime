"""The model half: answer echo requests from a pipeline thread."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from echo_model import EchoInput, EchoModel, EchoResult

from reactor_runtime.model_server import DEFERRED, Deferred, ModelServer, ServerRequest


class EchoServer(ModelServer[EchoInput, EchoResult]):
    """Apply each effect on a completion thread and defer its answer."""

    def load(self, config_path: Path | None) -> None:
        """Construct the echo model and start its completion thread."""
        self.model = EchoModel()
        self.model.load()
        self._work: queue.Queue[ServerRequest[EchoInput]] = queue.Queue()
        self._lock = threading.Lock()
        self._generations: dict[int, int] = {}
        threading.Thread(target=self._complete, name="echo-pipeline", daemon=True).start()

    def step(self, requests: list[ServerRequest[EchoInput]]) -> list[Deferred]:
        """Queue each request for the completion thread and return immediately."""
        for request in requests:
            self._work.put(request)
        return [DEFERRED] * len(requests)

    def session_opened(self, slot: int, generation: int) -> None:
        """Allow work for this generation on the completion thread."""
        with self._lock:
            self._generations[slot] = generation

    def session_closed(self, slot: int, generation: int) -> None:
        """Discard this generation's pending work between pipeline stages."""
        with self._lock:
            self._generations.pop(slot, None)

    def _complete(self) -> None:
        while True:
            request = self._work.get()
            with self._lock:
                live = self._generations.get(request.slot) == request.generation
            if live:
                try:
                    output = self.model.generate(request.payload)
                except Exception as exc:
                    self.answer(request, exc)
                else:
                    self.answer(request, output)
                    del output
            self._work.task_done()
            del request
