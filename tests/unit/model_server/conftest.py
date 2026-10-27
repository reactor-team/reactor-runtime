"""A model worker served on a thread of the test process, and clients to reach it."""

from __future__ import annotations

import secrets
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from reactor_runtime.model_server import ModelClient, ModelServer, ServerSettings
from reactor_runtime.model_server.server import WorkerServer
from reactor_runtime.model_server.shm import unlink_prefix


class Served:
    """A running :class:`WorkerServer` and the clients a test opened on it."""

    def __init__(
        self,
        worker: ModelServer,
        socket_path: Path,
        prefix: str,
        settings: ServerSettings | None = None,
        max_sessions: int | None = None,
    ) -> None:
        self.worker = worker
        self.prefix = prefix
        self.path = socket_path
        self.server = WorkerServer(
            worker, socket_path, shm_prefix=prefix, settings=settings, max_sessions=max_sessions
        )
        self.server.bind()
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()
        self.clients: list[ModelClient] = []

    def client(self, slot: int | None, *, reset: bool = True, **kwargs: Any) -> ModelClient:
        """Connect a client in *slot*, and reset it as the application does, unless told not to."""
        kwargs.setdefault("connect_timeout", 5.0)
        client = ModelClient.connect(socket_path=self.path, slot=slot, **kwargs)
        self.clients.append(client)
        if reset:
            client.reset()
        return client

    def halt(self) -> None:
        """Stop the server, as a worker that goes away, keeping its clients open."""
        self.server.stop()
        self.thread.join(timeout=10)

    def stop(self) -> None:
        for client in self.clients:
            client.close()
        self.server.stop()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "the worker's step thread did not stop"


@pytest.fixture
def serve() -> Iterator[Callable[[ModelServer], Served]]:
    """Return a function that serves a worker; everything it served stops after the test."""
    # A Unix socket path is limited to about 100 bytes, which pytest's tmp_path can pass.
    short = Path(tempfile.mkdtemp(prefix="rrw"))
    served: list[Served] = []
    prefix = f"rrt{secrets.token_hex(3)}_"

    def start(
        worker: ModelServer,
        path: Path | None = None,
        settings: ServerSettings | None = None,
        max_sessions: int | None = None,
    ) -> Served:
        if not hasattr(worker, "device"):
            worker.device = "cpu"
        worker.load(None)
        one = Served(worker, path or short / f"w{len(served)}.sock", prefix, settings, max_sessions)
        served.append(one)
        return one

    yield start
    for one in served:
        one.stop()
    unlink_prefix(prefix)
    shutil.rmtree(short, ignore_errors=True)
