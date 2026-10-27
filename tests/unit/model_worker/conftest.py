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

from reactor_runtime.model_worker import ModelWorker, WorkerClient
from reactor_runtime.model_worker.server import WorkerServer
from reactor_runtime.model_worker.shm import unlink_prefix


class Served:
    """A running :class:`WorkerServer` and the clients a test opened on it."""

    def __init__(self, worker: ModelWorker, socket_path: Path, prefix: str) -> None:
        self.worker = worker
        self.prefix = prefix
        self.path = socket_path
        self.server = WorkerServer(worker, socket_path, shm_prefix=prefix)
        self.server.bind()
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()
        self.clients: list[WorkerClient] = []

    def client(self, slot: int, **kwargs: Any) -> WorkerClient:
        client = WorkerClient.connect(
            socket_path=self.path, slot=slot, connect_timeout=5.0, **kwargs
        )
        self.clients.append(client)
        return client

    def stop(self) -> None:
        for client in self.clients:
            client.close()
        self.server.stop()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "the worker's step thread did not stop"


@pytest.fixture
def serve() -> Iterator[Callable[[ModelWorker], Served]]:
    """Return a function that serves a worker; everything it served stops after the test."""
    # A Unix socket path is limited to about 100 bytes, which pytest's tmp_path can pass.
    short = Path(tempfile.mkdtemp(prefix="rrw"))
    served: list[Served] = []
    prefix = f"rrt{secrets.token_hex(3)}_"

    def start(worker: ModelWorker, path: Path | None = None) -> Served:
        if not hasattr(worker, "device"):
            worker.device = "cpu"
        worker.load(None)
        one = Served(worker, path or short / f"w{len(served)}.sock", prefix)
        served.append(one)
        return one

    yield start
    for one in served:
        one.stop()
    unlink_prefix(prefix)
    shutil.rmtree(short, ignore_errors=True)
