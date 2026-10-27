"""The shared-worker example: each session's frames are its own, batched or alone.

The worker half is served on a thread of the test process and every session is
a real ``ModelClient``, so the frames cross the socket and shared memory as
they do between processes. The application half's option 2 is checked against
a worker that stops and comes back.
"""

from __future__ import annotations

import asyncio
import secrets
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from plasma import Plasma, PlasmaState
from plasma_contract import HEIGHT, WIDTH, Frame, Tick
from plasma_worker import PlasmaWorker

from reactor_runtime import ApplicationError
from reactor_runtime.manifest import load_config
from reactor_runtime.model_server import ModelClient, ServerRequest
from reactor_runtime.model_server.server import WorkerServer
from reactor_runtime.model_server.shm import unlink_prefix

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "shared_worker"


class CountingPlasma(PlasmaWorker):
    """The example's worker, recording the size of every step it runs."""

    batch_window = 0.05

    def step(self, requests: list[ServerRequest[Tick]]) -> list[Frame]:
        self.sizes.append(len(requests))
        return super().step(requests)

    def load(self, config_path: Path | None) -> None:
        super().load(config_path)
        self.sizes: list[int] = []


class Worker:
    """The example's worker served on a thread, which a test can stop and start again."""

    def __init__(self, directory: Path, prefix: str) -> None:
        self.path = directory / "w.sock"
        self.prefix = prefix
        self.model = CountingPlasma()
        self.model.device = "cpu"
        self.model.load(None)
        self.server: WorkerServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.server = WorkerServer(self.model, self.path, shm_prefix=self.prefix)
        self.server.bind()
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server is not None and self.thread is not None:
            self.server.stop()
            self.thread.join(5)
            self.server = None


@pytest.fixture
def worker() -> Iterator[Worker]:
    directory = Path(tempfile.mkdtemp(prefix="rrx"))
    one = Worker(directory, f"rrx{secrets.token_hex(3)}_")
    one.start()
    yield one
    one.stop()
    unlink_prefix(one.prefix)
    shutil.rmtree(directory, ignore_errors=True)


def _payloads(slot: int) -> list[Tick]:
    return [Tick(tick=tick * (slot + 1), hue=40 * slot) for tick in range(6)]


def _connect(path: Path, slot: int) -> ModelClient:
    client = ModelClient.connect(
        socket_path=path, slot=slot, request_type=Tick, result_type=Frame, connect_timeout=5
    )
    client.reset()
    return client


def _until(check: Callable[[], bool], seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_the_manifest_names_both_halves_and_the_serving_settings() -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert cfg.model_ref == "plasma:Plasma"
    assert cfg.worker is not None
    assert cfg.worker.import_ref == "plasma_worker:PlasmaWorker"
    assert dict(cfg.worker.settings) == {"max_batch": 4}


def test_each_session_gets_the_same_frames_batched_as_alone(worker: Worker) -> None:
    model, path = worker.model, worker.path
    clients = [_connect(path, slot) for slot in range(4)]
    try:
        alone = {slot: [clients[slot].step(p).pixels for p in _payloads(slot)] for slot in range(4)}
        assert set(model.sizes) == {1}
        model.sizes.clear()

        batched: dict[int, list[np.ndarray]] = {slot: [] for slot in range(4)}
        barrier = threading.Barrier(4)

        def run(slot: int) -> None:
            for payload in _payloads(slot):
                barrier.wait(5)
                batched[slot].append(clients[slot].step(payload).pixels)

        threads = [threading.Thread(target=run, args=(slot,)) for slot in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert max(model.sizes) > 1, "the sessions were never batched together"
        for slot in range(4):
            for got, want in zip(batched[slot], alone[slot], strict=True):
                assert got.shape == (HEIGHT, WIDTH, 3)
                assert np.array_equal(got, want)
        # Different sessions, different frames: nothing crossed between them.
        assert not np.array_equal(batched[0][3], batched[1][3])
    finally:
        for client in clients:
            client.close()


def test_the_application_half_ends_the_session_and_recovers_when_the_worker_returns(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REACTOR_MODEL_WORKER_SOCKET", str(worker.path))
    monkeypatch.setenv("REACTOR_SESSION_SLOT", "0")
    app = Plasma()
    reports: list[str | None] = []
    monkeypatch.setattr(app, "report_unavailable", lambda reason: _report(app, reports, reason))
    monkeypatch.setattr(app, "report_available", lambda: _report(app, reports, None))
    app.load(None)
    try:
        app.on_session_started()
        assert np.asarray(app.generate(PlasmaState()).main_video).shape == (HEIGHT, WIDTH, 3)

        worker.stop()
        _until(lambda: bool(reports))
        assert reports[0] is not None
        assert reports[0].startswith("worker_died:")
        with pytest.raises(ApplicationError):
            asyncio.run(app.process_input())

        worker.start()
        _until(lambda: reports[-1] is None)
        assert app.unavailable is None
        app.on_session_started()
        assert np.asarray(app.generate(PlasmaState()).main_video).shape == (HEIGHT, WIDTH, 3)
    finally:
        app.worker.close()


def _report(app: Plasma, reports: list[str | None], reason: str | None) -> None:
    reports.append(reason)
    app._reactor_unavailable = reason
