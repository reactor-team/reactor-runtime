"""The shared-worker example: each session's frames are its own, batched or alone.

The worker half is served on a thread of the test process and every session is
a real ``WorkerClient``, so the frames cross the socket and shared memory as
they do between the launcher's processes.
"""

from __future__ import annotations

import secrets
import shutil
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from plasma_worker import HEIGHT, WIDTH, PlasmaWorker

from reactor_runtime.manifest import load_config, load_worker_ref
from reactor_runtime.model_worker import WorkerClient, WorkerRequest
from reactor_runtime.model_worker.server import WorkerServer
from reactor_runtime.model_worker.shm import unlink_prefix

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "shared_worker"


class CountingPlasma(PlasmaWorker):
    """The example's worker, recording the size of every step it runs."""

    batch_window = 0.05

    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        self.sizes.append(len(requests))
        return super().step(requests)

    def load(self, config_path: Path | None) -> None:
        super().load(config_path)
        self.sizes: list[int] = []


@pytest.fixture
def worker() -> Iterator[tuple[CountingPlasma, Path, str]]:
    directory = Path(tempfile.mkdtemp(prefix="rrx"))
    prefix = f"rrx{secrets.token_hex(3)}_"
    model = CountingPlasma()
    model.device = "cpu"
    model.load(None)
    server = WorkerServer(model, directory / "w.sock", shm_prefix=prefix)
    server.bind()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    yield model, directory / "w.sock", prefix
    server.stop()
    thread.join(5)
    unlink_prefix(prefix)
    shutil.rmtree(directory, ignore_errors=True)


def _payloads(slot: int) -> list[dict[str, int]]:
    return [{"tick": tick * (slot + 1), "hue": 40 * slot} for tick in range(6)]


def test_the_manifest_names_both_halves() -> None:
    assert load_worker_ref(_EXAMPLE_DIR / "reactor.yaml") == "plasma_worker:PlasmaWorker"
    assert load_config(_EXAMPLE_DIR / "reactor.yaml").model_ref == "plasma:Plasma"


def test_each_session_gets_the_same_frames_batched_as_alone(
    worker: tuple[CountingPlasma, Path, str],
) -> None:
    model, path, _ = worker
    clients = [WorkerClient.connect(socket_path=path, slot=slot) for slot in range(4)]
    try:
        alone = {slot: [clients[slot].step(p) for p in _payloads(slot)] for slot in range(4)}
        assert set(model.sizes) == {1}
        model.sizes.clear()

        batched: dict[int, list[np.ndarray]] = {slot: [] for slot in range(4)}
        barrier = threading.Barrier(4)

        def run(slot: int) -> None:
            for payload in _payloads(slot):
                barrier.wait(5)
                batched[slot].append(clients[slot].step(payload))

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
