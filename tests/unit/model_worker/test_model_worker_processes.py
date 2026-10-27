"""The worker and its sessions across real processes: death on either side stays contained."""

from __future__ import annotations

import multiprocessing
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from reactor_runtime import ModelUnavailable
from reactor_runtime.model_worker import WorkerClient, WorkerCrashed
from reactor_runtime.model_worker.shm import unlink_prefix

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="process tests on Linux")

_WORKER = textwrap.dedent(
    """
    import os, time
    from reactor_runtime.model_worker import ModelWorker

    class Echo(ModelWorker):
        max_batch = 2

        def step(self, requests):
            out = []
            for r in requests:
                if isinstance(r.payload, dict) and "sleep" in r.payload:
                    time.sleep(r.payload["sleep"])
                    out.append(r.slot)
                else:
                    out.append(r.payload + r.slot)
            return out

        def session_closed(self, slot, generation):
            with open(os.environ["CLOSED_LOG"], "a") as f:
                f.write(f"{slot}\\n")
    """
)


class Worker:
    def __init__(self, directory: Path, prefix: str) -> None:
        self.socket = directory / "w.sock"
        self.prefix = prefix
        self.closed_log = directory / "closed.log"
        (directory / "echo_worker.py").write_text(_WORKER)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "reactor_runtime.model_worker",
                "--worker",
                "echo_worker:Echo",
                "--socket",
                str(self.socket),
            ],
            cwd=directory,
            env={**os.environ, "CLOSED_LOG": str(self.closed_log), "REACTOR_SHM_PREFIX": prefix},
        )
        ready = directory / "ready"
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert self.process.poll() is None, "the worker exited while starting"
            assert time.monotonic() < deadline, "the worker never became ready"
            time.sleep(0.05)

    def client(self, slot: int) -> WorkerClient:
        return WorkerClient.connect(socket_path=self.socket, slot=slot)

    def closed(self) -> list[int]:
        if not self.closed_log.exists():
            return []
        return [int(line) for line in self.closed_log.read_text().split()]


@pytest.fixture
def worker() -> Iterator[Worker]:
    directory = Path(tempfile.mkdtemp(prefix="rrp"))
    prefix = f"rrp{secrets.token_hex(3)}_"
    one = Worker(directory, prefix)
    yield one
    if one.process.poll() is None:
        one.process.terminate()
        one.process.wait(10)
    unlink_prefix(prefix)
    shutil.rmtree(directory, ignore_errors=True)


def test_a_worker_killed_mid_step_ends_the_waiting_session_at_once(worker: Worker) -> None:
    waiting, idle = worker.client(0), worker.client(1)
    errors: list[BaseException] = []

    def call() -> None:
        try:
            waiting.step({"sleep": 30})
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    time.sleep(0.3)
    started = time.monotonic()
    worker.process.send_signal(signal.SIGKILL)
    thread.join(10)
    assert time.monotonic() - started < 5
    assert isinstance(errors[0], WorkerCrashed)
    assert str(errors[0]).startswith("worker_died: ")
    with pytest.raises(WorkerCrashed, match=r"^worker_died: "):
        idle.step(1)
    waiting.close()
    idle.close()


def _session_main(socket: str, prefix: str, slot: int, started: object) -> None:
    client = WorkerClient.connect(socket_path=socket, slot=slot)
    started.set()  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    client.step({"sleep": 1.0})


def test_a_session_killed_mid_step_is_closed_and_the_others_go_on(worker: Worker) -> None:
    ctx = multiprocessing.get_context("spawn")
    started = ctx.Event()
    victim = ctx.Process(target=_session_main, args=(str(worker.socket), worker.prefix, 0, started))
    victim.start()
    assert started.wait(30)
    time.sleep(0.2)
    assert victim.pid is not None
    os.kill(victim.pid, signal.SIGKILL)
    victim.join(10)
    other = worker.client(1)
    assert other.step(41) == 42
    deadline = time.monotonic() + 10
    while 0 not in worker.closed() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert worker.closed() == [0]
    # The slot opens again for a restarted session process.
    again = worker.client(0)
    assert again.step(np.int64(5)) == 5
    other.close()
    again.close()


def test_sigterm_fails_queued_requests_and_frees_every_block(worker: Worker) -> None:
    running, queued, third = worker.client(0), worker.client(1), worker.client(2)
    results: dict[str, object] = {}

    def call(name: str, client: WorkerClient, payload: object) -> None:
        try:
            results[name] = client.step(payload)
        except Exception as exc:
            results[name] = exc

    threads = [threading.Thread(target=call, args=("running", running, {"sleep": 1.0}))]
    threads[0].start()
    time.sleep(0.3)
    # Both arrive while the first step runs, so they wait for the next one.
    threads.append(threading.Thread(target=call, args=("queued", queued, 5)))
    threads.append(threading.Thread(target=call, args=("queued2", third, 6)))
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.2)
    worker.process.send_signal(signal.SIGTERM)
    assert worker.process.wait(10) == 0
    for thread in threads:
        thread.join(10)
    assert results["running"] == 0
    for name in ("queued", "queued2"):
        error = results[name]
        assert isinstance(error, ModelUnavailable), error
        assert str(error) == "worker_died: the model worker is stopping"
    assert not list(Path("/dev/shm").glob(f"{worker.prefix}*"))
    for client in (running, queued, third):
        client.close()


def test_sigterm_stops_the_worker_promptly(worker: Worker) -> None:
    client = worker.client(0)
    assert client.step(1) == 1
    started = time.monotonic()
    worker.process.send_signal(signal.SIGTERM)
    assert worker.process.wait(10) == 0
    assert time.monotonic() - started < 3
    assert not worker.socket.exists()
    assert not (worker.socket.parent / "ready").exists()
    with pytest.raises(WorkerCrashed):
        client.step(1)
    client.close()
