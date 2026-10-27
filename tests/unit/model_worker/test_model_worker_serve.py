"""``reactor-runtime serve-worker``: what it reads, what it reports, and how it stops."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

from reactor_runtime import cli
from reactor_runtime.model_worker import WorkerClient

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="process tests on Linux")

_WORKER = """
import time
from reactor_runtime.model_worker import ModelWorker

class Slow(ModelWorker):
    def load(self, config_path):
        self.config = config_path
        time.sleep(float(__import__("os").environ.get("LOAD_SECONDS", "0")))

    def step(self, requests):
        if requests[0].payload == "hang":
            time.sleep(5)
        return [str(self.config) for _ in requests]
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _health(port: int) -> tuple[int | None, dict[str, object]]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    except OSError:
        return None, {}


def _start(directory: Path, port: int, **env: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "reactor_runtime.cli", "serve-worker"],
        cwd=directory,
        env={
            **os.environ,
            "REACTOR_MODEL_WORKER_SOCKET": str(directory / "run" / "sock"),
            "REACTOR_MODEL_WORKER_HEALTH_PORT": str(port),
            "REACTOR_SHM_PREFIX": f"rrs{port}",
            "HOST": "127.0.0.1",
            **env,
        },
    )


@pytest.fixture
def model_dir() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="rrs"))
    (directory / "slow_worker.py").write_text(_WORKER)
    (directory / "model.yml").write_text("{}\n")
    (directory / "reactor.yaml").write_text(
        "runtime:\n  import: x:Y\n  worker: slow_worker:Slow\n  config: model.yml\n"
    )
    return directory


def _wait(check: Callable[[], object], within: float = 30.0) -> None:
    deadline = time.monotonic() + within
    while not check():
        assert time.monotonic() < deadline, "not reached in time"
        time.sleep(0.05)


def test_serve_worker_reads_the_manifest_and_reports_its_phases(model_dir: Path) -> None:
    port = _free_port()
    process = _start(model_dir, port, LOAD_SECONDS="1.5", REACTOR_MODEL_WORKER_STUCK_SECONDS="1")
    try:
        _wait(lambda: _health(port)[0] is not None)
        assert _health(port) == (503, {"status": "loading"})
        _wait(lambda: _health(port)[0] == 200)
        assert (model_dir / "run" / "ready").read_text() == "ready\n"
        client = WorkerClient.connect(socket_path=model_dir / "run" / "sock", slot=0)
        # The manifest's config path reached load(), resolved.
        assert client.step("x") == str(model_dir / "model.yml")
        hung = threading.Thread(target=lambda: client.step("hang", timeout=10))
        hung.start()
        _wait(lambda: _health(port)[1].get("status") == "stuck", within=5)
        hung.join(10)
        assert _health(port)[0] == 200
        client.close()
        process.send_signal(signal.SIGTERM)
        assert process.wait(10) == 0
        assert not (model_dir / "run" / "ready").exists()
        assert not (model_dir / "run" / "sock").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_serve_worker_refuses_without_a_worker_half(tmp_path: Path) -> None:
    (tmp_path / "reactor.yaml").write_text("runtime:\n  import: x:Y\n")
    result = subprocess.run(
        [sys.executable, "-m", "reactor_runtime.model_worker"],
        cwd=tmp_path,
        capture_output=True,
        env={**os.environ, "REACTOR_MODEL_WORKER_SOCKET": str(tmp_path / "s")},
        check=False,
    )
    assert result.returncode == 1
    assert b"no worker half named" in result.stderr


def test_serve_worker_unlinks_blocks_a_previous_worker_left(model_dir: Path) -> None:
    from multiprocessing import shared_memory

    port = _free_port()
    stale = shared_memory.SharedMemory(name=f"rrs{port}dead_w0c0_0", create=True, size=64)
    stale.close()
    process = _start(model_dir, port)
    try:
        _wait(lambda: _health(port)[0] == 200)
        assert not Path(f"/dev/shm/rrs{port}dead_w0c0_0").exists()
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(10)


def test_the_cli_names_its_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["nope"])
    assert raised.value.code == 2
    assert "unknown command 'nope'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as raised:
        cli.main([])
    assert raised.value.code == 2


def test_the_cli_serve_is_the_runtime_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr("reactor_runtime.serve.main", lambda: called.append(True))
    cli.main(["serve"])
    assert called == [True]
