"""shared_worker_echo: the v1 connection churn, the model's engine in a shared model server."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

# A runtime from before the model server (runtime_ref) has nothing for this
# scenario to test. Only a missing package skips.
if importlib.util.find_spec("reactor_runtime.model_server") is None:
    pytest.skip("this runtime has no model_server package", allow_module_level=True)
from endurance.churn import run_churn
from endurance.harness import SUITE_DIR, RuntimeProcess, V1Client
from endurance.report import DEFAULT_OUTPUT_DIR

MODEL_DIR = SUITE_DIR / "model-shared-worker"

DESCRIPTION = (
    "One long-lived session on a model split in two: the tracked echo runs in the "
    "runtime, and its EchoModel in a model server process (python -m "
    "reactor_runtime.model_server) beside it, so every frame goes to the model server "
    "and back through its socket and shared memory. Each cycle a v1 client joins, "
    "streams until 60 frames come back, and closes. The session's one link to the "
    "model server lasts the whole run, so what the churn checks must not grow with it."
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_available(server: subprocess.Popen[bytes], port: int, timeout_s: float = 60.0) -> None:
    """Wait until the model server's ``/health`` says it serves."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        assert server.poll() is None, "the model server exited while starting"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as answer:
                if json.load(answer)["state"] == "available":
                    return
        except OSError:
            pass
        time.sleep(0.1)
    raise TimeoutError("the model server never became available")


@pytest.fixture
def runtime(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[RuntimeProcess]:
    """The model server, then a runtime serving the application half against it."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="rre"))
    socket_path = directory / "sock"
    (directory / "s0").mkdir()
    pythonpath = os.pathsep.join(
        [
            str(MODEL_DIR),
            str(SUITE_DIR / "model"),
            str(SUITE_DIR.parent / "examples" / "echo"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    health_port = _free_port()
    log = (DEFAULT_OUTPUT_DIR / f"{request.node.name}-model-server.log").open("w")
    server = subprocess.Popen(
        [sys.executable, "-m", "reactor_runtime.model_server"],
        cwd=MODEL_DIR,
        env={
            **os.environ,
            "PYTHONPATH": pythonpath,
            "HOST": "127.0.0.1",
            "REACTOR_MODEL_WORKER_SOCKET": str(socket_path),
            "REACTOR_MODEL_WORKER_HEALTH_PORT": str(health_port),
            "REACTOR_MAX_SESSIONS": "1",
            "REACTOR_SHM_SLOT_DIR": str(directory / "s{slot}"),
            "REACTOR_LOG_LEVEL": os.environ.get("REACTOR_LOG_LEVEL", "WARNING"),
        },
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    proc: RuntimeProcess | None = None
    try:
        _wait_available(server, health_port)
        monkeypatch.setenv("REACTOR_MODEL_WORKER_SOCKET", str(socket_path))
        monkeypatch.setenv("REACTOR_SESSION_SLOT", "0")
        monkeypatch.setenv("REACTOR_SHM_DIR", str(directory / "s0"))
        monkeypatch.setenv("PYTHONPATH", pythonpath)
        proc = RuntimeProcess(
            DEFAULT_OUTPUT_DIR / f"{request.node.name}-runtime.log", model_dir=MODEL_DIR
        )
        proc.wait_ready()
        yield proc
    finally:
        if proc is not None:
            proc.stop()
        server.terminate()
        try:
            server.wait(15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()
        shutil.rmtree(directory, ignore_errors=True)


async def test_shared_worker_echo(runtime: RuntimeProcess) -> None:
    """Release everything a connection held, with the model's engine in a shared model server."""
    await run_churn(
        runtime, test_name="shared-worker-echo", description=DESCRIPTION, client=V1Client
    )
