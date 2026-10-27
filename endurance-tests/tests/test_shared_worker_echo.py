"""shared_worker_echo: the v1 connection churn, the model's engine in a shared model worker."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

# A runtime from before the model worker (runtime_ref) has nothing for this
# scenario to test. Only a missing package skips.
if importlib.util.find_spec("reactor_runtime.model_server") is None:
    pytest.skip("this runtime has no model_server package", allow_module_level=True)
from endurance.churn import run_churn
from endurance.harness import SUITE_DIR, RuntimeProcess, V1Client
from endurance.report import DEFAULT_OUTPUT_DIR

MODEL_DIR = SUITE_DIR / "model-shared-worker"

DESCRIPTION = (
    "One long-lived session on a model split in two: the tracked echo runs in the "
    "runtime, and its EchoModel in a model worker process (python -m "
    "reactor_runtime.model_server) beside it, so every frame goes to the worker and "
    "back through its socket and shared memory. Each cycle a v1 client joins, "
    "streams until 60 frames come back, and closes. The session's one connection to "
    "the worker lasts the whole run, so what the churn checks must not grow with it."
)


@pytest.fixture
def runtime(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[RuntimeProcess]:
    """The model worker, then a runtime serving the session half against it."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="rre"))
    socket_path = directory / "sock"
    pythonpath = os.pathsep.join(
        [
            str(MODEL_DIR),
            str(SUITE_DIR / "model"),
            str(SUITE_DIR.parent / "examples" / "echo"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    log = (DEFAULT_OUTPUT_DIR / f"{request.node.name}-worker.log").open("w")
    worker = subprocess.Popen(
        [sys.executable, "-m", "reactor_runtime.model_server", "--socket", str(socket_path)],
        cwd=MODEL_DIR,
        env={
            **os.environ,
            "PYTHONPATH": pythonpath,
            "REACTOR_SHM_PREFIX": "rrendw",
            "REACTOR_LOG_LEVEL": os.environ.get("REACTOR_LOG_LEVEL", "WARNING"),
        },
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 60
    while not (directory / "ready").exists():
        assert worker.poll() is None, "the model worker exited while starting"
        assert time.monotonic() < deadline, "the model worker never became ready"
        time.sleep(0.1)
    monkeypatch.setenv("REACTOR_MODEL_WORKER_SOCKET", str(socket_path))
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    proc = RuntimeProcess(
        DEFAULT_OUTPUT_DIR / f"{request.node.name}-runtime.log", model_dir=MODEL_DIR
    )
    try:
        proc.wait_ready()
        yield proc
    finally:
        proc.stop()
        worker.terminate()
        try:
            worker.wait(15)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        log.close()
        shutil.rmtree(directory, ignore_errors=True)


async def test_shared_worker_echo(runtime: RuntimeProcess) -> None:
    """Release everything a connection held, with the model's engine in a shared worker."""
    await run_churn(
        runtime, test_name="shared-worker-echo", description=DESCRIPTION, client=V1Client
    )
