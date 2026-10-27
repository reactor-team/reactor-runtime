"""A model without a worker half never loads, starts, or reaches any model-worker code.

The single-session runtime is the default, and nothing of the worker path may
run for it: no import of ``reactor_runtime.model_worker``, no thread, no shared
memory, no change to its health or its session routes.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

_STARTER = Path(__file__).parents[3] / "examples" / "starter"

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="process tests on Linux")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _imported(stderr: str) -> set[str]:
    """The modules ``-X importtime`` reported, by their full name."""
    names = set()
    for line in stderr.splitlines():
        if line.startswith("import time:") and "|" in line:
            names.add(line.rsplit("|", 1)[1].strip())
    return names


def test_importing_the_runtime_imports_no_worker_code() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "importtime",
            "-c",
            "import reactor_runtime, reactor_runtime.serve, reactor_runtime.runner, "
            "reactor_runtime.manifest",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    imported = _imported(result.stderr)
    assert "reactor_runtime.serve" in imported
    assert not {m for m in imported if m.startswith("reactor_runtime.model_worker")}


def test_a_single_session_model_serves_without_any_worker_code(tmp_path: Path) -> None:
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("REACTOR_MODEL_WORKER")}
    env.update({"PORT": str(port), "HOST": "127.0.0.1"})
    err = (tmp_path / "stderr").open("w")
    process = subprocess.Popen(
        [sys.executable, "-X", "importtime", "-m", "reactor_runtime.serve"],
        cwd=_STARTER,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=err,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            assert process.poll() is None, (tmp_path / "stderr").read_text()
            try:
                with urllib.request.urlopen(f"{base}/session", timeout=2) as response:
                    if json.loads(response.read())["state"] == "ready":
                        break
            except OSError:
                pass
            assert time.monotonic() < deadline, "the runtime never became ready"
            time.sleep(0.2)
        request = urllib.request.Request(
            f"{base}/start_session",
            data=b"{}",
            method="POST",
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
        with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
            assert response.status == 200
        threads = {
            path.read_text().strip() for path in Path(f"/proc/{process.pid}/task").glob("*/comm")
        }
        assert "worker-client" not in threads
        process.send_signal(signal.SIGTERM)
        assert process.wait(30) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        err.close()
    imported = _imported((tmp_path / "stderr").read_text())
    # The model's own imports are recorded too (the starter draws with PIL).
    assert "PIL.ImageDraw" in imported
    assert not {m for m in imported if m.startswith("reactor_runtime.model_worker")}


def test_a_session_half_without_a_worker_socket_fails_its_start_clearly(tmp_path: Path) -> None:
    (tmp_path / "reactor.yaml").write_text("runtime:\n  import: half:Half\n")
    (tmp_path / "half.py").write_text(
        "from reactor_runtime import ReactorApp\n"
        "from reactor_runtime.model_worker import WorkerClient\n"
        "class Half(ReactorApp):\n"
        "    def load(self, config_path):\n"
        "        self.worker = WorkerClient.connect()\n"
        "    def generate(self, input):\n"
        "        return None\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("REACTOR_MODEL_WORKER")}
    env.update({"PORT": str(_free_port()), "HOST": "127.0.0.1", "EXIT_TIMEOUT_SECONDS": "5"})
    result = subprocess.run(
        [sys.executable, "-m", "reactor_runtime.serve"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "REACTOR_MODEL_WORKER_SOCKET is not set" in result.stderr
    assert "model failed to load" in result.stderr
