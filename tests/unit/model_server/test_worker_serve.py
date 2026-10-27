"""``python -m reactor_runtime.model_server``: what it reads, what it reports, and how it stops."""

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

from reactor_runtime.model_server import ModelClient, WorkerError, wire

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="process tests on Linux")

_WORKER = """
import os, time
from pathlib import Path
from reactor_runtime.model_server import ModelServer

class Slow(ModelServer):
    max_batch = 4
    shm_input_bytes = 1 << 16
    shm_output_bytes = 1 << 16

    def load(self, config_path):
        Path("loaded").write_text("yes")
        self.config = config_path
        time.sleep(float(os.environ.get("LOAD_SECONDS", "0")))

    def step(self, requests):
        if requests[0].payload == "hang":
            time.sleep(5)
        if requests[0].payload == "sticky":
            self.mark_unhealthy("CUDA error: an illegal memory access")
            raise RuntimeError("CUDA error: an illegal memory access")
        return [str(self.config) for _ in requests]


class Probed(Slow):
    def health(self):
        mode = Path("health").read_text().strip() if Path("health").exists() else ""
        if mode == "raise":
            raise RuntimeError("probe broke")
        return mode or None
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


def _manifest(directory: Path, worker: str) -> None:
    (directory / "reactor.yaml").write_text(
        f"runtime:\n  import: x:Y\n  config: model.yml\n  worker:\n{worker}"
    )


def _start(
    directory: Path, port: int, *args: str, stderr: int | None = None, **env: str
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "reactor_runtime.model_server", *args],
        cwd=directory,
        stderr=stderr,
        env={
            **{k: v for k, v in os.environ.items() if k != "REACTOR_MAX_SESSIONS"},
            "REACTOR_MODEL_WORKER_SOCKET": str(directory / "run" / "sock"),
            "REACTOR_MODEL_WORKER_HEALTH_PORT": str(port),
            "REACTOR_SHM_PREFIX": f"rrs{port}",
            "HOST": "127.0.0.1",
            **env,
        },
    )


def _run(directory: Path, **env: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "reactor_runtime.model_server"],
        cwd=directory,
        capture_output=True,
        env={
            **{k: v for k, v in os.environ.items() if k != "REACTOR_MAX_SESSIONS"},
            "REACTOR_MODEL_WORKER_SOCKET": str(directory / "run" / "sock"),
            **env,
        },
        timeout=60,
        check=False,
    )


@pytest.fixture
def model_dir() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="rrs"))
    (directory / "slow_worker.py").write_text(_WORKER)
    (directory / "model.yml").write_text("{}\n")
    _manifest(directory, "    import: slow_worker:Slow\n    max_batch: 2\n")
    return directory


def _wait(check: Callable[[], object], within: float = 30.0) -> None:
    deadline = time.monotonic() + within
    while not check():
        assert time.monotonic() < deadline, "not reached in time"
        time.sleep(0.05)


def _stop(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        return process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def _connect(directory: Path) -> ModelClient:
    client = ModelClient.connect(socket_path=directory / "run" / "sock", slot=0, connect_timeout=5)
    client.reset()
    return client


def test_the_worker_reads_its_manifest_block_and_reports_its_phases(model_dir: Path) -> None:
    port = _free_port()
    process = _start(model_dir, port, LOAD_SECONDS="1.5", REACTOR_MODEL_WORKER_STUCK_SECONDS="1")
    try:
        _wait(lambda: _health(port)[0] is not None)
        assert _health(port) == (503, {"status": "loading"})
        _wait(lambda: _health(port)[0] == 200)
        _, body = _health(port)
        assert body["status"] == "serving"
        assert body["protocol"] == wire.PROTOCOL
        assert isinstance(body["schema"], str)
        assert body["schema"]
        # The class's max_batch of 4, lowered to 2 by the manifest.
        assert body["settings"] == {
            "max_batch": 2,
            "batch_window": 0.0,
            "max_request_bytes": 256 << 20,
            "shm_input_bytes": 1 << 16,
            "shm_output_bytes": 1 << 16,
        }
        assert body["shm_slot_bytes"] == 2 << 16
        assert (model_dir / "run" / "ready").read_text() == "ready\n"
        client = _connect(model_dir)
        # The manifest's config path reached load(), resolved.
        assert client.step("x") == str(model_dir / "model.yml")
        hung = threading.Thread(target=lambda: client.step("hang", timeout=10))
        hung.start()
        _wait(lambda: _health(port)[1].get("status") == "stuck", within=5)
        hung.join(10)
        assert _health(port)[0] == 200
        client.close()
        assert _stop(process) == 0
        assert not (model_dir / "run" / "ready").exists()
        assert not (model_dir / "run" / "sock").exists()
    finally:
        _stop(process)


def test_the_worker_flag_overrides_the_manifest(model_dir: Path) -> None:
    port = _free_port()
    process = _start(model_dir, port, "--worker", "slow_worker:Probed")
    try:
        _wait(lambda: _health(port)[0] == 200)
        (model_dir / "health").write_text("the KV cache is corrupted")
        assert _health(port) == (
            503,
            {"status": "unhealthy", "reason": "the KV cache is corrupted"},
        )
    finally:
        _stop(process)


def test_health_answers_unhealthy_while_the_models_health_says_why(model_dir: Path) -> None:
    _manifest(model_dir, "    import: slow_worker:Probed\n")
    port = _free_port()
    process = _start(model_dir, port)
    try:
        _wait(lambda: _health(port)[0] == 200)
        (model_dir / "health").write_text("the GPU is not visible")
        assert _health(port) == (503, {"status": "unhealthy", "reason": "the GPU is not visible"})
        (model_dir / "health").write_text("raise")
        code, body = _health(port)
        assert code == 503
        assert body == {
            "status": "unhealthy",
            "reason": "health() raised RuntimeError: probe broke",
        }
        (model_dir / "health").write_text("")
        assert _health(port)[0] == 200
    finally:
        _stop(process)


def test_mark_unhealthy_from_a_step_latches_health_to_503(model_dir: Path) -> None:
    port = _free_port()
    process = _start(model_dir, port)
    try:
        _wait(lambda: _health(port)[0] == 200)
        client = _connect(model_dir)
        with pytest.raises(WorkerError, match="illegal memory access"):
            client.step("sticky")
        unhealthy = (503, {"status": "unhealthy", "reason": "CUDA error: an illegal memory access"})
        assert _health(port) == unhealthy
        # Latched: a step that succeeds does not clear it.
        assert client.step("x") == str(model_dir / "model.yml")
        assert _health(port) == unhealthy
        client.close()
    finally:
        _stop(process)


def test_the_worker_refuses_to_start_without_a_worker_half(tmp_path: Path) -> None:
    (tmp_path / "reactor.yaml").write_text("runtime:\n  import: x:Y\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert b"no worker half named: name runtime.worker.import" in result.stderr


@pytest.mark.parametrize(
    ("block", "reason"),
    [
        ("  worker: slow_worker:Slow\n", b"runtime.worker must be a block"),
        ("  worker:\n    max_batch: 2\n", b"runtime.worker.import must name"),
        (
            "  worker:\n    import: slow_worker:Slow\n    sessions: 4\n",
            b"unknown keys ['sessions']",
        ),
    ],
)
def test_a_malformed_worker_block_stops_the_start(
    model_dir: Path, block: str, reason: bytes
) -> None:
    (model_dir / "reactor.yaml").write_text(f"runtime:\n  import: x:Y\n{block}")
    result = _run(model_dir)
    assert result.returncode == 1
    assert reason in result.stderr
    assert not (model_dir / "loaded").exists()


@pytest.mark.parametrize(
    ("setting", "reason"),
    [
        ("max_batch: 8", b"above Slow.max_batch = 4"),
        ("shm_input_bytes: 1024", b"below Slow.shm_input_bytes"),
        ("batch_window: -1", b"at least 0"),
    ],
)
def test_settings_the_class_cannot_serve_stop_the_start_before_load(
    model_dir: Path, setting: str, reason: bytes
) -> None:
    _manifest(model_dir, f"    import: slow_worker:Slow\n    {setting}\n")
    result = _run(model_dir)
    assert result.returncode == 1
    assert reason in result.stderr
    assert not (model_dir / "loaded").exists()


def test_a_dev_shm_too_small_for_max_sessions_stops_the_start(model_dir: Path) -> None:
    result = _run(model_dir, REACTOR_MAX_SESSIONS=str(1 << 40))
    assert result.returncode == 1
    assert b"/dev/shm has" in result.stderr
    assert b"REACTOR_MAX_SESSIONS" in result.stderr
    assert not (model_dir / "loaded").exists()
    malformed = _run(model_dir, REACTOR_MAX_SESSIONS="many")
    assert malformed.returncode == 1
    assert b"REACTOR_MAX_SESSIONS 'many' must be an integer" in malformed.stderr


def test_a_dev_shm_with_room_for_max_sessions_starts(model_dir: Path) -> None:
    port = _free_port()
    process = _start(model_dir, port, REACTOR_MAX_SESSIONS="2")
    try:
        _wait(lambda: _health(port)[0] == 200)
    finally:
        assert _stop(process) == 0


def test_every_worker_line_names_its_role_and_a_sessions_lines_name_its_slot(
    model_dir: Path,
) -> None:
    port = _free_port()
    process = _start(model_dir, port, stderr=subprocess.PIPE, REACTOR_LOG_FORMAT="json")
    try:
        _wait(lambda: _health(port)[0] == 200)
        client = _connect(model_dir)
        client.close()
    finally:
        assert _stop(process) == 0
    assert process.stderr is not None
    records = [json.loads(line) for line in process.stderr.read().decode().splitlines()]
    assert records
    assert all(record["runtime_role"] == "model_server" for record in records)
    # Each generation a session's link reports ("model server link") has a worker line with the
    # same slot and generation: "session opened" for the first, "session reset" for each reset.
    named = ("session opened", "session reset", "session closed")
    sessions = [r for r in records if r["msg"] in named]
    assert [(r["msg"], r["slot"], r["generation"]) for r in sessions] == [
        ("session opened", 0, 1),
        ("session reset", 0, 2),
        ("session closed", 0, 2),
    ]


def test_the_worker_unlinks_blocks_its_predecessor_left_and_no_others(model_dir: Path) -> None:
    from multiprocessing import shared_memory

    port = _free_port()
    (model_dir / "run").mkdir()
    (model_dir / "run" / "shm-prefix").write_text(f"rrs{port}dead_")
    stale = shared_memory.SharedMemory(name=f"rrs{port}dead_w0c0_out", create=True, size=64)
    other = shared_memory.SharedMemory(name=f"rrs{port}live_w0c0_out", create=True, size=64)
    stale.close()
    process = _start(model_dir, port)
    try:
        _wait(lambda: _health(port)[0] == 200)
        assert not Path(f"/dev/shm/rrs{port}dead_w0c0_out").exists()
        # Another worker's blocks under the same base are left alone.
        assert Path(f"/dev/shm/rrs{port}live_w0c0_out").exists()
        assert (model_dir / "run" / "shm-prefix").read_text().startswith(f"rrs{port}")
    finally:
        _stop(process)
        other.close()
        other.unlink()
    assert not (model_dir / "run" / "shm-prefix").exists()
