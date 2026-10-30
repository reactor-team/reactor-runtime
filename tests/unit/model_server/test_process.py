"""The model server's process: check the deployment, load, serve, and run end to end."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reactor_runtime import log
from reactor_runtime.core import (
    SHM_PAGE_BYTES,
    HealthStatus,
    RuntimeState,
    WorkerConfig,
    shm_slot_bytes,
)
from reactor_runtime.model_server import (
    LinkConfig,
    ModelClient,
    ModelServer,
    ServerRequest,
)
from reactor_runtime.model_server.config import ServerProcessConfig, server_process_config
from reactor_runtime.model_server.process import ModelServerProcess


@dataclass(frozen=True)
class Tick:
    pixels: np.ndarray


@dataclass(frozen=True)
class Frame:
    pixels: np.ndarray


class Doubler(ModelServer[Tick, Frame]):
    def load(self, config_path: Path | None) -> None:
        self.loaded_from = config_path

    def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
        return [Frame(r.payload.pixels * 2) for r in requests]


class BrokenLoad(Doubler):
    def load(self, config_path: Path | None) -> None:
        raise RuntimeError("no weights")


# Every block one page of the slot bound, so the volumes below are whole pages
# of it on any host.
_BLOCK = SHM_PAGE_BYTES
_SLOT = shm_slot_bytes(_BLOCK, _BLOCK)


def _process(root: Path, ref: str = f"{__name__}:Doubler", sessions: int = 1) -> ModelServerProcess:
    for slot in range(sessions):
        (root / f"s{slot}").mkdir(exist_ok=True)
    config = ServerProcessConfig(
        socket=str(root / "sock"), max_sessions=sessions, slot_directory=str(root / "s{slot}")
    )
    worker = WorkerConfig(import_ref=ref, shm_input_bytes=_BLOCK, shm_output_bytes=_BLOCK)
    return ModelServerProcess(worker, root / "config.yml", config)


def test_it_loads_the_model_then_serves_sessions_over_shared_memory(tmp_path: Path) -> None:
    process = _process(tmp_path)
    (tmp_path / "s0" / "rrms_w0_left_0").write_bytes(b"x")  # a previous run's output block

    async def run() -> np.ndarray:
        assert process.state() is RuntimeState.LOADING
        await process.start()
        assert process.state() is RuntimeState.AVAILABLE
        assert (log.get_state(), log.get_runtime_state()) == (None, "available")
        assert not (tmp_path / "s0" / "rrms_w0_left_0").exists()
        config = LinkConfig(socket=str(tmp_path / "sock"), slot=0, shm_dir=tmp_path / "s0")
        client = await asyncio.to_thread(ModelClient.connect, config)
        await asyncio.to_thread(client.reset)
        frame = await asyncio.to_thread(client.step, Tick(np.arange(8, dtype=np.uint8)))
        client.close()
        await process.drain()
        await process.stop()
        return frame.pixels

    pixels = asyncio.run(run())

    np.testing.assert_array_equal(pixels, np.arange(8, dtype=np.uint8) * 2)
    assert process.state() is RuntimeState.TERMINATED
    assert log.get_runtime_state() == "terminated"
    assert process.health().status is HealthStatus.UNHEALTHY


@pytest.mark.parametrize(
    ("ref", "match"),
    [
        ("json:dumps", "does not name a ModelServer subclass"),
        (f"{__name__}:Missing", "does not name a ModelServer subclass"),
    ],
)
def test_a_manifest_that_names_no_model_server_stops_the_start(
    tmp_path: Path, ref: str, match: str
) -> None:
    with pytest.raises(SystemExit, match=match):
        asyncio.run(_process(tmp_path, ref).start())


def test_a_missing_slot_directory_stops_the_start_before_the_load(tmp_path: Path) -> None:
    process = _process(tmp_path, sessions=2)
    (tmp_path / "s1").rmdir()

    with pytest.raises(SystemExit, match="slot 1's shared-memory directory"):
        asyncio.run(process.start())


def test_a_slot_path_that_is_not_a_directory_stops_the_start(tmp_path: Path) -> None:
    process = _process(tmp_path, sessions=2)
    (tmp_path / "s1").rmdir()
    (tmp_path / "s1").write_bytes(b"")

    with pytest.raises(SystemExit, match=r"slot 1's shared-memory path .*s1 is not a directory"):
        asyncio.run(process.start())


class _Volume:
    """``os.statvfs`` for a volume of *total* bytes, its free space what its files leave."""

    def __init__(self, total: int) -> None:
        self.total = total

    def __call__(self, path: Path) -> Any:
        page = SHM_PAGE_BYTES
        used = sum(-(-f.stat().st_size // page) * page for f in Path(path).iterdir())
        return SimpleNamespace(
            f_frsize=page, f_blocks=self.total // page, f_bavail=(self.total - used) // page
        )


def test_a_full_slot_volume_starts_once_the_previous_runs_blocks_are_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _process(tmp_path)
    monkeypatch.setattr("reactor_runtime.model_server.process.os.statvfs", _Volume(_SLOT))
    # A run killed while a session held a request block and its retired one:
    # the volume is full until the model server reclaims its output block.
    (tmp_path / "s0" / "rrms_w0_left").write_bytes(b"x" * _BLOCK)
    (tmp_path / "s0" / "rrms_s0_left").write_bytes(b"x" * _BLOCK)
    (tmp_path / "s0" / "rrms_s0_retired").write_bytes(b"x" * _BLOCK)

    async def run() -> RuntimeState:
        await process.start()
        try:
            return process.state()
        finally:
            await process.stop()

    assert asyncio.run(run()) is RuntimeState.AVAILABLE
    assert sorted(f.name for f in (tmp_path / "s0").iterdir()) == [
        "rrms_s0_left",
        "rrms_s0_retired",
    ]


def test_a_slot_volume_smaller_than_the_slot_bound_stops_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _process(tmp_path)
    small = _SLOT - _BLOCK
    monkeypatch.setattr("reactor_runtime.model_server.process.os.statvfs", _Volume(small))

    with pytest.raises(
        SystemExit, match=rf"holds {small} bytes and one session needs up to {_SLOT}"
    ):
        asyncio.run(process.start())


def test_a_failed_load_terminates_and_asks_the_process_to_exit_cleanly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    process = _process(tmp_path, f"{__name__}:BrokenLoad")
    shutdowns: list[bool] = []
    process.request_shutdown = lambda *, failure=False: shutdowns.append(failure)

    with caplog.at_level(logging.ERROR, logger="reactor_runtime.model_server.process"):
        asyncio.run(process.start())

    assert process.state() is RuntimeState.TERMINATED
    assert process.health().status is HealthStatus.UNHEALTHY
    assert shutdowns == [False]
    [record] = [r for r in caplog.records if r.getMessage() == "the model failed to load"]
    assert record.exc_info is not None
    assert "no weights" in str(record.exc_info[1])
    assert not (tmp_path / "sock").exists()


def test_a_misspelled_module_stops_the_start_naming_it(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"names module 'doubleer', which does not exist"):
        asyncio.run(_process(tmp_path, "doubleer:Doubler").start())


def test_a_module_the_model_imports_and_lacks_is_the_models_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "needs_more.py").write_text("import not_installed_package\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        with pytest.raises(ModuleNotFoundError, match="not_installed_package"):
            asyncio.run(_process(tmp_path, "needs_more:Model").start())
    finally:
        sys.modules.pop("needs_more", None)


# -- the configuration ----------------------------------------------------------------------


def test_the_process_configuration_needs_the_session_count_and_the_slot_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REACTOR_MAX_SESSIONS", raising=False)
    monkeypatch.setenv("REACTOR_SHM_SLOT_DIR", "/run/reactor-shm/s{slot}")
    with pytest.raises(SystemExit, match="REACTOR_MAX_SESSIONS"):
        server_process_config()
    monkeypatch.setenv("REACTOR_MAX_SESSIONS", "3")
    monkeypatch.setenv("REACTOR_SHM_SLOT_DIR", "/dev/shm")
    with pytest.raises(SystemExit, match="REACTOR_SHM_SLOT_DIR"):
        server_process_config()
    monkeypatch.setenv("REACTOR_SHM_SLOT_DIR", "/run/reactor-shm/s{slot}")
    monkeypatch.setenv("REACTOR_MODEL_WORKER_HEALTH_PORT", "9100")

    config = server_process_config()

    assert (config.max_sessions, config.slot_directory, config.health_port) == (
        3,
        "/run/reactor-shm/s{slot}",
        9100,
    )


# -- end to end -----------------------------------------------------------------------------


_MODEL = """
from dataclasses import dataclass
import numpy as np
from reactor_runtime.model_server import ModelServer

@dataclass(frozen=True)
class Tick:
    pixels: np.ndarray

@dataclass(frozen=True)
class Frame:
    pixels: np.ndarray

class Doubler(ModelServer):
    max_batch = 2

    def step(self, requests):
        return [Frame(r.payload.pixels * 2) for r in requests]
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _health(port: int) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, ConnectionError):
        return None


def _spawn(
    root: Path, model: str, ref: str = "doubler:Doubler"
) -> tuple[subprocess.Popen[bytes], int]:
    """Start ``python -m reactor_runtime.model_server`` on a workspace holding *model*."""
    (root / f"{ref.partition(':')[0]}.py").write_text(model)
    (root / "reactor.yaml").write_text(
        textwrap.dedent(
            f"""\
            runtime:
              import: app:App
              worker:
                import: {ref}
                shm_input_bytes: 65536
                shm_output_bytes: 65536
            """
        )
    )
    (root / "s0").mkdir()
    port = _free_port()
    env = {
        **os.environ,
        "REACTOR_MODEL_WORKER_SOCKET": str(root / "sock"),
        "REACTOR_MAX_SESSIONS": "1",
        "REACTOR_SHM_SLOT_DIR": str(root / "s{slot}"),
        "REACTOR_MODEL_WORKER_HEALTH_PORT": str(port),
        "HOST": "127.0.0.1",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "reactor_runtime.model_server"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return process, port


def test_python_m_model_server_serves_a_workspace_and_stops_on_sigterm(tmp_path: Path) -> None:
    worker, port = _spawn(tmp_path, _MODEL)
    try:
        deadline = time.monotonic() + 60
        while (_health(port) or {}).get("state") != "available":
            assert worker.poll() is None, worker.stdout.read().decode() if worker.stdout else ""
            assert time.monotonic() < deadline, "the model server did not become available"
            time.sleep(0.1)

        sys.path.insert(0, str(tmp_path))
        try:
            import doubler  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]  # the workspace's own module
        finally:
            sys.path.remove(str(tmp_path))
        config = LinkConfig(socket=str(tmp_path / "sock"), slot=0, shm_dir=tmp_path / "s0")
        client = ModelClient.connect(config)
        client.reset()
        frame = client.step(doubler.Tick(np.full(4, 3, dtype=np.uint8)))
        client.close()
        np.testing.assert_array_equal(frame.pixels, np.full(4, 6, dtype=np.uint8))

        worker.send_signal(signal.SIGTERM)
        assert worker.wait(timeout=60) == 0
        assert not (tmp_path / "sock").exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        sys.modules.pop("doubler", None)


def test_python_m_model_server_exits_cleanly_when_the_model_fails_to_load(tmp_path: Path) -> None:
    model = _MODEL + textwrap.dedent(
        """
        class Broken(Doubler):
            def load(self, config_path):
                raise RuntimeError("no weights")
        """
    )
    worker, _ = _spawn(tmp_path, model, "doubler:Broken")
    try:
        assert worker.wait(timeout=60) == 0
        output = worker.stdout.read().decode() if worker.stdout else ""
        assert "the model failed to load" in output
        assert "no weights" in output
        assert not (tmp_path / "sock").exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_python_m_model_server_finishes_the_load_then_exits_on_a_sigterm_during_it(
    tmp_path: Path,
) -> None:
    model = _MODEL + textwrap.dedent(
        """
        import pathlib, time

        class Slow(Doubler):
            def load(self, config_path):
                pathlib.Path("loading").touch()
                while not pathlib.Path("release").exists():
                    time.sleep(0.05)
                pathlib.Path("loaded").touch()
        """
    )
    worker, _ = _spawn(tmp_path, model, "doubler:Slow")
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "loading").exists():
            assert worker.poll() is None, worker.stdout.read().decode() if worker.stdout else ""
            assert time.monotonic() < deadline, "the model never started loading"
            time.sleep(0.05)
        worker.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        assert worker.poll() is None, "the SIGTERM cut the load short"
        (tmp_path / "release").touch()

        assert worker.wait(timeout=60) == 0
        output = worker.stdout.read().decode() if worker.stdout else ""
        assert (tmp_path / "loaded").exists()
        assert "shutdown requested" in output
        assert not (tmp_path / "sock").exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
