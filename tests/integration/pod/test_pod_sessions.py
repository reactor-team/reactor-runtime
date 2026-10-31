"""A pod of one model server and several session runtimes, as separate processes.

Each process stands in for one container of a pod that splits a model in two:
the model server is ``python -m reactor_runtime.model_server``, and
every session is an unchanged ``python -m reactor_runtime.serve`` on its own
port. They share one socket directory, and each slot has a shared-memory
directory of its own. The model is ``examples/echo`` split in two. The test
plays the kubelet: it starts the processes, kills them, and starts them again.
A bare ``reactor_webrtc`` client (the endurance suite's) streams a webcam to a
session over real HTTP and WebRTC and counts the frames that come back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("reactor_webrtc")

from endurance.harness import BareClient, RuntimeEndpoint, gray_frame

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(sys.platform != "linux", reason="process tests on Linux"),
]

_ROOT = Path(__file__).parents[3]
_MODEL = Path(__file__).parent / "split_echo"
_GPU_MODEL = Path(__file__).parent / "split_echo_gpu"
_SESSIONS = 3
_FRAME = gray_frame(320, 240)


def _free_ports(count: int) -> int:
    for _ in range(50):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        socks = []
        try:
            for port in range(base, base + count):
                sock = socket.socket()
                sock.bind(("127.0.0.1", port))
                socks.append(sock)
        except OSError:
            continue
        finally:
            for sock in socks:
                sock.close()
        return base
    raise RuntimeError("no run of free ports")


class Pod:
    """The model server and the session processes, with what the containers would share."""

    def __init__(self, tmp_path: Path, workspace: Path = _MODEL, **env: str) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="rrpod"))
        self.socket = self.dir / "sock"
        self.slot_dirs = [self.dir / f"s{slot}" for slot in range(_SESSIONS)]
        for directory in self.slot_dirs:
            directory.mkdir()
        self.workspace = workspace
        base = _free_ports(_SESSIONS + 1)
        self.server_port = base
        self.ports = [base + 1 + slot for slot in range(_SESSIONS)]
        self.logs = tmp_path
        self.env = {
            **os.environ,
            "HOST": "127.0.0.1",
            "REACTOR_MODEL_WORKER_SOCKET": str(self.socket),
            "REACTOR_LOG_FORMAT": "json",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(_ROOT / "examples" / "echo"),
                    str(_MODEL),
                    str(_ROOT),
                    os.environ.get("PYTHONPATH", ""),
                ]
            ),
            "SIGTERM_GRACE_PERIOD": "5",
            "EXIT_TIMEOUT_SECONDS": "5",
            "FAULT_MARK": str(tmp_path / "fault.mark"),
            **env,
        }
        self.server: subprocess.Popen[bytes] | None = None
        self.sessions: list[subprocess.Popen[bytes] | None] = [None] * _SESSIONS
        self._starts: dict[str, int] = {}

    def _spawn(self, name: str, command: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
        count = self._starts[name] = self._starts.get(name, 0) + 1
        log = (self.logs / f"{name}.{count}.log").open("wb")
        return subprocess.Popen(
            command, cwd=self.workspace, env=env, stdout=log, stderr=subprocess.STDOUT
        )

    def start_server(self) -> None:
        env = {
            **self.env,
            "REACTOR_MODEL_WORKER_HEALTH_PORT": str(self.server_port),
            "REACTOR_MAX_SESSIONS": str(_SESSIONS),
            "REACTOR_SHM_SLOT_DIR": str(self.dir / "s{slot}"),
        }
        self.server = self._spawn(
            "server", [sys.executable, "-m", "reactor_runtime.model_server"], env
        )

    def wait_server(self, within: float = 60.0) -> None:
        deadline = time.monotonic() + within
        endpoint = RuntimeEndpoint(f"http://127.0.0.1:{self.server_port}")
        while time.monotonic() < deadline:
            assert self.server is not None
            assert self.server.poll() is None, self.log("server")
            with contextlib.suppress(OSError):
                # /health is 200 while the model loads too; its state says when it serves.
                status, body = endpoint.request("GET", "/health")
                if status == 200 and body.get("state") == "available":
                    return
            time.sleep(0.1)
        raise AssertionError("the model server never became available")

    def start_session(self, slot: int) -> None:
        env = {
            **self.env,
            "PORT": str(self.ports[slot]),
            "REACTOR_SESSION_SLOT": str(slot),
            "REACTOR_SHM_DIR": str(self.slot_dirs[slot]),
        }
        self.sessions[slot] = self._spawn(
            f"session{slot}", [sys.executable, "-m", "reactor_runtime.serve"], env
        )

    def unit(self, slot: int) -> RuntimeEndpoint:
        return RuntimeEndpoint(f"http://127.0.0.1:{self.ports[slot]}")

    def health(self, slot: int) -> tuple[int | None, dict[str, Any]]:
        try:
            return self.unit(slot).request("GET", "/health")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())
        except OSError:
            return None, {}

    def wait_available(self, slot: int, within: float = 60.0) -> None:
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            process = self.sessions[slot]
            assert process is not None
            assert process.poll() is None, self.log(f"session{slot}")
            status, body = self.health(slot)
            if status == 200 and body.get("state") == "available":
                return
            time.sleep(0.1)
        raise AssertionError(f"slot {slot} never became available: {self.health(slot)}")

    def start(self) -> None:
        self.start_server()
        self.wait_server()
        for slot in range(_SESSIONS):
            self.start_session(slot)
        for slot in range(_SESSIONS):
            self.wait_available(slot)

    def start_session_id(self, slot: int) -> str:
        return self.unit(slot).request("POST", "/start_session", {})[1]["session_id"]

    def log(self, name: str) -> str:
        count = self._starts.get(name, 1)
        return "\n".join(
            (self.logs / f"{name}.{n}.log").read_text(errors="replace")
            for n in range(1, count + 1)
            if (self.logs / f"{name}.{n}.log").exists()
        )

    def records(self, name: str) -> list[dict[str, Any]]:
        """Every JSON record the process *name* wrote, over all its starts."""
        found = []
        for line in self.log(name).splitlines():
            with contextlib.suppress(ValueError):
                record = json.loads(line)
                if isinstance(record, dict):
                    found.append(record)
        return found

    def blocks(self) -> list[Path]:
        return [block for directory in self.slot_dirs for block in directory.iterdir()]

    def stop(self) -> None:
        for process in [*self.sessions, self.server]:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(10)
        shutil.rmtree(self.dir, ignore_errors=True)


@pytest.fixture
def pod(tmp_path: Path) -> Iterator[Callable[..., Pod]]:
    made: list[Pod] = []

    def make(workspace: Path = _MODEL, **env: str) -> Pod:
        one = Pod(tmp_path, workspace, **env)
        made.append(one)
        return one

    yield make
    for one in made:
        one.stop()


async def _stream(pod: Pod, slot: int) -> int:
    try:
        session_id = pod.start_session_id(slot)
    except urllib.error.HTTPError as error:
        body = error.read()
        raise AssertionError(
            f"slot {slot} refused the start: {error.code} {body!r}; "
            f"session {pod.unit(slot).get('/session')}"
        ) from None
    client = BareClient(pod.unit(slot), session_id)
    return await client.run(_FRAME)


async def _until(check: Callable[[], Any], within: float = 60.0, what: str = "") -> Any:
    deadline = time.monotonic() + within
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"not reached in {within} s: {what}")
        await asyncio.sleep(0.2)


async def _exit_codes(pod: Pod, within: float = 30.0) -> list[int]:
    return [await asyncio.to_thread(p.wait, within) for p in pod.sessions if p is not None]


async def test_every_session_streams_through_the_one_model_server(pod: Any) -> None:
    one: Pod = pod()
    one.start()
    received = await asyncio.gather(*(_stream(one, slot) for slot in range(_SESSIONS)))
    assert all(count >= 60 for count in received), received
    opened = {
        r["slot"]
        for r in one.records("server")
        if r.get("msg") == "session opened" and r.get("runtime_role") == "model_server"
    }
    assert opened == set(range(_SESSIONS))
    for slot in range(_SESSIONS):
        records = one.records(f"session{slot}")
        # Every session process is the plain runtime: none loaded the model itself.
        assert not any(r.get("msg") == "loading the model" for r in records)
        assert any(r.get("runtime_role") == "session" and r.get("slot") == slot for r in records)


async def test_sessions_that_start_before_the_model_server_wait_in_load_without_exiting(
    pod: Any,
) -> None:
    one: Pod = pod()
    for slot in range(_SESSIONS):
        one.start_session(slot)
    await _until(lambda: one.health(0)[0] is not None, what="slot 0 answering")
    # connect() blocks in load(): the runtime is in its ordinary loading state.
    _, body = one.health(0)
    assert body.get("state") == "loading", body
    await asyncio.sleep(2)
    assert all(p is not None and p.poll() is None for p in one.sessions)
    assert one.health(0)[1].get("state") == "loading"
    one.start_server()
    one.wait_server()
    for slot in range(_SESSIONS):
        one.wait_available(slot, within=30)
    assert await _stream(one, 1) >= 60


async def test_a_killed_model_server_takes_every_session_runtime_down_and_restarts_wait_for_it(
    pod: Any,
) -> None:
    one: Pod = pod()
    one.start()
    client = BareClient(one.unit(0), one.start_session_id(0))
    assert await client.open(_FRAME) >= 60
    assert one.server is not None
    one.server.send_signal(signal.SIGKILL)
    one.server.wait(10)

    # The live session and the idle ones alike exit as a failure, at once.
    started = time.monotonic()
    assert await _exit_codes(one) == [1] * _SESSIONS
    assert time.monotonic() - started < 15
    client.close()
    for slot in range(_SESSIONS):
        lost = [r for r in one.records(f"session{slot}") if r.get("msg") == "lost the model server"]
        assert [r["slot"] for r in lost] == [slot]

    # The kubelet restarts every container; the sessions wait in load() for the model server.
    for slot in range(_SESSIONS):
        one.start_session(slot)
    await _until(lambda: one.health(0)[1].get("state") == "loading", what="slot 0 loading")
    one.start_server()
    one.wait_server()
    for slot in range(_SESSIONS):
        one.wait_available(slot, within=30)
    received = await asyncio.gather(*(_stream(one, slot) for slot in range(_SESSIONS)))
    assert all(count >= 60 for count in received), received


async def test_a_killed_session_process_leaves_the_others_serving(pod: Any) -> None:
    one: Pod = pod()
    one.start()
    victim = one.sessions[2]
    assert victim is not None
    victim.send_signal(signal.SIGKILL)
    victim.wait(10)
    received = await asyncio.gather(_stream(one, 0), _stream(one, 1))
    assert all(count >= 60 for count in received)
    await _until(
        lambda: any(
            r.get("msg") == "session closed" and r.get("slot") == 2 for r in one.records("server")
        ),
        what="slot 2 closed",
    )
    one.start_session(2)
    one.wait_available(2)
    assert await _stream(one, 2) >= 60


@pytest.mark.parametrize("stage", ["input", "output", "model"])
async def test_a_fault_in_one_session_ends_only_that_session(pod: Any, stage: str) -> None:
    one: Pod = pod(FAULT_STAGE=stage, FAULT_SLOT="1")
    one.start()
    faulty = asyncio.create_task(_stream(one, 1))
    healthy = await asyncio.gather(_stream(one, 0), _stream(one, 2))
    assert all(count >= 60 for count in healthy)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(faulty, 60)
    process = one.sessions[1]
    assert process is not None
    # A model that crashes ends its session and its process, for the kubelet
    # to restart that container alone.
    assert await asyncio.to_thread(process.wait, 30) == 1
    assert f"injected {stage} fault in slot 1" in one.log("session1")
    assert one.server is not None
    assert one.server.poll() is None
    assert all(p is not None and p.poll() is None for p in (one.sessions[0], one.sessions[2]))
    one.start_session(1)
    one.wait_available(1)
    assert await _stream(one, 1) >= 60


async def test_sigterm_drains_each_session_then_the_model_server_stops(pod: Any) -> None:
    one: Pod = pod()
    one.start()
    clients = []
    for slot in range(_SESSIONS):
        client = BareClient(one.unit(slot), one.start_session_id(slot))
        assert await client.open(_FRAME) >= 60
        clients.append(client)
    started = time.monotonic()
    for process in one.sessions:
        assert process is not None
        process.send_signal(signal.SIGTERM)
    assert await _exit_codes(one) == [0] * _SESSIONS
    assert time.monotonic() - started < 5 + 5 + 5
    for client in clients:
        client.close()
    assert one.server is not None
    one.server.send_signal(signal.SIGTERM)
    assert await asyncio.to_thread(one.server.wait, 10) == 0
    assert not one.socket.exists()
    for slot in range(_SESSIONS):
        assert not any(
            r.get("msg") == "lost the model server" for r in one.records(f"session{slot}")
        )
    assert one.blocks() == []


def _gpu_apps(index: str) -> dict[int, int]:
    """The pids using the GPU at *index*, with the MiB each holds."""
    uuid = subprocess.run(
        ["nvidia-smi", "-i", index, "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    apps = {}
    for row in rows:
        gpu, pid, used = (part.strip() for part in row.split(","))
        if gpu == uuid:
            apps[int(pid)] = int(used)
    return apps


@pytest.mark.skipif(
    not os.getenv("REACTOR_TEST_GPU"), reason="set REACTOR_TEST_GPU to a GPU index to run"
)
async def test_one_gpu_model_server_serves_every_session_with_one_copy_of_the_weights(
    pod: Any,
) -> None:
    gpu = os.environ["REACTOR_TEST_GPU"]
    one: Pod = pod(_GPU_MODEL)
    one.env["CUDA_VISIBLE_DEVICES"] = gpu
    one.start_server()
    one.wait_server(within=300)
    # The session runtimes see no GPU, as session containers request none.
    one.env["CUDA_VISIBLE_DEVICES"] = ""
    for slot in range(_SESSIONS):
        one.start_session(slot)
    for slot in range(_SESSIONS):
        one.wait_available(slot)
    received = await asyncio.gather(*(_stream(one, slot) for slot in range(_SESSIONS)))
    assert all(count >= 60 for count in received), received
    assert one.server is not None
    apps = _gpu_apps(gpu)
    assert list(apps) == [one.server.pid], apps
    assert apps[one.server.pid] >= 2048
    # A restart reloads the one copy; the session runtimes go down and wait for it.
    one.server.send_signal(signal.SIGKILL)
    one.server.wait(30)
    assert await _exit_codes(one) == [1] * _SESSIONS
    for slot in range(_SESSIONS):
        one.start_session(slot)
    one.env["CUDA_VISIBLE_DEVICES"] = gpu
    one.start_server()
    one.wait_server(within=300)
    for slot in range(_SESSIONS):
        one.wait_available(slot, within=60)
    received = await asyncio.gather(*(_stream(one, slot) for slot in range(_SESSIONS)))
    assert all(count >= 60 for count in received), received
    assert list(_gpu_apps(gpu)) == [one.server.pid]
