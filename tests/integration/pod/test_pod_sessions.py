"""A pod of one model worker and several session runtimes, as separate processes.

Each process stands in for one container of the pod the docs describe: the
worker is ``python -m reactor_runtime.model_worker`` (``reactor-runtime
serve-worker``), and every session is an unchanged ``python -m
reactor_runtime.serve`` on its own port, all sharing one socket directory and
``/dev/shm``. The model is ``examples/echo`` split in two. The test plays the
kubelet: it starts the processes, kills them, and starts them again. A bare
``reactor_webrtc`` client (the endurance suite's) streams a webcam to a session
over real HTTP and WebRTC and counts the frames that come back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
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


def _blocks(prefix: str) -> list[Path]:
    return list(Path("/dev/shm").glob(f"{prefix}*"))


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


class Pod:
    """The worker and the session processes, with what the containers would share."""

    def __init__(self, tmp_path: Path, **env: str) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="rrpod"))
        self.socket = self.dir / "sock"
        base = _free_ports(_SESSIONS + 1)
        self.worker_port = base
        self.ports = [base + 1 + slot for slot in range(_SESSIONS)]
        self.logs = tmp_path
        self.env = {
            **os.environ,
            "HOST": "127.0.0.1",
            "REACTOR_MODEL_WORKER_SOCKET": str(self.socket),
            "REACTOR_SHM_PREFIX": f"rrpod{secrets.token_hex(2)}",
            "PYTHONPATH": os.pathsep.join(
                [str(_ROOT / "examples" / "echo"), str(_ROOT), os.environ.get("PYTHONPATH", "")]
            ),
            "SIGTERM_GRACE_PERIOD": "5",
            "EXIT_TIMEOUT_SECONDS": "5",
            "FAULT_MARK": str(tmp_path / "fault.mark"),
            **env,
        }
        self.worker: subprocess.Popen[bytes] | None = None
        self.sessions: list[subprocess.Popen[bytes] | None] = [None] * _SESSIONS
        self._starts: dict[str, int] = {}

    def _spawn(self, name: str, command: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
        count = self._starts[name] = self._starts.get(name, 0) + 1
        log = (self.logs / f"{name}.{count}.log").open("wb")
        return subprocess.Popen(command, cwd=_MODEL, env=env, stdout=log, stderr=subprocess.STDOUT)

    def start_worker(self) -> None:
        env = {**self.env, "REACTOR_MODEL_WORKER_HEALTH_PORT": str(self.worker_port)}
        self.worker = self._spawn(
            "worker", [sys.executable, "-m", "reactor_runtime.model_worker"], env
        )

    def wait_worker(self, within: float = 60.0) -> None:
        deadline = time.monotonic() + within
        endpoint = RuntimeEndpoint(f"http://127.0.0.1:{self.worker_port}")
        while time.monotonic() < deadline:
            assert self.worker is not None
            assert self.worker.poll() is None, self.log("worker")
            with contextlib.suppress(OSError):
                if endpoint.request("GET", "/health")[0] == 200:
                    return
            time.sleep(0.1)
        raise AssertionError("the worker never became ready")

    def start_session(self, slot: int) -> None:
        env = {**self.env, "PORT": str(self.ports[slot]), "REACTOR_SESSION_SLOT": str(slot)}
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

    def wait_healthy(self, slot: int, within: float = 60.0) -> None:
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            process = self.sessions[slot]
            assert process is not None
            assert process.poll() is None, self.log(f"session{slot}")
            # /health is 200 while the model loads too; ready is its state word.
            status, body = self.health(slot)
            if status == 200 and body.get("state") == "available":
                return
            time.sleep(0.1)
        raise AssertionError(f"slot {slot} never became healthy: {self.health(slot)}")

    def start(self) -> None:
        self.start_worker()
        self.wait_worker()
        for slot in range(_SESSIONS):
            self.start_session(slot)
        for slot in range(_SESSIONS):
            self.wait_healthy(slot)

    def start_session_id(self, slot: int) -> str:
        return self.unit(slot).request("POST", "/start_session", {})[1]["session_id"]

    def events(self, slot: int) -> list[dict[str, Any]]:
        """Read the session's journal so far."""
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", self.ports[slot], timeout=1.0)
        conn.request("GET", "/events?since=0")
        response = conn.getresponse()
        events: list[dict[str, Any]] = []
        with contextlib.suppress(TimeoutError, OSError):
            while True:
                line = response.fp.readline()
                if not line:
                    break
                if line.startswith(b"data: "):
                    events.append(json.loads(line[6:]))
        conn.close()
        return events

    def log(self, name: str) -> str:
        count = self._starts.get(name, 1)
        return "\n".join(
            (self.logs / f"{name}.{n}.log").read_text(errors="replace")
            for n in range(1, count + 1)
            if (self.logs / f"{name}.{n}.log").exists()
        )

    def stop(self) -> None:
        for process in [*self.sessions, self.worker]:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(10)
        shutil.rmtree(self.dir, ignore_errors=True)
        for block in Path("/dev/shm").glob(f"{self.env['REACTOR_SHM_PREFIX']}*"):
            block.unlink(missing_ok=True)


@pytest.fixture
def pod(tmp_path: Path) -> Iterator[Callable[..., Pod]]:
    made: list[Pod] = []

    def make(**env: str) -> Pod:
        one = Pod(tmp_path, **env)
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


async def test_every_session_streams_through_the_one_worker(pod: Any) -> None:
    one: Pod = pod()
    one.start()
    received = await asyncio.gather(*(_stream(one, slot) for slot in range(_SESSIONS)))
    assert all(count >= 60 for count in received), received
    worker_log = one.log("worker")
    for slot in range(_SESSIONS):
        assert f"slot={slot}" in worker_log
    # Every session process is the plain runtime: none loaded the model itself.
    assert "model worker loaded" not in one.log("session0")


async def test_sessions_that_start_before_the_worker_wait_without_exiting(pod: Any) -> None:
    one: Pod = pod()
    for slot in range(_SESSIONS):
        one.start_session(slot)
    await _until(lambda: one.health(0)[0] == 503, what="slot 0 answering unavailable")
    _, body = one.health(0)
    assert body["detail"].startswith("worker_unavailable: ")
    await asyncio.sleep(2)
    assert all(p is not None and p.poll() is None for p in one.sessions)
    one.start_worker()
    one.wait_worker()
    for slot in range(_SESSIONS):
        one.wait_healthy(slot, within=30)
    assert await _stream(one, 1) >= 60


async def test_a_killed_worker_ends_live_sessions_and_they_recover(pod: Any) -> None:
    one: Pod = pod()
    one.start()
    client = BareClient(one.unit(0), one.start_session_id(0))
    assert await client.open(_FRAME) >= 60
    pids = [p.pid for p in one.sessions if p is not None]
    assert one.worker is not None
    one.worker.send_signal(signal.SIGKILL)
    one.worker.wait(10)

    await _until(lambda: one.health(0)[0] == 503, what="slot 0 not ready")
    _, body = one.health(0)
    assert body["detail"].startswith("worker_died: ")
    await _until(
        lambda: one.unit(0).get("/session")["state"] == "ready", what="slot 0 session ended"
    )
    stops = [e for e in one.events(0) if e.get("event") == "stop_session"]
    assert stops, one.events(0)
    assert stops[-1]["detail"]["reason"] == "error"
    assert stops[-1]["detail"]["close_reason"].startswith("worker_died: ")
    with pytest.raises(urllib.error.HTTPError) as refused:
        one.unit(1).request("POST", "/start_session", {})
    assert refused.value.code == 503
    client.close()
    # No session process exits for it.
    assert [pid for pid in pids if _alive(pid)] == pids

    one.start_worker()
    one.wait_worker()
    for slot in range(_SESSIONS):
        one.wait_healthy(slot, within=30)
    received = await asyncio.gather(*(_stream(one, slot) for slot in range(_SESSIONS)))
    assert all(count >= 60 for count in received), received
    assert [p.pid for p in one.sessions if p is not None] == pids


async def test_a_killed_session_process_leaves_the_others_serving(pod: Any) -> None:
    one: Pod = pod()
    one.start()
    victim = one.sessions[2]
    assert victim is not None
    victim.send_signal(signal.SIGKILL)
    victim.wait(10)
    received = await asyncio.gather(_stream(one, 0), _stream(one, 1))
    assert all(count >= 60 for count in received)
    await _until(lambda: "session closed slot=2" in one.log("worker"), what="slot 2 closed")
    one.start_session(2)
    one.wait_healthy(2)
    assert await _stream(one, 2) >= 60


@pytest.mark.parametrize("stage", ["input", "output", "worker"])
async def test_a_fault_in_one_session_ends_only_that_session(pod: Any, stage: str) -> None:
    one: Pod = pod(FAULT_STAGE=stage, FAULT_SLOT="1")
    one.start()
    faulty = asyncio.create_task(_stream(one, 1))
    healthy = await asyncio.gather(_stream(one, 0), _stream(one, 2))
    assert all(count >= 60 for count in healthy)
    try:
        await asyncio.wait_for(faulty, 60)
    except Exception as exc:
        detail = exc.read() if isinstance(exc, urllib.error.HTTPError) else b""
        print("faulty stream ended with", repr(exc), detail, one.health(1))
    process = one.sessions[1]
    assert process is not None
    # Today's runtime: a model that crashes ends its session and its process,
    # for the kubelet to restart that container alone.
    assert await asyncio.to_thread(process.wait, 30) != 0
    assert f"injected {stage} fault in slot 1" in one.log("session1") + one.log("worker")
    assert one.worker is not None
    assert one.worker.poll() is None
    assert all(p is not None and p.poll() is None for p in (one.sessions[0], one.sessions[2]))
    one.start_session(1)
    one.wait_healthy(1)
    assert await _stream(one, 1) >= 60


async def test_sigterm_drains_each_session_then_the_worker_stops(pod: Any) -> None:
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
    codes = [await asyncio.to_thread(p.wait, 30) for p in one.sessions if p is not None]
    assert codes == [0] * _SESSIONS
    assert time.monotonic() - started < 5 + 5 + 5
    for client in clients:
        client.close()
    assert one.worker is not None
    one.worker.send_signal(signal.SIGTERM)
    assert await asyncio.to_thread(one.worker.wait, 10) == 0
    assert not one.socket.exists()
    for slot in range(_SESSIONS):
        assert "worker_died" not in one.log(f"session{slot}")
    assert not _blocks(one.env["REACTOR_SHM_PREFIX"])


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
async def test_one_gpu_worker_serves_every_session_with_one_copy_of_the_weights(
    pod: Any,
) -> None:
    gpu = os.environ["REACTOR_TEST_GPU"]
    one: Pod = pod(REACTOR_WORKER_IMPORT="gpu_worker:GpuEchoWorker")
    one.env["CUDA_VISIBLE_DEVICES"] = gpu
    one.start_worker()
    one.wait_worker(within=300)
    # The session runtimes see no GPU, as session containers request none.
    one.env["CUDA_VISIBLE_DEVICES"] = ""
    for slot in range(_SESSIONS):
        one.start_session(slot)
    for slot in range(_SESSIONS):
        one.wait_healthy(slot)
    received = await asyncio.gather(*(_stream(one, slot) for slot in range(_SESSIONS)))
    assert all(count >= 60 for count in received), received
    assert one.worker is not None
    apps = _gpu_apps(gpu)
    assert list(apps) == [one.worker.pid], apps
    assert apps[one.worker.pid] >= 2048
    # A worker restart reloads the one copy; the sessions recover in place.
    pids = [p.pid for p in one.sessions if p is not None]
    one.worker.send_signal(signal.SIGKILL)
    one.worker.wait(30)
    await _until(lambda: one.health(0)[0] == 503, what="slot 0 not ready")
    one.env["CUDA_VISIBLE_DEVICES"] = gpu
    one.start_worker()
    one.wait_worker(within=300)
    for slot in range(_SESSIONS):
        one.wait_healthy(slot, within=60)
    received = await asyncio.gather(*(_stream(one, slot) for slot in range(_SESSIONS)))
    assert all(count >= 60 for count in received), received
    assert list(_gpu_apps(gpu)) == [one.worker.pid]
    assert [p.pid for p in one.sessions if p is not None] == pids
