"""Three step sessions served at once by one runtime process, over HTTP.

Boots the runtime the way ``reactor run`` does, on the batch example, and
starts three step sessions concurrently. Each session's frames are compared,
by the fingerprint the example keeps with every step, with the frames the
same seed yields in a session that runs alone: in the same process, and in a
process capped at one session.

Requires the native ``reactor_webrtc`` wheel the runtime's entry point loads;
the module skips cleanly when it is absent.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

pytest.importorskip("reactor_webrtc")

pytestmark = pytest.mark.asyncio

_EXAMPLE = Path(__file__).parents[2] / "examples" / "batch"
_STEPS = 6
_SEEDS = (11, 22, 33)

_MANIFEST = """\
$schema: reactor/v2
model:
  name: batch
runtime:
  import: batch:Batch
  config: config.yaml
  step_results:
    enabled: true
    queue: 256
"""

_CONFIG = """\
width: 64
height: 64
latent: 16
channels: 8
layers: 2
backend: numpy
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _runtime(workspace: Path, *, max_sessions: int | None = None) -> Iterator[str]:
    """Run the runtime in *workspace* and yield its base URL once it is available."""
    port = _free_port()
    env = {
        **os.environ,
        "PORT": str(port),
        "HOST": "127.0.0.1",
        "PYTHONPATH": str(_EXAMPLE),
        "REACTOR_LOG_LEVEL": "WARNING",
    }
    if max_sessions is not None:
        env["REACTOR_MAX_SESSIONS"] = str(max_sessions)
    process = subprocess.Popen(
        [sys.executable, "-c", "from reactor_runtime.serve import main; main()"],
        cwd=workspace,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                assert process.stderr is not None
                raise AssertionError(f"runtime exited: {process.stderr.read().decode()[-2000:]}")
            try:
                if httpx.get(f"{base}/health", timeout=1).json()["state"] == "available":
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise AssertionError("runtime did not become available")
            time.sleep(0.1)
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


async def _run_session(client: httpx.AsyncClient, seed: int) -> list[str]:
    """Run one step session with *seed* and return its frames' fingerprints, in order."""
    sid = str(uuid.uuid4())
    started = await client.post(
        "/start_session",
        json={"session_id": sid, "steps": _STEPS, "starting_input": {"state": {"seed": seed}}},
    )
    assert started.status_code == 200, started.text
    assert started.json()["session_id"] in (sid, "00000000-0000-0000-0000-000000000000")
    async with asyncio.timeout(60):
        while True:
            # A model keeps stepping until the session's end reaches it, so a
            # step past the count may be kept too.
            listed = (await client.get(f"/sessions/{sid}/steps")).json()
            if set(range(1, _STEPS + 1)) <= set(listed.get("steps", [])):
                break
            await asyncio.sleep(0.05)
    digests = []
    for step in range(1, _STEPS + 1):
        response = await client.get(f"/sessions/{sid}/steps/{step}/frame.sha256")
        assert response.status_code == 200
        digests.append(response.text)
    async with asyncio.timeout(30):
        while True:
            if not (await client.get("/sessions")).json()["sessions"]:
                break
            await asyncio.sleep(0.05)
    return digests


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "reactor.yaml").write_text(_MANIFEST)
    (tmp_path / "config.yaml").write_text(_CONFIG)
    return tmp_path


async def test_concurrent_sessions_each_get_the_frames_they_get_alone(workspace: Path) -> None:
    with _runtime(workspace) as base:
        async with httpx.AsyncClient(base_url=base, timeout=10) as client:
            health = (await client.get("/health")).json()
            assert health["capacity"] == {"slots": 4, "free": 4}

            together = await asyncio.gather(*(_run_session(client, seed) for seed in _SEEDS))
            alone = [await _run_session(client, seed) for seed in _SEEDS]

    assert len({tuple(digests) for digests in together}) == len(_SEEDS)
    assert together == alone

    with _runtime(workspace, max_sessions=1) as base:
        async with httpx.AsyncClient(base_url=base, timeout=10) as client:
            health = (await client.get("/health")).json()
            assert health["capacity"] == {"slots": 1, "free": 1}
            single = await _run_session(client, _SEEDS[0])

    assert single == together[0]
