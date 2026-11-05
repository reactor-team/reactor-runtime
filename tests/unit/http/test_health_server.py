"""A process that serves no sessions answers ``/health`` with the runtime's own health model."""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request

from reactor_runtime.core import Health, HealthStatus, RuntimeState
from reactor_runtime.http import HealthServer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(port: int) -> tuple[int, dict[str, object]]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_health_answers_the_verdict_the_state_and_why() -> None:
    report = [Health.healthy()]
    state = [RuntimeState.LOADING]
    port = _free_port()
    server = HealthServer(
        host="127.0.0.1",
        port=port,
        process_health=lambda: report[0],
        state=lambda: state[0],
        depends_on=("model_server",),
    )

    async def run() -> list[tuple[int, dict[str, object]]]:
        await server.start()
        try:
            seen = [await asyncio.to_thread(_get, port)]
            state[0] = RuntimeState.AVAILABLE
            seen.append(await asyncio.to_thread(_get, port))
            report[0] = Health(HealthStatus.UNHEALTHY, "stuck: one step has run for 130 s")
            seen.append(await asyncio.to_thread(_get, port))
            return seen
        finally:
            await server.drain()
            await server.stop()

    seen = asyncio.run(run())

    assert server.depends_on == ("model_server",)
    assert seen[0] == (200, {"status": "healthy", "state": "loading", "detail": None})
    assert seen[1] == (200, {"status": "healthy", "state": "available", "detail": None})
    assert seen[2] == (
        503,
        {
            "status": "unhealthy",
            "state": "available",
            "detail": "stuck: one step has run for 130 s",
        },
    )
