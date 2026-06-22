import asyncio

from fastapi import FastAPI

from reactor_runtime.core import HealthStatus, RuntimeConfig
from reactor_runtime.http import HttpServer
from reactor_runtime.runner.runner import Runner
from reactor_runtime.transport.router import SessionControl, TransportRouter


class FakeRouter(TransportRouter):
    """A transport router that records its mount and adds one route."""

    def __init__(self) -> None:
        self.mounted_with: SessionControl | None = None

    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        self.mounted_with = runner

        @app.get("/sessions/{sid}/transport/fake/ping")
        async def ping(sid: str) -> dict[str, str]:
            return {"sid": sid}


def _runner() -> Runner:
    return Runner(RuntimeConfig(model_ref="fake:Model"))


def test_mounts_route_groups_and_each_transport() -> None:
    runner = _runner()
    fake = FakeRouter()

    server = HttpServer(RuntimeConfig(model_ref="fake:Model"), runner, [fake])

    assert fake.mounted_with is runner
    paths = {getattr(route, "path", "") for route in server._app.routes}
    assert {"/start_session", "/stop_session", "/session", "/events", "/health"} <= paths
    assert "/sessions/{sid}/transport/fake/ping" in paths


async def test_start_drain_stop_lifecycle() -> None:
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=0), _runner(), []
    )

    await server.start()
    try:
        for _ in range(200):
            if server._server is not None and server._server.started:
                break
            await asyncio.sleep(0.01)
        assert server._server is not None
        assert server._server.started
        assert server.health().status is HealthStatus.HEALTHY
    finally:
        await server.drain()
        assert server._server is not None
        assert server._server.should_exit is True
        await asyncio.wait_for(server.stop(), timeout=5.0)
