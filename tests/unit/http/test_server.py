import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reactor_runtime.core import Health, HealthStatus, RuntimeConfig
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

    server = HttpServer(RuntimeConfig(model_ref="fake:Model"), runner, [fake], Health.healthy)

    assert fake.mounted_with is runner
    paths = {getattr(route, "path", "") for route in server._app.routes}
    assert {"/start_session", "/stop_session", "/session", "/events", "/health"} <= paths
    assert "/sessions/{sid}/transport/fake/ping" in paths


def test_cors_preflight_is_answered() -> None:
    server = HttpServer(RuntimeConfig(model_ref="fake:Model"), _runner(), [], Health.healthy)
    client = TestClient(server._app)

    response = client.options(
        "/start_session",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


async def test_start_drain_stop_lifecycle() -> None:
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=0),
        _runner(),
        [],
        Health.healthy,
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


async def test_stop_swallows_a_failed_serve_task() -> None:
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=0),
        _runner(),
        [],
        Health.healthy,
    )

    async def boom() -> None:
        raise RuntimeError("serve crashed")

    server._serve_task = asyncio.create_task(boom())
    # A serve task that fails during shutdown is logged, not raised, so a late
    # serve error cannot abort the ordered teardown of the rest of the service.
    await server.stop()


async def test_graceful_shutdown_is_bounded_by_the_grace_period() -> None:
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=0, grace_period=7.0),
        _runner(),
        [],
        Health.healthy,
    )

    await server.start()
    try:
        assert server._server is not None
        assert server._server.config.timeout_graceful_shutdown == 7
    finally:
        await server.drain()
        await asyncio.wait_for(server.stop(), timeout=5.0)
