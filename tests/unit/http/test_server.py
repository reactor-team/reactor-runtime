import asyncio
import socket

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reactor_runtime.core import Health, HealthStatus, RuntimeConfig
from reactor_runtime.http import HttpServer
from reactor_runtime.http.server import _HTTP_SHUTDOWN_SECONDS
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.runner.runner import Runner
from reactor_runtime.transport.router import SessionControl, TransportRouter


class FakeRouter(TransportRouter):
    """A transport router that records its mount and adds one route."""

    protocol = "fake"
    version = "0.1"

    def __init__(self) -> None:
        self.mounted_with: SessionControl | None = None

    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        self.mounted_with = runner

        @app.get("/sessions/{sid}/transport/fake/ping")
        async def ping(sid: str) -> dict[str, str]:
            return {"sid": sid}


def _runner() -> Runner:
    return Runner(RuntimeConfig(model_ref="fake:Model"))


def _metrics() -> RuntimeMetrics:
    return RuntimeMetrics(version="0.0.0", model="fake:Model")


def test_mounts_route_groups_and_each_transport() -> None:
    runner = _runner()
    fake = FakeRouter()

    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model"), runner, [fake], Health.healthy, _metrics()
    )

    assert fake.mounted_with is runner
    paths = {getattr(route, "path", "") for route in server._app.routes}
    assert {"/start_session", "/stop_session", "/session", "/events", "/health"} <= paths
    assert "/sessions/{sid}/transport/fake/ping" in paths
    assert runner.descriptor()["transports"] == [{"protocol": "fake", "version": "0.1"}]


def test_cors_preflight_is_answered() -> None:
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model"), _runner(), [], Health.healthy, _metrics()
    )
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
        _metrics(),
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


async def test_serves_a_connection_queued_on_a_listener_bound_before_it_started() -> None:
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=port),
        _runner(),
        [],
        Health.healthy,
        _metrics(),
        listeners=[listener],
    )
    # A probe that connects before the server runs waits in the backlog.
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET /health HTTP/1.1\r\nHost: runtime\r\nConnection: close\r\n\r\n")
    await writer.drain()

    await server.start()
    try:
        response = await asyncio.wait_for(reader.read(), timeout=5.0)
        assert response.startswith(b"HTTP/1.1 200")
    finally:
        writer.close()
        await server.drain()
        await asyncio.wait_for(server.stop(), timeout=5.0)


async def test_stop_swallows_a_failed_serve_task() -> None:
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=0),
        _runner(),
        [],
        Health.healthy,
        _metrics(),
    )

    async def boom() -> None:
        raise RuntimeError("serve crashed")

    server._serve_task = asyncio.create_task(boom())
    # A serve task that fails during shutdown is logged, not raised, so a late
    # serve error cannot abort the ordered teardown of the rest of the service.
    await server.stop()


async def test_graceful_shutdown_is_bounded_short_whatever_the_grace_period() -> None:
    # The session drain is the runner's and runs before this server stops, so
    # the requests still open here are long-lived subscriptions that are cut,
    # not waited on: the grace period must not reach uvicorn.
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model", host="127.0.0.1", port=0, grace_period=60.0),
        _runner(),
        [],
        Health.healthy,
        _metrics(),
    )

    await server.start()
    try:
        assert server._server is not None
        assert server._server.config.timeout_graceful_shutdown == _HTTP_SHUTDOWN_SECONDS
        assert _HTTP_SHUTDOWN_SECONDS <= 2
    finally:
        await server.drain()
        await asyncio.wait_for(server.stop(), timeout=5.0)


def test_a_router_that_names_no_transport_is_served_but_not_listed() -> None:
    class Unnamed(TransportRouter):
        """A router written before routers named their transport."""

        def mount(self, app: FastAPI, runner: SessionControl) -> None:
            @app.get("/sessions/{sid}/transport/unnamed/ping")
            async def ping(sid: str) -> dict[str, str]:
                return {"sid": sid}

    runner = _runner()
    server = HttpServer(
        RuntimeConfig(model_ref="fake:Model"), runner, [Unnamed()], Health.healthy, _metrics()
    )

    paths = {getattr(route, "path", "") for route in server._app.routes}
    assert "/sessions/{sid}/transport/unnamed/ping" in paths
    assert runner.descriptor()["transports"] == []
