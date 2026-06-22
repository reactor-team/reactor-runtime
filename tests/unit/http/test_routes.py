import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from reactor_runtime.core import RuntimeConfig
from reactor_runtime.http import EgressRoutes, SessionRoutes
from reactor_runtime.http.routes import _stream_events
from reactor_runtime.model import InputField, event
from reactor_runtime.model.reactor_model import ReactorModel
from reactor_runtime.model.tracks import Output, Video
from reactor_runtime.runner.runner import SESSION_ID, Runner


class FakeOut(Output):
    main: Video


class FakeModel(ReactorModel):
    """A minimal model with one track and one command that idles when run."""

    output: FakeOut

    @event(name="set_mode")
    async def set_mode(self, mode: str = InputField(min_length=1)) -> None: ...

    def load(self, config: dict[str, Any]) -> None: ...

    async def run(self) -> None:
        await asyncio.sleep(60)


def _app(runner: Runner) -> FastAPI:
    app = FastAPI()
    SessionRoutes(runner).mount(app)
    EgressRoutes(runner).mount(app)
    return app


@pytest.fixture
async def client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, Runner]]:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(RuntimeConfig(model_ref="fake:Model"))
    await runner.start()
    transport = httpx.ASGITransport(app=_app(runner))
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client, runner
    finally:
        await runner.stop()


async def test_start_session_returns_the_descriptor(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.post("/start_session", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "waiting"
    assert body["session_id"]
    assert "main" in body["tracks"]
    assert "paths" in body["schema"]


async def test_get_session_reports_the_current_state(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.get("/session")

    assert response.status_code == 200
    assert response.json()["state"] == "ready"


async def test_stop_session_closes(client: tuple[httpx.AsyncClient, Runner]) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post("/stop_session")

    assert response.status_code == 200


async def test_enforce_returns_ok(client: tuple[httpx.AsyncClient, Runner]) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post(f"/sessions/{SESSION_ID}/enforce", json={"block": True})

    assert response.status_code == 200


async def test_enforce_rejects_an_unknown_sid(client: tuple[httpx.AsyncClient, Runner]) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post("/sessions/not-the-session/enforce", json={"block": True})

    assert response.status_code == 404


async def test_enforce_rejects_when_no_session_is_running(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client

    response = await http_client.post(f"/sessions/{SESSION_ID}/enforce", json={"block": True})

    assert response.status_code == 400


async def test_health_is_ok_once_the_model_is_up(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


async def test_events_replays_the_backlog_as_sse(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    # The egress stream is unbounded, so drive the generator directly and read
    # the first replayed message rather than consuming an endless HTTP body.
    _, runner = client
    stream = _stream_events(runner, 0)
    try:
        message = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert message.startswith("id: 1\n")
        body = json.loads(message.split("data: ", 1)[1].strip())
        assert body["type"] == "transition"
        assert body["to"] == "ready"
    finally:
        await stream.aclose()
