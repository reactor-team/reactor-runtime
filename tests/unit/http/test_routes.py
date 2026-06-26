import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from reactor_runtime import InputField, Output, ReactorModel, Video, event
from reactor_runtime.core import RuntimeConfig
from reactor_runtime.http import EgressRoutes, SessionRoutes, UploadRoutes
from reactor_runtime.http.routes import _resume_from, _stream_events
from reactor_runtime.runner.runner import SESSION_ID, Runner


class FakeOut(Output):
    main: Video


class FakeModel(ReactorModel):
    """A minimal model with one track and one command that idles when run."""

    output: FakeOut

    @event(name="set_mode")
    async def set_mode(self, mode: str = InputField(min_length=1)) -> None: ...

    def load(self, config_path: Path | None) -> None: ...

    async def run(self) -> None:
        await asyncio.sleep(60)


def _app(runner: Runner) -> FastAPI:
    app = FastAPI()
    SessionRoutes(runner).mount(app)
    EgressRoutes(runner).mount(app)
    UploadRoutes(runner).mount(app)
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


async def test_start_session_serves_the_capabilities_descriptor(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.post("/start_session", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "waiting"
    assert body["session_id"]
    assert body["cluster"] == "local"
    assert body["model"]["name"]
    assert body["server_info"]["server_version"]
    caps = body["capabilities"]
    assert caps["protocol_version"] == "v0"
    assert {"name": "main", "kind": "video", "direction": "recvonly"} in caps["tracks"]
    # Commands are served at /schema, not carried on the descriptor.
    assert caps["commands"] == []
    assert "schema" not in body


async def test_get_session_serves_the_capabilities_descriptor(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.get("/session")

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "ready"
    assert body["cluster"] == "local"
    assert "capabilities" in body


async def test_schema_serves_the_model_contract(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.get("/schema")

    assert response.status_code == 200
    body = response.json()
    assert body  # the model is loaded, so the contract is non-empty
    assert "set_mode" in json.dumps(body)


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


async def test_create_upload_allocates_a_slot(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post(
        f"/sessions/{SESSION_ID}/uploads",
        json={"name": "cat.png", "size": 4, "mime_type": "image/png"},
    )

    assert response.status_code == 201
    body = response.json()
    upload_id = body["presigned_id"]
    assert upload_id
    assert body["presigned_url"].endswith(f"/uploads/{upload_id}")
    assert body["path"] == f"sessions/{SESSION_ID}/uploads/{upload_id}/cat.png"


async def test_create_upload_rejects_an_unknown_sid(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post(
        "/sessions/not-the-session/uploads",
        json={"name": "cat.png", "size": 4, "mime_type": "image/png"},
    )

    assert response.status_code == 404


async def test_create_upload_rejects_a_non_positive_size(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post(
        f"/sessions/{SESSION_ID}/uploads",
        json={"name": "cat.png", "size": 0, "mime_type": "image/png"},
    )

    assert response.status_code == 400


async def test_put_upload_stores_the_bytes(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, runner = client
    await http_client.post("/start_session", json={})
    created = await http_client.post(
        f"/sessions/{SESSION_ID}/uploads",
        json={"name": "cat.png", "size": 4, "mime_type": "image/png"},
    )
    upload_id = created.json()["presigned_id"]

    response = await http_client.put(f"/uploads/{upload_id}", content=b"\x89PNG")

    assert response.status_code == 200
    assert (await runner.uploads.fetch(upload_id)).data == b"\x89PNG"


async def test_put_upload_to_an_unknown_slot_is_not_found(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.put("/uploads/nope", content=b"data")
    assert response.status_code == 404


async def test_put_upload_twice_conflicts(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})
    created = await http_client.post(
        f"/sessions/{SESSION_ID}/uploads",
        json={"name": "cat.png", "size": 4, "mime_type": "image/png"},
    )
    upload_id = created.json()["presigned_id"]
    await http_client.put(f"/uploads/{upload_id}", content=b"\x89PNG")

    response = await http_client.put(f"/uploads/{upload_id}", content=b"\x89PNG")

    assert response.status_code == 409


async def test_put_upload_with_a_size_mismatch_is_rejected(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})
    created = await http_client.post(
        f"/sessions/{SESSION_ID}/uploads",
        json={"name": "cat.png", "size": 4, "mime_type": "image/png"},
    )
    upload_id = created.json()["presigned_id"]

    response = await http_client.put(f"/uploads/{upload_id}", content=b"xx")

    assert response.status_code == 400


async def test_put_upload_rejects_an_oversized_content_length(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})
    created = await http_client.post(
        f"/sessions/{SESSION_ID}/uploads",
        json={"name": "cat.png", "size": 4, "mime_type": "image/png"},
    )
    upload_id = created.json()["presigned_id"]

    # The body's Content-Length (18) does not match the slot's declared size (4),
    # so the write is rejected from the header before the body is buffered.
    response = await http_client.put(f"/uploads/{upload_id}", content=b"way-too-many-bytes")

    assert response.status_code == 400


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


def test_resume_from_reads_a_numeric_last_event_id() -> None:
    assert _resume_from("7") == 7


def test_resume_from_ignores_a_missing_or_non_numeric_header() -> None:
    assert _resume_from(None) is None
    assert _resume_from("") is None
    assert _resume_from("not-a-number") is None
