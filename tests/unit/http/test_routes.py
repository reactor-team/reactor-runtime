import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request

from reactor_runtime import InputField, Output, ReactorModel, Video, event
from reactor_runtime.core import Health, HealthStatus, RuntimeConfig
from reactor_runtime.http import EgressRoutes, RecordingRoutes, SessionRoutes, UploadRoutes
from reactor_runtime.http.routes import _read_capped, _resume_from, _stream_events
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.runner.runner import SESSION_ID, Runner

_RECORDING_ID = "00000000-0000-0000-0000-000000000001"


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


def _metrics() -> RuntimeMetrics:
    return RuntimeMetrics(version="1.2.3", model="fake:Model")


def _app(
    runner: Runner,
    process_health: Callable[[], Health] | None = None,
    metrics: RuntimeMetrics | None = None,
) -> FastAPI:
    app = FastAPI()
    SessionRoutes(runner).mount(app)
    EgressRoutes(runner, process_health or runner.health, metrics or _metrics()).mount(app)
    UploadRoutes(runner).mount(app)
    RecordingRoutes(runner).mount(app)
    return app


def _seed_recording(runner: Runner, root: Path, *segments: str) -> Path:
    """Point the recorder's serving root at *root* and write fake segments."""
    runner.recorder._root = root
    session_dir = root / _RECORDING_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    for name in segments:
        (session_dir / name).write_bytes(b"data")
    return session_dir


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(FakeModel)


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


async def test_start_session_conflicts_when_already_running(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post("/start_session", json={})

    assert response.status_code == 409


async def test_stop_session_conflicts_when_nothing_is_running(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client

    response = await http_client.post("/stop_session")

    assert response.status_code == 409


async def test_start_session_is_unavailable_while_the_model_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    # Constructed but never started, so the session sits in CREATED (model not loaded).
    runner = Runner(RuntimeConfig(model_ref="fake:Model"))
    transport = httpx.ASGITransport(app=_app(runner))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post("/start_session", json={})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"


async def test_moderated_stop_returns_ok(client: tuple[httpx.AsyncClient, Runner]) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post("/stop_session", json={"moderate": True})

    assert response.status_code == 200


async def test_moderated_stop_ends_the_session_as_moderated(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, runner = client
    await http_client.post("/start_session", json={})

    await http_client.post("/stop_session", json={"moderate": True})

    transitions = [event.transition for _seq, event in runner.events._history]
    stops = [t for t in transitions if t.event.name.lower() == "stop_session"]
    assert stops
    assert stops[-1].detail["reason"] == "moderated"


async def test_stop_session_with_an_explicit_plain_body_stays_stopped(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, runner = client
    await http_client.post("/start_session", json={})

    response = await http_client.post("/stop_session", json={"moderate": False})

    assert response.status_code == 200
    transitions = [event.transition for _seq, event in runner.events._history]
    stops = [t for t in transitions if t.event.name.lower() == "stop_session"]
    assert stops[-1].detail["reason"] == "stopped"


async def test_moderated_stop_conflicts_when_nothing_is_running(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client

    response = await http_client.post("/stop_session", json={"moderate": True})

    assert response.status_code == 409


async def test_reasoned_stop_returns_ok(client: tuple[httpx.AsyncClient, Runner]) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post("/stop_session", json={"reason": "deployment"})

    assert response.status_code == 200


async def test_reasoned_stop_threads_the_close_reason_into_the_journal(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, runner = client
    await http_client.post("/start_session", json={})

    await http_client.post("/stop_session", json={"moderate": False, "reason": "deployment"})

    transitions = [event.transition for _seq, event in runner.events._history]
    stops = [t for t in transitions if t.event.name.lower() == "stop_session"]
    assert stops
    assert stops[-1].detail["reason"] == "stopped"
    assert stops[-1].detail["close_reason"] == "deployment"


async def test_stop_session_without_a_reason_journals_no_close_reason(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, runner = client
    await http_client.post("/start_session", json={})

    await http_client.post("/stop_session", json={"moderate": True})

    transitions = [event.transition for _seq, event in runner.events._history]
    stops = [t for t in transitions if t.event.name.lower() == "stop_session"]
    assert "close_reason" not in stops[-1].detail


async def test_reasoned_stop_conflicts_when_nothing_is_running(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client

    response = await http_client.post("/stop_session", json={"reason": "deployment"})

    assert response.status_code == 409


async def test_stop_session_rejects_an_oversized_reason(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.post("/stop_session", json={"reason": "x" * 65})

    assert response.status_code == 422


async def test_health_is_available_once_the_model_is_up(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "state": "available", "detail": None}


async def test_health_is_ok_and_loading_while_the_model_loads() -> None:
    # Constructed but never started, so the session sits in CREATED: healthy —
    # nothing is broken — with the lifecycle word saying it cannot serve yet.
    runner = Runner(RuntimeConfig(model_ref="fake:Model"))
    transport = httpx.ASGITransport(app=_app(runner))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "state": "loading", "detail": None}


async def test_health_is_serving_while_a_session_is_open(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})

    response = await http_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["state"] == "serving"


async def test_health_is_503_and_terminated_after_a_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnloadableModel(FakeModel):
        def load(self, config_path: Path | None) -> None:
            raise RuntimeError("weights missing")

    monkeypatch.setattr(
        "reactor_runtime.runner.runner.import_model_class", lambda ref: UnloadableModel
    )
    runner = Runner(RuntimeConfig(model_ref="fake:Model"))
    await runner.start()
    transport = httpx.ASGITransport(app=_app(runner))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["state"] == "terminated"
    assert body["detail"]


async def test_health_status_comes_from_the_injected_process_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The verdict is the injected process aggregate, not the runner's own
    # report: a broken sibling component turns /health unhealthy while the
    # state keeps reading the runner's lifecycle word.
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(RuntimeConfig(model_ref="fake:Model"))
    await runner.start()
    app = _app(runner, lambda: Health(HealthStatus.UNHEALTHY, "http server not started"))
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
            response = await http_client.get("/health")
    finally:
        await runner.stop()

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "state": "available",
        "detail": "http server not started",
    }


async def test_metrics_renders_the_injected_registry(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    response = await http_client.get("/metrics")

    assert response.status_code == 200
    # A scraper selects the parser from the media type, so the endpoint must
    # answer in the Prometheus text format rather than the JSON the rest of the
    # surface speaks.
    assert response.headers["content-type"].startswith("text/plain")
    assert 'runtime_info{model="fake:Model",version="1.2.3"} 1.0' in response.text


async def test_metrics_answers_before_the_model_loads() -> None:
    # Constructed but never started, so the model has not loaded. The scrape
    # still answers, which makes a slow or failing load observable instead of a
    # gap in the series.
    runner = Runner(RuntimeConfig(model_ref="fake:Model"))
    transport = httpx.ASGITransport(app=_app(runner))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.get("/metrics")

    assert response.status_code == 200
    assert "runtime_info" in response.text


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


async def test_create_upload_honours_a_supplied_id(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, runner = client
    await http_client.post("/start_session", json={})

    created = await http_client.post(
        f"/sessions/{SESSION_ID}/uploads",
        json={"name": "cat.png", "size": 4, "mime_type": "image/png", "upload_id": "platform-1"},
    )
    assert created.status_code == 201
    assert created.json()["presigned_id"] == "platform-1"

    await http_client.put("/uploads/platform-1", content=b"\x89PNG")
    assert (await runner.uploads.fetch("platform-1")).data == b"\x89PNG"


async def test_create_upload_rejects_a_reserved_id(
    client: tuple[httpx.AsyncClient, Runner],
) -> None:
    http_client, _ = client
    await http_client.post("/start_session", json={})
    payload = {"name": "cat.png", "size": 4, "mime_type": "image/png", "upload_id": "dup"}
    await http_client.post(f"/sessions/{SESSION_ID}/uploads", json=payload)

    response = await http_client.post(f"/sessions/{SESSION_ID}/uploads", json=payload)

    assert response.status_code == 409


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


class _StreamingRequest:
    """A stand-in request that yields a fixed sequence of body chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def stream(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


async def test_read_capped_returns_a_body_within_the_limit() -> None:
    request = cast(Request, _StreamingRequest([b"ab", b"cd"]))
    assert await _read_capped(request, 4) == b"abcd"


async def test_read_capped_aborts_once_the_limit_is_exceeded() -> None:
    # A chunked body with no honest Content-Length is stopped mid-stream rather
    # than buffered whole, so an oversized upload cannot exhaust memory.
    request = cast(Request, _StreamingRequest([b"abcd", b"e"]))
    with pytest.raises(HTTPException) as raised:
        await _read_capped(request, 4)
    assert raised.value.status_code == 400


async def test_clips_serves_a_ready_manifest(
    client: tuple[httpx.AsyncClient, Runner], tmp_path: Path
) -> None:
    http_client, runner = client
    _seed_recording(runner, tmp_path, "init.mp4", "chunk_00000.m4s", "chunk_00001.m4s")

    response = await http_client.get(f"/clips?session_id={_RECORDING_ID}&start=0&end=4")

    assert response.status_code == 200
    assert "#EXT-X-MAP" in response.text


async def test_clips_is_pending_until_the_boundary_lands(
    client: tuple[httpx.AsyncClient, Runner], tmp_path: Path
) -> None:
    http_client, runner = client
    _seed_recording(runner, tmp_path, "init.mp4", "chunk_00000.m4s")

    response = await http_client.get(f"/clips?session_id={_RECORDING_ID}&start=0&end=4")

    assert response.status_code == 202
    assert response.headers["Retry-After"] == "2"


async def test_clips_is_gone_for_an_unknown_recording(
    client: tuple[httpx.AsyncClient, Runner], tmp_path: Path
) -> None:
    http_client, runner = client
    runner.recorder._root = tmp_path

    response = await http_client.get(f"/clips?session_id={_RECORDING_ID}&start=0&end=4")

    assert response.status_code == 410


async def test_clips_rejects_a_bad_range(
    client: tuple[httpx.AsyncClient, Runner], tmp_path: Path
) -> None:
    http_client, runner = client
    _seed_recording(runner, tmp_path, "init.mp4", "chunk_00000.m4s", "chunk_00001.m4s")

    response = await http_client.get(f"/clips?session_id={_RECORDING_ID}&start=5&end=4")

    assert response.status_code == 400


async def test_clip_chunk_serves_a_segment(
    client: tuple[httpx.AsyncClient, Runner], tmp_path: Path
) -> None:
    http_client, runner = client
    _seed_recording(runner, tmp_path, "init.mp4")

    response = await http_client.get(f"/clips/chunks/{_RECORDING_ID}/init.mp4")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"


async def test_clip_chunk_rejects_a_non_artifact_name(
    client: tuple[httpx.AsyncClient, Runner], tmp_path: Path
) -> None:
    http_client, runner = client
    _seed_recording(runner, tmp_path, "init.mp4")

    response = await http_client.get(f"/clips/chunks/{_RECORDING_ID}/secrets.txt")

    assert response.status_code == 404


async def test_clip_chunk_is_gone_for_an_unknown_recording(
    client: tuple[httpx.AsyncClient, Runner], tmp_path: Path
) -> None:
    http_client, runner = client
    runner.recorder._root = tmp_path

    response = await http_client.get(f"/clips/chunks/{_RECORDING_ID}/init.mp4")

    assert response.status_code == 410


def test_resume_from_reads_a_numeric_last_event_id() -> None:
    assert _resume_from("7") == 7


def test_resume_from_ignores_a_missing_or_non_numeric_header() -> None:
    assert _resume_from(None) is None
    assert _resume_from("") is None
    assert _resume_from("not-a-number") is None
