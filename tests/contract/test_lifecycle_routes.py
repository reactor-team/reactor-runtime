"""Lock the session-lifecycle routes: paths, status codes, and the descriptor.

A consumer drives the session over ``POST /start_session`` / ``POST
/stop_session`` and branches on the status codes: ``200`` success, ``409``
wrong-state conflict, and the two-way ``503`` split — a ``Retry-After`` header
means the model is still loading (retry), no header means the process is
terminated (give up). The ``/start_session`` response body is the session
descriptor whose ``capabilities`` and ``recording`` blocks the consumer reads.
"""

from __future__ import annotations

from contract_helpers import (
    FIXED_SESSION_ID,
    Harness,
    UnloadableModel,
    running_runtime,
)


async def test_start_session_returns_the_descriptor(harness: Harness) -> None:
    response = await harness.client.post("/start_session", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == FIXED_SESSION_ID
    assert body["state"] == "waiting"
    assert body["model"]["name"]
    capabilities = body["capabilities"]
    assert capabilities["protocol_version"] == "v0"
    assert capabilities["commands"] == []
    # Track directions are client-perspective: a track the model emits is one
    # the client receives (recvonly), an input track is one it sends (sendonly).
    assert {"name": "main", "kind": "video", "direction": "recvonly"} in capabilities["tracks"]
    assert {"name": "webcam", "kind": "video", "direction": "sendonly"} in capabilities["tracks"]
    assert body["recording"] == {"enabled": False, "chunk_seconds": 4}


async def test_get_session_reports_the_current_state(harness: Harness) -> None:
    response = await harness.client.get("/session")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == FIXED_SESSION_ID
    assert body["state"] == "ready"


async def test_start_while_the_model_loads_is_503_with_retry_after() -> None:
    async with running_runtime(start=False) as harness:
        response = await harness.client.post("/start_session", json={})

    assert response.status_code == 503
    # The boot window: a Retry-After header tells the caller to try again.
    assert "retry-after" in response.headers


async def test_start_on_a_terminated_process_is_503_without_retry_after() -> None:
    async with running_runtime(model_cls=UnloadableModel) as harness:
        response = await harness.client.post("/start_session", json={})

    assert response.status_code == 503
    # No Retry-After: the split a consumer reads as "terminated, do not retry".
    assert "retry-after" not in response.headers


async def test_starting_twice_conflicts(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await harness.client.post("/start_session", json={})

    assert response.status_code == 409


async def test_stop_returns_200(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await harness.client.post("/stop_session")

    assert response.status_code == 200


async def test_stopping_with_nothing_running_conflicts(harness: Harness) -> None:
    response = await harness.client.post("/stop_session")

    assert response.status_code == 409


async def test_health_reports_status_state_and_detail(harness: Harness) -> None:
    response = await harness.client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "state", "detail"}
    assert body["status"] == "healthy"
    assert body["state"] == "available"


async def test_health_is_200_and_loading_while_the_model_loads() -> None:
    async with running_runtime(start=False) as harness:
        response = await harness.client.get("/health")

    # A loading model is not broken: the verdict stays healthy and the
    # lifecycle word alone says the process cannot open a session yet.
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["state"] == "loading"


async def test_health_is_serving_while_a_session_is_open(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await harness.client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["state"] == "serving"


async def test_health_is_503_once_terminated() -> None:
    async with running_runtime(model_cls=UnloadableModel) as harness:
        response = await harness.client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert set(body) == {"status", "state", "detail"}
    assert body["status"] == "unhealthy"
    assert body["state"] == "terminated"


async def test_moderated_stop_returns_200(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    response = await harness.client.post("/stop_session", json={"moderate": True})

    assert response.status_code == 200


async def test_moderated_stop_with_nothing_running_conflicts(harness: Harness) -> None:
    response = await harness.client.post("/stop_session", json={"moderate": True})

    assert response.status_code == 409
