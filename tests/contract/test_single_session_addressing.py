"""The session-addressed routes on a runtime that hosts one session.

A single-session runtime answers the same routes a multi-session one does, so
one caller can drive either: the session id is optional where a runtime of one
session needs none, and accepted as the fixed transport id or the session's
own id where it is given.
"""

from __future__ import annotations

import pytest
from contract_helpers import Harness, read_sse, wait_for_state

from reactor_runtime.runner.runner import SESSION_ID

pytestmark = pytest.mark.asyncio


async def test_sessions_lists_the_running_session_and_the_capacity(harness: Harness) -> None:
    empty = (await harness.client.get("/sessions")).json()
    assert empty == {"capacity": {"slots": 1, "free": 1}, "sessions": []}

    await harness.client.post("/start_session", json={"session_id": "own-id"})

    listed = (await harness.client.get("/sessions")).json()
    assert listed["capacity"] == {"slots": 1, "free": 0}
    assert [entry["session_id"] for entry in listed["sessions"]] == [SESSION_ID]


async def test_sessions_lists_the_session_while_it_closes(harness: Harness) -> None:
    await harness.client.post("/start_session", json={})

    harness.runner.stop_session()

    assert [entry["state"] for entry in harness.runner.sessions()] == ["closing"]


async def test_session_reads_by_either_id_and_refuses_another(harness: Harness) -> None:
    await harness.client.post("/start_session", json={"session_id": "own-id"})

    for sid in (SESSION_ID, "own-id"):
        response = await harness.client.get("/session", params={"session_id": sid})
        assert response.status_code == 200
    assert (await harness.client.get("/session", params={"session_id": "x"})).status_code == 404


async def test_stop_takes_an_optional_session_id(harness: Harness) -> None:
    await harness.client.post("/start_session", json={"session_id": "own-id"})

    refused = await harness.client.post("/stop_session", json={"session_id": "other"})
    stopped = await harness.client.post("/stop_session", json={"session_id": "own-id"})

    assert refused.status_code == 404
    assert stopped.status_code == 200
    await wait_for_state(harness.runner, "ready")


async def test_events_name_the_session_and_its_stream_filters_to_it(harness: Harness) -> None:
    await harness.client.post("/start_session", json={"session_id": "first"})
    await harness.client.post("/stop_session")
    await wait_for_state(harness.runner, "ready")
    await harness.client.post("/start_session", json={"session_id": "second"})

    first = await read_sse(harness.app, "/sessions/first/events?since=0", count=3)
    live = await read_sse(harness.app, f"/sessions/{SESSION_ID}/events?since=0", count=1)

    assert [frame.payload["event"] for frame in first] == [
        "start_session",
        "stop_session",
        "cleanup_complete",
    ]
    assert {frame.payload["session_id"] for frame in first} == {"first"}
    assert live[0].payload["session_id"] == "second"
