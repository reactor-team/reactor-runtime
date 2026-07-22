"""Lock the journal facts the runner emits end to end, with their detail.

The vocabulary tests pin what the state machine can say; these pin what the
running system actually says on its public faces — the transitions a consumer
mirroring a real session observes for boot, connection churn, teardown, a
timeout, a moderation stop, and a crash, each with the ``detail`` keys it
branches on.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from contract_helpers import (
    ContractModel,
    CrashingModel,
    FakeConnection,
    Harness,
    JournalReader,
    UnloadableModel,
    running_runtime,
)

from reactor_runtime.core import ConnId, RuntimeConfig

_PLATFORM_SESSION_ID = "7d9f5c1e-1111-2222-3333-444444444444"


def _capture_recorder_starts(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Intercept the recorder's start to observe the id a session records under."""
    captured: list[str] = []

    def capture(session_id: str) -> None:
        captured.append(session_id)

    monkeypatch.setattr(harness.runner.recorder, "start", capture)
    return captured


async def test_the_connection_lifecycle_journals_each_move_with_its_detail(
    harness: Harness,
) -> None:
    journal = JournalReader(harness.runner)
    try:
        boot = await journal.expect("initialization_success")
        assert (boot["from"], boot["to"]) == ("created", "ready")

        harness.runner.start_session({"session_id": _PLATFORM_SESSION_ID})
        started = await journal.expect("start_session")
        assert (started["from"], started["to"]) == ("ready", "waiting")
        # The start parameters ride the transition verbatim.
        assert started["detail"] == {"params": {"session_id": _PLATFORM_SESSION_ID}}

        answer = {"type": "answer", "sdp": "v=0"}
        harness.runner.connection_answered(ConnId(1002), answer)
        answered = await journal.expect("connection_answered")
        assert (answered["from"], answered["to"]) == ("waiting", "waiting")
        assert answered["detail"] == {"conn_id": 1002, "answer": answer}

        harness.runner.connection_opened(FakeConnection(1002))
        first = await journal.expect("connection_opened")
        assert (first["from"], first["to"]) == ("waiting", "streaming")
        assert first["detail"] == {"conn_id": 1002}

        harness.runner.connection_opened(FakeConnection(1003))
        second = await journal.expect("connection_opened")
        assert (second["from"], second["to"]) == ("streaming", "streaming")
        assert second["detail"] == {"conn_id": 1003}

        harness.runner.connection_closed(ConnId(1003))
        closed = await journal.expect("connection_closed")
        assert (closed["from"], closed["to"]) == ("streaming", "streaming")
        assert closed["detail"] == {"conn_id": 1003}

        harness.runner.connection_closed(ConnId(1002))
        orphaned = await journal.expect("connection_closed")
        assert (orphaned["from"], orphaned["to"]) == ("streaming", "orphaned")
        assert orphaned["detail"] == {"conn_id": 1002}

        harness.runner.stop_session()
        stopped = await journal.expect("stop_session")
        assert (stopped["from"], stopped["to"]) == ("orphaned", "closing")
        assert stopped["detail"] == {"reason": "stopped"}

        cleaned = await journal.expect("cleanup_complete")
        assert (cleaned["from"], cleaned["to"]) == ("closing", "ready")
        assert cleaned["detail"] == {"reason": "stopped"}
    finally:
        await journal.aclose()


async def test_a_failed_model_load_journals_initialization_fail() -> None:
    async with running_runtime(model_cls=UnloadableModel) as harness:
        journal = JournalReader(harness.runner)
        try:
            failed = await journal.expect("initialization_fail")
            assert (failed["from"], failed["to"]) == ("created", "terminated")
        finally:
            await journal.aclose()


async def test_a_run_loop_crash_journals_a_terminal_eviction_with_the_error() -> None:
    async with running_runtime(model_cls=CrashingModel) as harness:
        journal = JournalReader(harness.runner)
        try:
            evicted = await journal.expect("eviction")
            assert evicted["to"] == "terminated"
            assert evicted["detail"]["reason"] == "error"
            assert "model exploded" in evicted["detail"]["error"]
        finally:
            await journal.aclose()


async def test_a_clientless_session_times_out_and_unwinds() -> None:
    cfg = RuntimeConfig(model_ref="contract:Model", orphan_timeout=0.05)
    async with running_runtime(model_cls=ContractModel, cfg=cfg) as harness:
        journal = JournalReader(harness.runner)
        try:
            harness.runner.start_session({})
            timed_out = await journal.expect("timeout")
            assert (timed_out["from"], timed_out["to"]) == ("waiting", "closing")
            assert timed_out["detail"] == {"reason": "timed_out"}

            cleaned = await journal.expect("cleanup_complete")
            assert cleaned["detail"] == {"reason": "timed_out"}
        finally:
            await journal.aclose()


async def test_a_moderated_stop_journals_the_session_as_moderated(harness: Harness) -> None:
    journal = JournalReader(harness.runner)
    try:
        await harness.client.post("/start_session", json={})
        response = await harness.client.post("/stop_session", json={"moderate": True})
        assert response.status_code == 200

        stopped = await journal.expect("stop_session")
        assert stopped["detail"] == {"reason": "moderated"}
    finally:
        await journal.aclose()


async def test_the_recording_is_addressed_by_the_start_session_id(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_recorder_starts(harness, monkeypatch)

    await harness.client.post("/start_session", json={"session_id": _PLATFORM_SESSION_ID})
    await asyncio.sleep(0)

    # The recorder stores and serves the session under the id the caller
    # supplied, so /clips and chunk_ready line up with the caller's id space.
    assert captured == [_PLATFORM_SESSION_ID]


async def test_a_session_without_an_id_records_under_a_fresh_uuid(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_recorder_starts(harness, monkeypatch)

    await harness.client.post("/start_session", json={})
    await asyncio.sleep(0)

    assert len(captured) == 1
    minted = captured[0]
    assert minted != "00000000-0000-0000-0000-000000000000"
    assert uuid.UUID(minted)
