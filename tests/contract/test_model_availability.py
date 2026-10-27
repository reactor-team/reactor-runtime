"""Lock what a model reporting itself unavailable does to the session surface.

A model whose work runs somewhere it does not own (a model worker in another
process) calls ``set_unavailable(reason)`` when that goes away: the live
session stops with ``error`` and the reason, ``/health`` answers 503 with the
reason, ``/start_session`` answers 503, and the process stays up. Once the
model calls ``set_available()`` everything serves as before.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np
import pytest
from contract_helpers import Harness, JournalReader, SteppingOutput, running_runtime, wait_for_state

from reactor_runtime import ModelUnavailable, ReactorApp

_LOADED: list[ReactorApp] = []


class DependentModel(ReactorApp):
    """A model on the default step loop whose steps fail while its dependency is gone."""

    def load(self, config_path: Path | None) -> None:
        self.gone = threading.Event()
        self.steps = 0
        _LOADED.append(self)

    def generate(self, input: object) -> SteppingOutput:
        if self.gone.is_set():
            raise ModelUnavailable("worker_died: gone")
        self.steps += 1
        return SteppingOutput(frames=np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    _LOADED.clear()
    async with running_runtime(DependentModel) as running:
        yield running


def _model() -> DependentModel:
    model = _LOADED[-1]
    assert isinstance(model, DependentModel)
    return model


async def _from_thread(call: Callable[[], object]) -> None:
    thread = threading.Thread(target=call)
    thread.start()
    await asyncio.to_thread(thread.join)


async def test_an_unavailable_model_stops_the_live_session_with_its_reason(
    harness: Harness,
) -> None:
    journal = JournalReader(harness.runner)
    try:
        assert (await harness.client.post("/start_session", json={})).status_code == 200
        model = _model()
        model.gone.set()
        await _from_thread(lambda: model.set_unavailable("worker_died: socket closed"))
        error = await journal.expect("error")
        assert error["detail"] == {"message": "worker_died: socket closed"}
        stopped = await journal.expect("stop_session")
        assert stopped["detail"] == {
            "reason": "error",
            "close_reason": "worker_died: socket closed",
        }
        await wait_for_state(harness.runner, "ready")
    finally:
        await journal.aclose()


async def test_health_and_start_answer_503_until_the_model_is_available(
    harness: Harness,
) -> None:
    model = _model()
    await _from_thread(lambda: model.set_unavailable("worker_died: socket closed"))
    await asyncio.sleep(0.05)
    health = await harness.client.get("/health")
    assert health.status_code == 503
    assert health.json()["detail"] == "worker_died: socket closed"
    assert health.json()["state"] == "available"
    started = await harness.client.post("/start_session", json={})
    assert started.status_code == 503
    assert started.json()["detail"] == "worker_died: socket closed"
    assert started.headers["retry-after"] == "1"

    await _from_thread(model.set_available)
    await asyncio.sleep(0.05)
    assert (await harness.client.get("/health")).status_code == 200
    assert (await harness.client.post("/start_session", json={})).status_code == 200


async def test_a_repeat_report_updates_the_reason_and_stops_nothing_more(
    harness: Harness,
) -> None:
    model = _model()
    journal = JournalReader(harness.runner)
    try:
        await _from_thread(lambda: model.set_unavailable("worker_unavailable: connecting"))
        await journal.expect("error")
        await _from_thread(lambda: model.set_unavailable("worker_died: refused"))
        await asyncio.sleep(0.05)
        assert (await harness.client.get("/health")).json()["detail"] == "worker_died: refused"
    finally:
        await journal.aclose()


async def test_a_model_unavailable_from_load_is_reported_once_it_loads() -> None:
    class UnavailableAtLoad(DependentModel):
        def load(self, config_path: Path | None) -> None:
            super().load(config_path)
            self.set_unavailable("worker_unavailable: no worker yet")

    _LOADED.clear()
    async with running_runtime(UnavailableAtLoad) as harness:
        await asyncio.sleep(0.05)
        health = await harness.client.get("/health")
        assert health.status_code == 503
        assert health.json()["detail"] == "worker_unavailable: no worker yet"
        assert (await harness.client.post("/start_session", json={})).status_code == 503


async def test_a_step_that_raises_model_unavailable_is_dropped_not_a_crash(
    harness: Harness,
) -> None:
    model = _model()
    model.gone.set()
    assert (await harness.client.post("/start_session", json={})).status_code == 200
    # No client connects, so give the loop a moment either way; the session
    # must still be running, with no crash and no step.
    await asyncio.sleep(0.2)
    assert harness.runner.health().status.value == "healthy"
    model.gone.clear()
    assert (await harness.client.post("/stop_session", json={})).status_code == 200
    await wait_for_state(harness.runner, "ready")
