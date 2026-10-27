"""Lock what an application reporting its model unavailable does to the session surface.

An application whose model runs somewhere it does not own (a model worker in
another process) decides what a failure of it means in ``process_output()``.
Re-raising ends the session and the process (option 1). Calling
``report_unavailable(reason)`` (option 2) stops the live session with
``error`` and the reason, makes ``/health`` and ``/start_session`` answer 503,
and keeps the process up; once the application calls ``report_available()``
everything serves as before. Nothing in the runtime calls either on its own.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np
import pytest
from contract_helpers import Harness, JournalReader, SteppingOutput, running_runtime, wait_for_state

from reactor_runtime import ReactorApp, StepOutcome

_LOADED: list[ReactorApp] = []


class DependentModel(ReactorApp):
    """A model on the default step loop whose steps fail while its dependency is gone.

    Its ``process_output()`` takes option 2: it reports the model unavailable
    and emits nothing.
    """

    def load(self, config_path: Path | None) -> None:
        self.gone = threading.Event()
        self.steps = 0
        self.outcomes: list[StepOutcome] = []
        _LOADED.append(self)

    def generate(self, input: object) -> SteppingOutput:
        if self.gone.is_set():
            raise RuntimeError("worker_died: gone")
        self.steps += 1
        return SteppingOutput(frames=np.zeros((2, 2, 3), dtype=np.uint8))

    async def process_output(self, outcome: StepOutcome) -> SteppingOutput | None:
        self.outcomes.append(outcome)
        if outcome.error is not None:
            self.report_unavailable(str(outcome.error))
            return None
        return outcome.result


class ReraisingModel(DependentModel):
    """The same model taking option 1: a failed step is re-raised."""

    async def process_output(self, outcome: StepOutcome) -> SteppingOutput | None:
        if outcome.error is not None:
            raise outcome.error
        return outcome.result


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
        await _from_thread(lambda: model.report_unavailable("worker_died: socket closed"))
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
    await _from_thread(lambda: model.report_unavailable("worker_died: socket closed"))
    await asyncio.sleep(0.05)
    health = await harness.client.get("/health")
    assert health.status_code == 503
    assert health.json()["detail"] == "worker_died: socket closed"
    assert health.json()["state"] == "available"
    started = await harness.client.post("/start_session", json={})
    assert started.status_code == 503
    assert started.json()["detail"] == "worker_died: socket closed"
    assert started.headers["retry-after"] == "1"

    await _from_thread(model.report_available)
    await asyncio.sleep(0.05)
    assert (await harness.client.get("/health")).status_code == 200
    assert (await harness.client.post("/start_session", json={})).status_code == 200


async def test_a_repeat_report_updates_the_reason_and_stops_nothing_more(
    harness: Harness,
) -> None:
    model = _model()
    journal = JournalReader(harness.runner)
    try:
        await _from_thread(lambda: model.report_unavailable("worker_unavailable: connecting"))
        await journal.expect("error")
        await _from_thread(lambda: model.report_unavailable("worker_died: refused"))
        await asyncio.sleep(0.05)
        assert (await harness.client.get("/health")).json()["detail"] == "worker_died: refused"
    finally:
        await journal.aclose()


async def test_a_model_unavailable_from_load_is_reported_once_it_loads() -> None:
    class UnavailableAtLoad(DependentModel):
        def load(self, config_path: Path | None) -> None:
            super().load(config_path)
            self.report_unavailable("worker_unavailable: no worker yet")

    _LOADED.clear()
    async with running_runtime(UnavailableAtLoad) as harness:
        await asyncio.sleep(0.05)
        health = await harness.client.get("/health")
        assert health.status_code == 503
        assert health.json()["detail"] == "worker_unavailable: no worker yet"
        assert (await harness.client.post("/start_session", json={})).status_code == 503


async def test_option_2_a_failed_step_reaches_process_output_and_ends_the_session(
    harness: Harness,
) -> None:
    model = _model()
    model.gone.set()
    journal = JournalReader(harness.runner)
    try:
        started = await harness.client.post("/start_session", json={"steps": 50})
        assert started.status_code == 200
        error = await journal.expect("error")
        assert error["detail"] == {"message": "worker_died: gone"}
        stopped = await journal.expect("stop_session")
        assert stopped["detail"] == {"reason": "error", "close_reason": "worker_died: gone"}
        await wait_for_state(harness.runner, "ready")
    finally:
        await journal.aclose()
    assert model.outcomes
    assert isinstance(model.outcomes[0].error, RuntimeError)
    health = await harness.client.get("/health")
    assert (health.status_code, health.json()["detail"]) == (503, "worker_died: gone")
    refused = await harness.client.post("/start_session", json={})
    assert refused.status_code == 503
    assert refused.headers["retry-after"] == "1"
    assert model.unavailable == "worker_died: gone"

    model.gone.clear()
    await _from_thread(model.report_available)
    await asyncio.sleep(0.05)
    assert model.unavailable is None
    assert (await harness.client.get("/health")).status_code == 200


async def test_option_1_a_reraised_step_error_evicts_the_session() -> None:
    _LOADED.clear()
    async with running_runtime(ReraisingModel) as harness:
        model = _model()
        model.gone.set()
        journal = JournalReader(harness.runner)
        try:
            started = await harness.client.post("/start_session", json={"steps": 50})
            assert started.status_code == 200
            evicted = await journal.expect("eviction")
            assert evicted["to"] == "terminated"
            assert evicted["detail"]["reason"] == "error"
            assert "worker_died: gone" in evicted["detail"]["error"]
        finally:
            await journal.aclose()
        assert model.unavailable is None


async def test_the_runtime_never_reports_unavailable_on_its_own(harness: Harness) -> None:
    model = _model()
    assert (await harness.client.post("/start_session", json={"steps": 3})).status_code == 200
    await wait_for_state(harness.runner, "ready")
    assert model.steps >= 1
    assert model.unavailable is None
    assert (await harness.client.get("/health")).status_code == 200
