import asyncio
import logging
import threading
from pathlib import Path

import pytest

from reactor_runtime import Output, ReactorModel, Video
from reactor_runtime.core import (
    Health,
    HealthStatus,
    RuntimeConfig,
    RuntimeState,
    SessionEvent,
    SessionState,
    TransitionEvent,
)
from reactor_runtime.http import HttpServer
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.runner.runner import Runner
from reactor_runtime.service import Service


class FakeComponent:
    """A service component that records the lifecycle calls it receives."""

    def __init__(
        self,
        name: str,
        depends_on: tuple[str, ...] = (),
        *,
        trace: list[str] | None = None,
        health: Health | None = None,
        fail_start: bool = False,
        fail_drain: bool = False,
        fail_stop: bool = False,
    ) -> None:
        self.name = name
        self.depends_on = depends_on
        self._trace = trace if trace is not None else []
        self._health = health if health is not None else Health.healthy()
        self._fail_start = fail_start
        self._fail_drain = fail_drain
        self._fail_stop = fail_stop

    async def start(self) -> None:
        if self._fail_start:
            self._trace.append(f"start-fail:{self.name}")
            raise RuntimeError(f"{self.name} failed to start")
        self._trace.append(f"start:{self.name}")

    async def drain(self) -> None:
        self._trace.append(f"drain:{self.name}")
        if self._fail_drain:
            raise RuntimeError(f"{self.name} failed to drain")

    async def stop(self) -> None:
        self._trace.append(f"stop:{self.name}")
        if self._fail_stop:
            raise RuntimeError(f"{self.name} failed to stop")

    def health(self) -> Health:
        return self._health


def _no_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Service, "_install_signal_handlers", lambda self: None)


async def test_starts_edge_first_and_winds_down_edge_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    trace: list[str] = []
    service = Service()
    # Added out of order to prove the edge-first ordering (the reverse of
    # dependency order) drives every phase: the HTTP edge that fronts the runner
    # comes up first and the runner — the core — is the last thing released.
    service.add(FakeComponent("http", ("runner",), trace=trace))
    service.add(FakeComponent("runner", (), trace=trace))

    task = asyncio.create_task(service.run())
    service.request_shutdown()
    await asyncio.wait_for(task, timeout=2.0)

    assert trace == [
        "start:http",
        "start:runner",
        "drain:http",
        "drain:runner",
        "stop:http",
        "stop:runner",
    ]


async def test_lifecycle_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _no_signals(monkeypatch)
    service = Service()
    service.add(FakeComponent("runner", ()))
    service.add(FakeComponent("http", ("runner",)))

    with caplog.at_level(logging.INFO, logger="reactor_runtime.service"):
        task = asyncio.create_task(service.run())
        service.request_shutdown()
        await asyncio.wait_for(task, timeout=2.0)

    logged = [
        (record.getMessage(), getattr(record, "reactor_fields", {}).get("component"))
        for record in caplog.records
    ]
    assert ("starting component", "runner") in logged
    assert ("stopping component", "runner") in logged
    assert ("runtime stopped", None) in logged


def test_duplicate_component_name_is_rejected() -> None:
    service = Service()
    service.add(FakeComponent("runner"))
    with pytest.raises(ValueError, match="duplicate"):
        service.add(FakeComponent("runner"))


async def test_unknown_dependency_is_rejected() -> None:
    service = Service()
    service.add(FakeComponent("http", ("runner",)))
    with pytest.raises(ValueError, match="unknown component dependency"):
        await service.run()


async def test_dependency_cycle_is_rejected() -> None:
    service = Service()
    service.add(FakeComponent("a", ("b",)))
    service.add(FakeComponent("b", ("a",)))
    with pytest.raises(ValueError, match="cycle"):
        await service.run()


async def test_failed_start_drains_and_stops_only_what_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    trace: list[str] = []
    service = Service()
    # http (the edge) starts first; the runner fails to start, so only http —
    # the one component that came up — is drained and stopped.
    service.add(FakeComponent("runner", (), trace=trace, fail_start=True))
    service.add(FakeComponent("http", ("runner",), trace=trace))

    with pytest.raises(RuntimeError):
        await service.run()

    assert trace == ["start:http", "start-fail:runner", "drain:http", "stop:http"]


async def test_shutdown_winds_down_the_rest_when_a_component_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    trace: list[str] = []
    service = Service()
    service.add(FakeComponent("runner", (), trace=trace))
    service.add(FakeComponent("http", ("runner",), trace=trace, fail_drain=True, fail_stop=True))

    task = asyncio.create_task(service.run())
    service.request_shutdown()
    await asyncio.wait_for(task, timeout=2.0)

    # `http` (the edge) drains and stops first and raises on both, but the
    # failure is isolated: `runner` still drains and stops all the way down.
    assert trace == [
        "start:http",
        "start:runner",
        "drain:http",
        "drain:runner",
        "stop:http",
        "stop:runner",
    ]


def test_health_aggregates_to_the_worst_status() -> None:
    service = Service()
    service.add(FakeComponent("runner", health=Health.healthy()))
    service.add(FakeComponent("http", health=Health(HealthStatus.UNHEALTHY, "not started")))

    rolled = service.health()
    assert rolled.status is HealthStatus.UNHEALTHY
    assert rolled.detail == "not started"


# --- the HTTP edge is up before the model finishes loading (REA-3604) ---------

_LOAD_GATE = threading.Event()


class _GatedOut(Output):
    main: Video


class _GatedModel(ReactorModel):
    """A model whose load blocks off the event loop until a test releases it."""

    output: _GatedOut

    def load(self, config_path: Path | None) -> None:
        _LOAD_GATE.wait(timeout=5.0)

    async def run(self) -> None:
        await asyncio.sleep(60)


def _state(runner: Runner) -> SessionState:
    # A call boundary so the type checker does not narrow the session state
    # across the awaits that change it.
    return runner._sm.current_state


async def test_http_surface_is_up_before_the_model_finishes_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    _LOAD_GATE.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: _GatedModel)
    cfg = RuntimeConfig(model_ref="x:_GatedModel", host="127.0.0.1", port=0)
    service = Service()
    runner = Runner(cfg)
    metrics = RuntimeMetrics(version="0.0.0", model=cfg.model_ref)
    http = HttpServer(cfg, runner, [], process_health=service.health, metrics=metrics)
    service.add(runner)
    service.add(http)

    task = asyncio.create_task(service.run())
    try:
        # The edge binds first: HTTP is accepting while the model is still loading.
        for _ in range(500):
            if http._server is not None and http._server.started:
                break
            await asyncio.sleep(0.01)
        assert http._server is not None
        assert http._server.started
        assert _state(runner) is SessionState.CREATED
        # A loading model is healthy — the lifecycle word, not the verdict,
        # says it cannot serve yet — so the process aggregate is healthy too.
        assert runner.health().status is HealthStatus.HEALTHY
        assert runner.state() is RuntimeState.LOADING
        assert service.health().status is HealthStatus.HEALTHY

        # Releasing the load lets the runner reach READY and journal the init fact,
        # which is emitted while the HTTP surface is already live.
        _LOAD_GATE.set()
        for _ in range(500):
            if _state(runner) is SessionState.READY:
                break
            await asyncio.sleep(0.01)
        assert _state(runner) is SessionState.READY
        assert runner.state() is RuntimeState.AVAILABLE
        journalled = [event for _seq, event in runner._events._history]
        assert any(
            isinstance(event, TransitionEvent)
            and event.transition.event is SessionEvent.INITIALIZATION_SUCCESS
            for event in journalled
        )
    finally:
        _LOAD_GATE.set()
        service.request_shutdown()
        await asyncio.wait_for(task, timeout=5.0)
