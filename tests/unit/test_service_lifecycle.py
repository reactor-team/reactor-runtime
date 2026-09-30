import asyncio
import logging
import threading
from pathlib import Path

import httpx
import pytest

from reactor_runtime import Output, ReactorApp, Video
from reactor_runtime import service as service_module
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


class _GatedModel(ReactorApp):
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


# --- a requested shutdown ends the process inside a deadline (REA-6768) -------


class _StuckComponent(FakeComponent):
    """A component whose stop never returns — a thread the runtime does not own."""

    async def stop(self) -> None:
        self._trace.append(f"stop:{self.name}")
        await asyncio.Event().wait()


def _intercept_exit(monkeypatch: pytest.MonkeyPatch) -> tuple[list[int], threading.Event]:
    """Replace the forced exit with a recorder, so the test process survives it."""
    exits: list[int] = []
    fired = threading.Event()

    def record(code: int) -> None:
        exits.append(code)
        fired.set()

    monkeypatch.setattr(service_module, "_exit", record)
    return exits, fired


async def test_a_stuck_wind_down_is_forced_out_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _no_signals(monkeypatch)
    exits, fired = _intercept_exit(monkeypatch)
    trace: list[str] = []
    service = Service(exit_timeout=0.2, grace_period=60.0)
    service.add(_StuckComponent("runner", (), trace=trace))

    with caplog.at_level(logging.ERROR, logger="reactor_runtime.service"):
        task = asyncio.create_task(service.run())
        for _ in range(100):
            if "start:runner" in trace:
                break
            await asyncio.sleep(0.01)
        service.request_shutdown(failure=True)

        # The deadline fires off the event loop, so a wind-down stuck on the
        # loop cannot hold it up; it carries the failure status.
        assert await asyncio.to_thread(fired.wait, 2.0)
    assert exits == [1]
    assert service.exit_code == 1
    assert "stop:runner" in trace
    assert not task.done()
    # Why the process left is on record before it does.
    assert any("shutdown deadline exceeded" in r.getMessage() for r in caplog.records)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_failure_deadline_skips_the_session_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    exits, fired = _intercept_exit(monkeypatch)
    # A crash has no session left to drain, so its budget is exit_timeout
    # alone; the grace period would otherwise hold the restart up for nothing.
    service = Service(exit_timeout=0.2, grace_period=60.0)
    service.add(_StuckComponent("runner", ()))

    task = asyncio.create_task(service.run())
    await asyncio.sleep(0.01)
    service.request_shutdown(failure=True)

    assert await asyncio.to_thread(fired.wait, 2.0)
    assert exits == [1]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_requested_stop_keeps_the_session_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    exits, fired = _intercept_exit(monkeypatch)
    service = Service(exit_timeout=0.1, grace_period=0.4)
    service.add(_StuckComponent("runner", ()))

    task = asyncio.create_task(service.run())
    await asyncio.sleep(0.01)
    service.request_shutdown()

    # Inside the grace period the deadline is still pending.
    assert not await asyncio.to_thread(fired.wait, 0.2)
    assert await asyncio.to_thread(fired.wait, 2.0)
    assert exits == [0]
    assert service.exit_code == 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_crash_during_a_requested_stop_is_still_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    _intercept_exit(monkeypatch)
    service = Service(exit_timeout=60.0, grace_period=60.0)
    service.add(FakeComponent("runner", ()))

    task = asyncio.create_task(service.run())
    service.request_shutdown()
    service.request_shutdown(failure=True)
    await asyncio.wait_for(task, timeout=2.0)

    assert service.exit_code == 1


async def test_a_wind_down_that_completes_disarms_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    exits, fired = _intercept_exit(monkeypatch)
    service = Service(exit_timeout=0.1, grace_period=0.0)
    service.add(FakeComponent("runner", ()))

    task = asyncio.create_task(service.run())
    service.request_shutdown(failure=True)
    await asyncio.wait_for(task, timeout=2.0)

    assert service.exit_code == 1
    assert not await asyncio.to_thread(fired.wait, 0.3)
    assert exits == []


async def test_bounding_the_process_exit_arms_a_fresh_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    exits, fired = _intercept_exit(monkeypatch)
    service = Service(exit_timeout=0.1, grace_period=0.0)
    service.add(FakeComponent("runner", ()))

    task = asyncio.create_task(service.run())
    service.request_shutdown(failure=True)
    await asyncio.wait_for(task, timeout=2.0)
    # The interpreter's own teardown — thread joins the runtime does not
    # control — is what this second deadline bounds.
    service.bound_process_exit()

    assert await asyncio.to_thread(fired.wait, 2.0)
    assert exits == [1]


class _CrashingModel(ReactorApp):
    """A model whose run loop dies as soon as it starts."""

    output: _GatedOut

    def load(self, config_path: Path | None) -> None: ...

    async def run(self) -> None:
        raise RuntimeError("gpu fell off")


def _bound_port(http: HttpServer) -> int:
    assert http._server is not None
    return http._server.servers[0].sockets[0].getsockname()[1]


async def test_a_crashed_model_ends_the_service_in_seconds_with_a_subscriber_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    _intercept_exit(monkeypatch)
    monkeypatch.setattr(
        "reactor_runtime.runner.runner.import_model_class", lambda ref: _CrashingModel
    )
    # The grace period is what an open `/events` subscription used to cost the
    # exit: uvicorn waited it out before cutting the stream. Left long here to
    # prove the wind-down does not depend on it any more.
    cfg = RuntimeConfig(model_ref="x:_CrashingModel", host="127.0.0.1", port=0, grace_period=60.0)
    service = Service(exit_timeout=30.0, grace_period=cfg.grace_period)
    runner = Runner(cfg)
    runner.request_shutdown = service.request_shutdown
    metrics = RuntimeMetrics(version="0.0.0", model=cfg.model_ref)
    http = HttpServer(cfg, runner, [], process_health=service.health, metrics=metrics)
    service.add(runner)
    service.add(http)

    task = asyncio.create_task(service.run())
    try:
        for _ in range(500):
            if http._server is not None and http._server.started:
                break
            await asyncio.sleep(0.01)
        assert http._server is not None
        assert http._server.started
        # A subscriber holds the stream open across the crash and the exit, as
        # the platform's own event consumer does.
        url = f"http://127.0.0.1:{_bound_port(http)}/events"
        async with httpx.AsyncClient() as client, client.stream("GET", url) as stream:
            assert stream.status_code == 200
            started_at = asyncio.get_running_loop().time()
            await asyncio.wait_for(task, timeout=10.0)
            elapsed = asyncio.get_running_loop().time() - started_at
        assert elapsed < 5.0
        assert _state(runner) is SessionState.TERMINATED
        assert service.exit_code == 1
    finally:
        if not task.done():
            service.request_shutdown()
            await asyncio.wait_for(task, timeout=10.0)
