import asyncio
import logging

import pytest

from reactor_runtime.core import Health, HealthStatus
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


async def test_starts_in_dependency_order_then_reverses_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_signals(monkeypatch)
    trace: list[str] = []
    service = Service()
    # Added out of dependency order to prove the topological sort drives ordering.
    service.add(FakeComponent("http", ("runner",), trace=trace))
    service.add(FakeComponent("runner", (), trace=trace))

    task = asyncio.create_task(service.run())
    service.request_shutdown()
    await asyncio.wait_for(task, timeout=2.0)

    assert trace == [
        "start:runner",
        "start:http",
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
    service.add(FakeComponent("runner", (), trace=trace))
    service.add(FakeComponent("http", ("runner",), trace=trace, fail_start=True))

    with pytest.raises(RuntimeError):
        await service.run()

    assert trace == ["start:runner", "start-fail:http", "drain:runner", "stop:runner"]


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

    # `http` drains and stops first (reverse order) and raises on both, but the
    # failure is isolated: `runner` still drains and stops all the way down.
    assert trace == [
        "start:runner",
        "start:http",
        "drain:http",
        "drain:runner",
        "stop:http",
        "stop:runner",
    ]


def test_health_aggregates_to_the_worst_status() -> None:
    service = Service()
    service.add(FakeComponent("runner", health=Health.healthy()))
    service.add(FakeComponent("http", health=Health(HealthStatus.DEGRADED, "warming up")))

    assert service.health().status is HealthStatus.DEGRADED
