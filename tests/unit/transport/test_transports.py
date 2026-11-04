import asyncio
import logging

import pytest
from fastapi import FastAPI

from reactor_runtime.core import Health, HealthStatus
from reactor_runtime.transport.router import SessionControl, TransportRouter, Transports


class _Router(TransportRouter):
    def __init__(self, *, settles_after: float = 0.0, health: Health | None = None) -> None:
        self._settles_after = settles_after
        self._health = health or Health.healthy()
        self.settled = False
        self.cancelled = False

    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        pass

    async def settle(self) -> None:
        try:
            await asyncio.sleep(self._settles_after)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.settled = True

    def health(self) -> Health:
        return self._health


class _Plain(TransportRouter):
    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        pass


async def test_a_router_settles_at_once_and_is_healthy_unless_it_says_otherwise() -> None:
    router = _Plain()

    await asyncio.wait_for(router.settle(), timeout=0.1)

    assert router.health().status is HealthStatus.HEALTHY


async def test_stop_waits_for_every_router_to_settle() -> None:
    routers = [_Router(settles_after=0.05), _Router()]

    await Transports(routers, settle_timeout=1.0).stop()

    assert all(router.settled for router in routers)


async def test_stop_gives_up_at_the_bound_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    slow = _Router(settles_after=10.0)

    with caplog.at_level(logging.WARNING, logger="reactor_runtime.transport.router"):
        await asyncio.wait_for(Transports([slow], settle_timeout=0.05).stop(), timeout=1.0)

    assert slow.cancelled
    assert not slow.settled
    assert any(
        r.getMessage().startswith("transport work still running at shutdown")
        for r in caplog.records
    )


def test_health_aggregates_the_routers() -> None:
    sick = Health(HealthStatus.UNHEALTHY, "a room leave is stuck")
    transports = Transports([_Router(), _Router(health=sick)], settle_timeout=1.0)

    health = transports.health()

    assert health.status is HealthStatus.UNHEALTHY
    assert "a room leave is stuck" in str(health.detail)
