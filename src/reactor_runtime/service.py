"""The service — the runtime's lifecycle root.

A tiny in-process supervision tree that owns the running of the process: the
sole arbiter of start, drain, and stop ordering, the one signal owner, and the
one readiness source. Components declare their startup dependencies; the service
starts them in dependency order, runs one blocking main loop until a shutdown is
requested, then drains and stops them in reverse so each component's intake
closes before the components it depends on wind down.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

from reactor_runtime.core import Health, ServiceComponent
from reactor_runtime.log import get_logger

logger = get_logger(__name__)


class Service:
    """The control block supervising the runtime's components.

    Components are hooked on with :meth:`add`; none manages its own place in the
    lifecycle. :meth:`run` starts them in dependency order, blocks on a single
    shutdown event, and on shutdown drains then stops them in reverse.
    """

    def __init__(self) -> None:
        """Start with no components and an unset shutdown signal."""
        self._components: dict[str, ServiceComponent] = {}
        self._shutdown = asyncio.Event()

    def add(self, component: ServiceComponent) -> None:
        """Register a component under its name.

        Args:
            component: The component to supervise.

        Raises:
            ValueError: If a component with the same name is already registered.
        """
        if component.name in self._components:
            raise ValueError(f"duplicate component name '{component.name}'")
        self._components[component.name] = component

    async def run(self) -> None:
        """Start every component, block until shutdown, then drain and stop.

        Components start in dependency order; the one signal handler and the one
        main loop live here. On shutdown — requested by a signal or
        :meth:`request_shutdown` — each started component is drained and then
        stopped in reverse order, so a component's intake closes before the
        components it depends on wind down. Cleanup runs for whatever started,
        even if a later start fails.
        """
        order = self._topological_order()
        started: list[ServiceComponent] = []
        try:
            for component in order:
                logger.info("starting component", component=component.name)
                await component.start()
                started.append(component)
            self._install_signal_handlers()
            logger.info("runtime started", components=[component.name for component in started])
            await self._shutdown.wait()
            logger.info("shutdown requested; draining")
        finally:
            for component in reversed(started):
                logger.info("draining component", component=component.name)
                await component.drain()
            for component in reversed(started):
                logger.info("stopping component", component=component.name)
                await component.stop()
            logger.info("runtime stopped")

    def request_shutdown(self) -> None:
        """Signal the main loop to begin draining and stopping."""
        self._shutdown.set()

    def health(self) -> Health:
        """Aggregate every component's health into one process readiness."""
        return Health.aggregate(component.health() for component in self._components.values())

    def _install_signal_handlers(self) -> None:
        """Route SIGTERM and SIGINT to a shutdown request, where supported."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_shutdown)

    def _topological_order(self) -> list[ServiceComponent]:
        """Order components so each starts after the components it depends on.

        Returns:
            The components in dependency order.

        Raises:
            ValueError: If a dependency names an unknown component, or the
                dependencies form a cycle.
        """
        ordered: list[ServiceComponent] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(name: str) -> None:
            if name in done:
                return
            if name in visiting:
                raise ValueError(f"dependency cycle through '{name}'")
            component = self._components.get(name)
            if component is None:
                raise ValueError(f"unknown component dependency '{name}'")
            visiting.add(name)
            for dependency in component.depends_on:
                visit(dependency)
            visiting.discard(name)
            done.add(name)
            ordered.append(component)

        for name in self._components:
            visit(name)
        return ordered
