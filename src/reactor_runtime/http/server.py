"""The HTTP server — the runtime's single ASGI ingress.

Assembles the FastAPI application from the fixed route groups and one mount per
transport, then runs it under uvicorn. It is a :class:`ServiceComponent` that
starts after the runner and drains before it: draining stops accepting new
requests, stop awaits the server's own shutdown. It never sees a ``Connection``,
SDP, or ICE — those stay inside the routers and acceptors it mounts.
"""

from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from reactor_runtime.core import Health, HealthStatus, RuntimeConfig, ServiceComponent
from reactor_runtime.http.routes import EgressRoutes, SessionRoutes, UploadRoutes
from reactor_runtime.log import get_logger
from reactor_runtime.runner import Runner
from reactor_runtime.transport.router import TransportRouter

logger = get_logger(__name__)


class _ServerWithoutSignals(uvicorn.Server):
    """A uvicorn server that leaves signal handling to the service.

    The service owns the one signal handler for the process, so the server must
    not install its own.
    """

    def install_signal_handlers(self) -> None:
        """Do nothing — the service owns process signals."""


class HttpServer(ServiceComponent):
    """The single HTTP ingress, mounting the route groups and transport routers.

    Built with the runner it exposes and the transport routers that grow the
    surface per connection type. The FastAPI app is assembled at construction;
    the uvicorn server is created and run on :meth:`start`.
    """

    name = "http"
    depends_on: tuple[str, ...] = ("runner",)

    def __init__(
        self, cfg: RuntimeConfig, runner: Runner, transports: list[TransportRouter]
    ) -> None:
        """Assemble the app from the route groups and each transport's routes.

        Args:
            cfg: The configuration naming the address to bind.
            runner: The runner the routes drive and the transports report into.
            transports: One router per connection type, each mounted onto the app.
        """
        self._cfg = cfg
        self._app = FastAPI(title="reactor-runtime")
        # A standalone runtime is called directly by browser clients from their
        # own origin, so the session and signalling routes must answer the
        # cross-origin preflight. Auth rides the Authorization header, never a
        # cookie, so credentialed CORS is unnecessary and a wildcard origin is
        # safe; an operator fronting the runtime can tighten this upstream.
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        SessionRoutes(runner).mount(self._app)
        EgressRoutes(runner).mount(self._app)
        UploadRoutes(runner).mount(self._app)
        for transport in transports:
            transport.mount(self._app, runner)
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Bind the configured address and serve the app in the background."""
        # `/events` is an unbounded stream, so a client still subscribed when the
        # server drains never lets uvicorn's graceful shutdown complete on its
        # own. Bound the wait with the same grace period a draining session gets,
        # so `stop()` cannot hang on a long-lived connection.
        config = uvicorn.Config(
            self._app,
            host=self._cfg.host,
            port=self._cfg.port,
            log_level="warning",
            timeout_graceful_shutdown=int(self._cfg.grace_period),
        )
        self._server = _ServerWithoutSignals(config)
        self._serve_task = asyncio.create_task(self._server.serve())

    async def drain(self) -> None:
        """Stop accepting new requests, letting in-flight ones finish."""
        if self._server is not None:
            self._server.should_exit = True

    async def stop(self) -> None:
        """Await the server's own shutdown, bounded by the graceful-shutdown timeout.

        The wait is already bounded — ``timeout_graceful_shutdown`` caps how long
        uvicorn lingers on a still-open connection — so this only awaits the serve
        task to settle. A failure surfacing from that task during shutdown is
        logged rather than raised, so a late serve error cannot abort the rest of
        the ordered teardown.
        """
        if self._serve_task is not None:
            try:
                await self._serve_task
            except Exception:
                logger.exception("http serve task failed during shutdown")

    def health(self) -> Health:
        """Report ready once the server has finished starting."""
        if self._server is not None and self._server.started:
            return Health.healthy()
        return Health(HealthStatus.DEGRADED, "http server not started")
