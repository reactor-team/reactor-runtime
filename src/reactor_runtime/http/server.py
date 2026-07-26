"""The HTTP server — the runtime's single ASGI ingress.

Assembles the FastAPI application from the fixed route groups and one mount per
transport, then runs it under uvicorn. It is a :class:`ServiceComponent` that
comes up before the runner — so the surface is observable while the model loads
— and drains then stops before it, so intake closes before the model thread is
released. It never sees a ``Connection``, SDP, or ICE — those stay inside the
routers and acceptors it mounts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from reactor_runtime.core import Health, HealthStatus, RuntimeConfig, ServiceComponent
from reactor_runtime.http.routes import (
    EgressRoutes,
    RecordingRoutes,
    SessionRoutes,
    UploadRoutes,
)
from reactor_runtime.log import get_logger
from reactor_runtime.runner import Runner
from reactor_runtime.transport.router import TransportRouter

logger = get_logger(__name__)


def build_app(
    runner: Runner,
    transports: list[TransportRouter],
    process_health: Callable[[], Health],
) -> FastAPI:
    """Assemble the runtime's ASGI application from its route groups.

    The one place the HTTP surface is composed: the fixed route groups, the
    CORS policy, and one mount per transport. :class:`HttpServer` serves the
    result; the OpenAPI spec renderer reads the same assembly so the published
    contract is exactly the served surface.

    Args:
        runner: The runner the routes drive and the transports report into.
        transports: One router per connection type, each mounted onto the app.
        process_health: The health report ``/health`` answers with — the
            whole-process aggregate in the served assembly.

    Returns:
        The fully assembled FastAPI application.
    """
    app = FastAPI(title="reactor-runtime")
    # A standalone runtime is called directly by browser clients from their
    # own origin, so the session and signalling routes must answer the
    # cross-origin preflight. Auth rides the Authorization header, never a
    # cookie, so credentialed CORS is unnecessary and a wildcard origin is
    # safe; an operator fronting the runtime can tighten this upstream.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    SessionRoutes(runner).mount(app)
    EgressRoutes(runner, process_health).mount(app)
    UploadRoutes(runner).mount(app)
    RecordingRoutes(runner).mount(app)
    for transport in transports:
        transport.mount(app, runner)
    return app


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

    It holds the concrete :class:`~reactor_runtime.runner.Runner` deliberately:
    as the assembler of the route groups it hands them the full session-control
    and read surface they drive (start/stop session, descriptor, schema, events,
    health, recorder, uploads), which is broader than the neutral
    :class:`~reactor_runtime.transport.router.SessionControl` the transports it
    mounts hold.
    """

    name = "http"
    depends_on: tuple[str, ...] = ("runner",)

    def __init__(
        self,
        cfg: RuntimeConfig,
        runner: Runner,
        transports: list[TransportRouter],
        process_health: Callable[[], Health],
    ) -> None:
        """Assemble the app from the route groups and each transport's routes.

        Args:
            cfg: The configuration naming the address to bind.
            runner: The runner the routes drive and the transports report into.
            transports: One router per connection type, each mounted onto the app.
            process_health: The health report ``/health`` answers with,
                injected by the assembly so the endpoint speaks for the whole
                process rather than any one component.
        """
        self._cfg = cfg
        self._app = build_app(runner, transports, process_health)
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Bind the configured address and serve the app, returning once it accepts.

        The server runs in a background task, but start does not return until the
        socket is bound and accepting — so the component that starts after it (the
        runner, which loads the model) comes up with the HTTP surface already
        live, and a director subscribed to ``/events`` sees the model's
        initialization transitions as they happen rather than only via backlog.
        """
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
        await self._await_bound(self._server, self._serve_task)

    @staticmethod
    async def _await_bound(server: uvicorn.Server, serve_task: asyncio.Task[None]) -> None:
        """Wait until uvicorn has bound its socket, or surface a bind failure.

        Polls the server's own ``started`` flag; if the serve task finishes before
        the socket is up it failed to bind, so await it to re-raise the error
        rather than spin forever.
        """
        while not server.started:
            if serve_task.done():
                await serve_task
                return
            await asyncio.sleep(0.01)

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
        """Report healthy once the server has finished starting.

        The not-started report is unobservable on the wire — no request reaches
        ``/health`` before the socket is up — so it exists only for the process
        aggregate, where a server that should be serving but is not is broken.
        """
        if self._server is not None and self._server.started:
            return Health.healthy()
        return Health(HealthStatus.UNHEALTHY, "http server not started")
