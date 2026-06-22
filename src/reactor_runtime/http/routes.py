"""The fixed HTTP route groups over the runner.

Thin endpoint groups that translate HTTP requests into calls on the runner and
read its egress journal. They hold the runner only as a caller of its session-
control surface and a reader of its read-only properties; the model, signalling,
and connections never surface here.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from reactor_runtime.core import HealthStatus
from reactor_runtime.http.events import format_sse
from reactor_runtime.runner import Runner
from reactor_runtime.transport.router import SessionNotRunningError, UnknownSessionError


class EnforceRequest(BaseModel):
    """A moderation verdict posted against the active session."""

    block: bool = True


class SessionRoutes:
    """Session lifecycle control, driving the runner's session-control face."""

    def __init__(self, runner: Runner) -> None:
        """Bind the route group to the runner it drives."""
        self._runner = runner

    def mount(self, app: FastAPI) -> None:
        """Register the session-control routes against *app*."""
        runner = self._runner

        @app.post("/start_session")
        async def start_session(
            params: Annotated[dict[str, Any] | None, Body()] = None,
        ) -> dict[str, Any]:
            runner.start_session(params or {})
            return runner.descriptor()

        @app.get("/session")
        async def get_session() -> dict[str, Any]:
            return runner.descriptor()

        @app.post("/stop_session")
        async def stop_session() -> Response:
            runner.stop_session()
            return Response(status_code=200)

        @app.post("/sessions/{sid}/enforce")
        async def enforce(sid: str, req: EnforceRequest) -> Response:
            try:
                runner.require_session_running(sid)
            except SessionNotRunningError:
                raise HTTPException(status_code=400, detail="No session running") from None
            except UnknownSessionError:
                raise HTTPException(status_code=404, detail="Unknown session") from None
            runner.enforce(req.block)
            return Response(status_code=200)


class EgressRoutes:
    """The egress journal and liveness over HTTP."""

    def __init__(self, runner: Runner) -> None:
        """Bind the route group to the runner it reads."""
        self._runner = runner

    def mount(self, app: FastAPI) -> None:
        """Register the egress and health routes against *app*."""
        runner = self._runner

        @app.get("/events")
        async def events(since: int | None = None) -> StreamingResponse:
            return StreamingResponse(_stream_events(runner, since), media_type="text/event-stream")

        @app.get("/health")
        async def health() -> JSONResponse:
            report = runner.health()
            code = 503 if report.status is HealthStatus.UNHEALTHY else 200
            return JSONResponse(
                status_code=code,
                content={"status": report.status.value, "detail": report.detail},
            )


async def _stream_events(runner: Runner, since: int | None) -> AsyncGenerator[str, None]:
    """Yield the runner's events as SSE messages, resuming after *since*.

    Sequence numbers are gap-free, so the running count started from *since* (or
    the journal's current end when *since* is absent) labels each message.
    """
    seq = since if since is not None else runner.events.snapshot().last_seq
    async for event in runner.events.subscribe(since):
        seq += 1
        yield format_sse(seq, event)
