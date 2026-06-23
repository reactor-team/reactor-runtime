"""The fixed HTTP route groups over the runner.

Thin endpoint groups that translate HTTP requests into calls on the runner and
read its egress journal. They hold the runner only as a caller of its session-
control surface and a reader of its read-only properties; the model, signalling,
and connections never surface here.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import Body, FastAPI, Header, HTTPException
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
            body = params or {}
            runner.start_session(body)
            return _descriptor(runner, _sdk_version_from_body(body))

        @app.get("/session")
        async def get_session(
            sdk_version: Annotated[str | None, Header(alias="reactor-sdk-version")] = None,
        ) -> dict[str, Any]:
            return _descriptor(runner, sdk_version)

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


def _descriptor(runner: Runner, sdk_version: str | None) -> dict[str, Any]:
    """Pick the session descriptor shape for the requesting client.

    The body is version-gated, never the flow: an SDK version of 2 or below, or
    none at all, gets the legacy shape; a newer one gets the runtime-native
    shape.
    """
    if _is_legacy_client(sdk_version):
        return runner.legacy_descriptor()
    return runner.descriptor()


def _is_legacy_client(sdk_version: str | None) -> bool:
    """Whether a client's SDK version asks for the legacy descriptor shape.

    Absent or major version 2 or below is legacy — so the shipped v0 client,
    which sends no version on session calls, gets the legacy shape untouched. A
    major version of 3 or more (the v1-protocol client) gets the new shape.
    """
    if sdk_version is None:
        return True
    try:
        return int(sdk_version.split(".", 1)[0]) <= 2
    except ValueError:
        return True


def _sdk_version_from_body(body: dict[str, Any]) -> str | None:
    """Read ``client_info.sdk_version`` from a session-call body, if present."""
    client_info = body.get("client_info")
    if isinstance(client_info, dict):
        version = client_info.get("sdk_version")
        if isinstance(version, str):
            return version
    return None


async def _stream_events(runner: Runner, since: int | None) -> AsyncGenerator[str, None]:
    """Yield the runner's events as SSE messages, resuming after *since*.

    Sequence numbers are gap-free, so the running count started from *since* (or
    the journal's current end when *since* is absent) labels each message.
    """
    seq = since if since is not None else runner.events.snapshot().last_seq
    async for event in runner.events.subscribe(since):
        seq += 1
        yield format_sse(seq, event)
