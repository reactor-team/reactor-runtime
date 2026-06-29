"""The fixed HTTP route groups over the runner.

Thin endpoint groups that translate HTTP requests into calls on the runner and
read its egress journal. They hold the runner only as a caller of its session-
control surface and a reader of its read-only properties; the model, signalling,
and connections never surface here.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from reactor_runtime.core import HealthStatus
from reactor_runtime.http.events import format_sse
from reactor_runtime.runner import Runner
from reactor_runtime.transport.router import SessionNotRunningError, UnknownSessionError
from reactor_runtime.upload_store import (
    UnknownUploadError,
    UploadAlreadyCompleteError,
    UploadIdTakenError,
    UploadSizeMismatchError,
)


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

        @app.get("/schema")
        async def get_schema() -> dict[str, Any]:
            return runner.schema()

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


class CreateUploadRequest(BaseModel):
    """Metadata a client announces to reserve an upload slot.

    ``upload_id`` is optional: when omitted the store mints one, and when present
    the slot is reserved under that exact id so a caller can seed bytes a later
    reference already names.
    """

    name: str
    size: int
    mime_type: str
    upload_id: str | None = None


class UploadRoutes:
    """File-upload ingress: slot allocation and the byte write.

    Two endpoints reproduce the client's two-step upload against the local store:
    a ``POST`` reserves a slot and hands back the URL to write to, and a ``PUT``
    delivers the bytes. The model reads the result later, when a command or a
    file-uploaded notification references the slot.
    """

    def __init__(self, runner: Runner) -> None:
        """Bind the route group to the runner whose upload store it drives."""
        self._runner = runner

    def mount(self, app: FastAPI) -> None:
        """Register the upload routes against *app*."""
        runner = self._runner

        @app.post("/sessions/{sid}/uploads")
        async def create_upload(
            sid: str, req: CreateUploadRequest, request: Request
        ) -> JSONResponse:
            try:
                runner.require_session_running(sid)
            except SessionNotRunningError:
                raise HTTPException(status_code=400, detail="No session running") from None
            except UnknownSessionError:
                raise HTTPException(status_code=404, detail="Unknown session") from None
            if not req.name:
                raise HTTPException(status_code=400, detail="name is required")
            if not req.mime_type:
                raise HTTPException(status_code=400, detail="mime_type is required")
            if req.size <= 0:
                raise HTTPException(status_code=400, detail="size must be > 0")
            try:
                upload_id = runner.uploads.create_slot(
                    req.name, req.mime_type, req.size, req.upload_id
                )
            except UploadIdTakenError:
                raise HTTPException(status_code=409, detail="Upload id already reserved") from None
            base = str(request.base_url).rstrip("/")
            return JSONResponse(
                status_code=201,
                content={
                    "presigned_id": upload_id,
                    "presigned_url": f"{base}/uploads/{upload_id}",
                    "path": f"sessions/{sid}/uploads/{upload_id}/{req.name}",
                },
            )

        @app.put("/uploads/{upload_id}")
        async def put_upload(upload_id: str, request: Request) -> Response:
            # The slot knows the exact byte count it expects, so a write whose
            # declared length is wrong is rejected before the body is read —
            # otherwise an oversized payload is buffered whole only to fail.
            try:
                expected = runner.uploads.expected_size(upload_id)
            except UnknownUploadError:
                raise HTTPException(status_code=404, detail="Upload not found") from None
            except UploadAlreadyCompleteError:
                raise HTTPException(status_code=409, detail="Upload already completed") from None
            declared = request.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) != expected:
                raise HTTPException(
                    status_code=400, detail=f"expected {expected} bytes, got {declared}"
                )
            body = await _read_capped(request, expected)
            try:
                runner.uploads.put(upload_id, body)
            except UnknownUploadError:
                raise HTTPException(status_code=404, detail="Upload not found") from None
            except UploadAlreadyCompleteError:
                raise HTTPException(status_code=409, detail="Upload already completed") from None
            except UploadSizeMismatchError as error:
                raise HTTPException(status_code=400, detail=str(error)) from None
            return Response(status_code=200)


async def _read_capped(request: Request, limit: int) -> bytes:
    """Read the request body, aborting once it runs past *limit* bytes.

    Streaming the body and stopping at the slot's expected size bounds the memory
    a single upload can take: a chunked ``PUT`` that declares no length (or lies
    about it) cannot buffer an unbounded payload before the store rejects it. The
    store stays the authoritative validator of the exact byte count; this only
    caps what is read off the wire.

    Raises:
        HTTPException: 400 if the body exceeds *limit* bytes.
    """
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(status_code=400, detail=f"expected {limit} bytes, got more")
    return bytes(body)


class EgressRoutes:
    """The egress journal and liveness over HTTP."""

    def __init__(self, runner: Runner) -> None:
        """Bind the route group to the runner it reads."""
        self._runner = runner

    def mount(self, app: FastAPI) -> None:
        """Register the egress and health routes against *app*."""
        runner = self._runner

        @app.get("/events")
        async def events(
            since: int | None = None,
            last_event_id: Annotated[str | None, Header()] = None,
        ) -> StreamingResponse:
            resume = since if since is not None else _resume_from(last_event_id)
            return StreamingResponse(_stream_events(runner, resume), media_type="text/event-stream")

        @app.get("/health")
        async def health() -> JSONResponse:
            report = runner.health()
            code = 503 if report.status is HealthStatus.UNHEALTHY else 200
            return JSONResponse(
                status_code=code,
                content={"status": report.status.value, "detail": report.detail},
            )


def _resume_from(last_event_id: str | None) -> int | None:
    """Read a ``Last-Event-ID`` header as a resumption point, when it is one.

    A standard ``EventSource`` replays the last ``id:`` it received on
    auto-reconnect, so an integer value resumes the stream after that sequence;
    a missing or non-numeric header resumes from live. An explicit ``?since=``
    always takes precedence over the header.

    Args:
        last_event_id: The ``Last-Event-ID`` request header, if the client sent
            one.

    Returns:
        The sequence number to resume after, or ``None`` to resume from live.
    """
    if last_event_id is None or not last_event_id.isdigit():
        return None
    return int(last_event_id)


async def _stream_events(runner: Runner, since: int | None) -> AsyncGenerator[str, None]:
    """Yield the runner's events as SSE messages, resuming after *since*.

    Sequence numbers are gap-free, so the running count started from *since* (or
    the journal's current end when *since* is absent) labels each message.
    """
    seq = since if since is not None else runner.events.snapshot().last_seq
    async for event in runner.events.subscribe(since):
        seq += 1
        yield format_sse(seq, event)
