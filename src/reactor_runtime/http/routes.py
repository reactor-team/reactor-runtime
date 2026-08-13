"""The fixed HTTP route groups over the runner.

Thin endpoint groups that translate HTTP requests into calls on the runner and
read its egress journal. They hold the runner only as a caller of its session-
control surface and a reader of its read-only properties; the model, signalling,
and connections never surface here.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Annotated, Any

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from reactor_runtime.core import Health, HealthStatus, RuntimeState, SessionState
from reactor_runtime.http.events import format_sse
from reactor_runtime.metrics import CONTENT_TYPE as METRICS_CONTENT_TYPE
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.recording import ClipManifest, ClipSessionGoneError, Pending
from reactor_runtime.runner import Runner
from reactor_runtime.transport.router import (
    ErrorDetail,
    SessionNotRunningError,
    SessionTransitionError,
    UnknownSessionError,
)
from reactor_runtime.upload_store import (
    UnknownUploadError,
    UploadAlreadyCompleteError,
    UploadIdTakenError,
    UploadSizeMismatchError,
)


class StopSessionRequest(BaseModel):
    """Optional qualifiers on a session stop.

    ``moderate`` marks the stop as a content-moderation verdict: the session
    ends as moderated and clients are notified before their connections close.
    ``reason`` is the platform's close-reason token (e.g. ``"deployment"``):
    when set, clients receive a session-ended notice naming the reason before
    their connections close. A moderated stop outranks the token — clients see
    only the moderation notice. The body itself is optional — a bare
    ``POST /stop_session`` is a plain stop.
    """

    moderate: bool = False
    reason: str = Field(default="", max_length=64)


# The non-2xx statuses a route can answer, declared so the published contract
# carries every code a client branches on (FastAPI cannot infer a runtime
# raise). Bare statuses + the error shape only; the motives live in the code.
_TRANSITION_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: {"model": ErrorDetail},
    503: {"model": ErrorDetail},
}
_GUARD_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorDetail},
    404: {"model": ErrorDetail},
}


def _transition_rejection(error: SessionTransitionError) -> HTTPException:
    """Map a rejected session start/stop to its HTTP response.

    The session's current state is the motive. A model still loading
    (``CREATED``) is a transient unavailability, so it answers ``503`` with a
    ``Retry-After``; a terminated process is a permanent ``503``. Every other
    wrong-state — a session already running, one still closing, or nothing to
    stop — is a ``409`` conflict. The detail names the action and the state, so
    the caller sees exactly why the transition was refused.
    """
    state = error.state
    detail = f"cannot {error.action} session while {state.name.lower()}"
    if state is SessionState.CREATED:
        return HTTPException(
            status_code=503,
            detail=f"{detail}: the model is still loading",
            headers={"Retry-After": "1"},
        )
    if state is SessionState.TERMINATED:
        return HTTPException(status_code=503, detail=f"{detail}: the model is terminated")
    return HTTPException(status_code=409, detail=detail)


def _require_session(runner: Runner, sid: str) -> None:
    """Admit a request against *sid*, or answer 400 (no session) / 404 (wrong id)."""
    try:
        runner.require_session_running(sid)
    except SessionNotRunningError:
        raise HTTPException(status_code=400, detail="No session running") from None
    except UnknownSessionError:
        raise HTTPException(status_code=404, detail="Unknown session") from None


class SessionRoutes:
    """Session lifecycle control, driving the runner's session-control face."""

    def __init__(self, runner: Runner) -> None:
        """Bind the route group to the runner it drives."""
        self._runner = runner

    def mount(self, app: FastAPI) -> None:
        """Register the session-control routes against *app*."""
        runner = self._runner

        @app.post("/start_session", responses=_TRANSITION_RESPONSES)
        async def start_session(
            params: Annotated[dict[str, Any] | None, Body()] = None,
        ) -> dict[str, Any]:
            try:
                runner.start_session(params or {})
            except SessionTransitionError as rejected:
                raise _transition_rejection(rejected) from None
            return runner.descriptor()

        @app.get("/session")
        async def get_session() -> dict[str, Any]:
            return runner.descriptor()

        @app.get("/schema")
        async def get_schema() -> dict[str, Any]:
            return runner.schema()

        @app.post("/stop_session", responses=_TRANSITION_RESPONSES)
        async def stop_session(
            req: Annotated[StopSessionRequest | None, Body()] = None,
        ) -> Response:
            try:
                runner.stop_session(
                    moderated=req.moderate if req else False,
                    reason=req.reason if req else "",
                )
            except SessionTransitionError as rejected:
                raise _transition_rejection(rejected) from None
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


class CreateUploadResponse(BaseModel):
    """The reserved slot: its id, the URL to PUT the bytes to, and its path."""

    presigned_id: str
    presigned_url: str
    path: str


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

        @app.post(
            "/sessions/{sid}/uploads",
            status_code=201,
            responses={**_GUARD_RESPONSES, 409: {"model": ErrorDetail}},
        )
        async def create_upload(
            sid: str, req: CreateUploadRequest, request: Request
        ) -> CreateUploadResponse:
            _require_session(runner, sid)
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
            return CreateUploadResponse(
                presigned_id=upload_id,
                presigned_url=f"{base}/uploads/{upload_id}",
                path=f"sessions/{sid}/uploads/{upload_id}/{req.name}",
            )

        @app.put(
            "/uploads/{upload_id}",
            openapi_extra={
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/octet-stream": {
                            "schema": {"type": "string", "format": "binary"}
                        }
                    },
                }
            },
            responses={**_GUARD_RESPONSES, 409: {"model": ErrorDetail}},
        )
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


class RecordingRoutes:
    """Clip manifest and segment serving over HTTP.

    Two endpoints reproduce the clip contract the shipped client speaks: a
    ``GET /clips`` resolves a marker range to an HLS manifest (200), a
    not-ready hint (202), or gone (410), and a ``GET /clips/chunks/...`` serves
    the fMP4 segments the manifest points at, straight from local disk.
    """

    def __init__(self, runner: Runner) -> None:
        """Bind the route group to the runner whose recorder it reads."""
        self._runner = runner

    def mount(self, app: FastAPI) -> None:
        """Register the clip routes against *app*."""
        runner = self._runner

        # response_class=Response keeps FastAPI from advertising a default
        # application/json body these responses never carry.
        @app.get(
            "/clips",
            response_class=Response,
            responses={
                200: {
                    "description": "The clip's HLS manifest.",
                    "content": {"application/vnd.apple.mpegurl": {"schema": {"type": "string"}}},
                },
                202: {"description": "Still being prepared; retry later."},
                400: {"model": ErrorDetail},
                410: {"model": ErrorDetail},
            },
        )
        async def get_clip_manifest(session_id: str, start: float, end: float) -> Response:
            try:
                result = runner.recorder.manifest(session_id, start, end)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from None
            if isinstance(result, ClipManifest):
                return Response(content=result.body, media_type=result.media_type)
            if isinstance(result, Pending):
                return Response(status_code=202, headers={"Retry-After": str(result.retry_after)})
            raise HTTPException(status_code=410, detail="Recording unknown or aged out")

        @app.get(
            "/clips/chunks/{session_id}/{filename}",
            response_class=Response,
            responses={
                200: {
                    "description": "The fMP4 segment bytes.",
                    "content": {
                        "video/mp4": {"schema": {"type": "string", "format": "binary"}},
                        "video/iso.segment": {"schema": {"type": "string", "format": "binary"}},
                    },
                },
                404: {"model": ErrorDetail},
                410: {"model": ErrorDetail},
            },
        )
        async def get_clip_chunk(session_id: str, filename: str) -> FileResponse:
            try:
                path = runner.recorder.chunk_path(session_id, filename)
            except ClipSessionGoneError:
                detail = "Recording unknown or aged out"
                raise HTTPException(status_code=410, detail=detail) from None
            if path is None:
                raise HTTPException(status_code=404, detail="Segment not found")
            media_type = "video/mp4" if filename == "init.mp4" else "video/iso.segment"
            return FileResponse(path, media_type=media_type)


class HealthResponse(BaseModel):
    """The process verdict, the lifecycle word behind it, and why.

    ``status`` is the machine-checkable verdict a probe branches on; ``state``
    says what the process is doing, which a probe reads to tell a runtime that
    cannot serve yet from one that is finished; ``detail`` explains an unhealthy
    verdict.
    """

    status: HealthStatus
    state: RuntimeState
    detail: str | None


class EgressRoutes:
    """The egress journal, liveness, and the metrics scrape over HTTP."""

    def __init__(
        self,
        runner: Runner,
        process_health: Callable[[], Health],
        metrics: RuntimeMetrics,
    ) -> None:
        """Bind the route group to the runner it reads, the health source, and the metrics.

        Args:
            runner: The runner whose journal and lifecycle state the routes read.
            process_health: The health report ``/health`` answers with —
                injected so the endpoint speaks for the whole process, not for
                any one component.
            metrics: The registry ``/metrics`` renders, injected for the same
                reason: the assembly owns it, and every component that observes
                on it holds the one instance.
        """
        self._runner = runner
        self._process_health = process_health
        self._metrics = metrics

    def mount(self, app: FastAPI) -> None:
        """Register the egress, health, and metrics routes against *app*."""
        runner = self._runner
        process_health = self._process_health
        metrics = self._metrics

        @app.get(
            "/events",
            response_class=Response,
            responses={
                200: {
                    "description": "The egress journal as a server-sent event stream.",
                    "content": {"text/event-stream": {"schema": {"type": "string"}}},
                }
            },
        )
        async def events(
            since: int | None = None,
            last_event_id: Annotated[str | None, Header()] = None,
        ) -> StreamingResponse:
            resume = since if since is not None else _resume_from(last_event_id)
            return StreamingResponse(_stream_events(runner, resume), media_type="text/event-stream")

        @app.get("/health", responses={503: {"description": "The process is unhealthy."}})
        async def health(response: Response) -> HealthResponse:
            report = process_health()
            response.status_code = 503 if report.status is HealthStatus.UNHEALTHY else 200
            return HealthResponse(status=report.status, state=runner.state(), detail=report.detail)

        # A scraper reads this endpoint on its own schedule, and it answers from
        # the moment the socket binds — before the model loads — so a slow load
        # is itself observable rather than a gap in the series.
        @app.get(
            "/metrics",
            response_class=Response,
            responses={
                200: {
                    "description": "The process's instruments in the Prometheus text format.",
                    "content": {"text/plain": {"schema": {"type": "string"}}},
                }
            },
        )
        async def scrape_metrics() -> Response:
            return Response(content=metrics.render(), media_type=METRICS_CONTENT_TYPE)


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

    Each event carries its own sequence number, emitted as the SSE ``id``. The
    journal bounds its memory and may drop events for a consumer that falls
    behind, so the numbers are not necessarily contiguous: a jump is how a
    consumer learns it missed events and should reconcile from ``GET /session``.
    """
    async for seq, event in runner.events.subscribe(since):
        yield format_sse(seq, event)
