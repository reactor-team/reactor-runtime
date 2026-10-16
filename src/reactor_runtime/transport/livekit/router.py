"""The LiveKit transport router.

Mounts the ``/sessions/{sid}/transport/livekit`` route group, shaped like the
WebRTC one. A caller opens a room connection with the room's URL and an access
token for the session's participant, and gets its connection id at once: the
join runs in the background, as a WebRTC negotiation does. The caller polls the
connection for the outcome of the join, and leaves the room by deleting it. A
session has at most one room connection at a time.

The LiveKit SDK ships as the runtime's ``livekit`` extra. This module imports
without it, but the room it joins needs it, so a runtime mounts these routes
only when :func:`livekit_available` reports the extra installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, SecretStr, field_validator

from reactor_runtime.transport.livekit.version import TRANSPORT_NAME, TRANSPORT_VERSION
from reactor_runtime.transport.router import (
    ErrorDetail,
    SessionControl,
    TransportRouter,
    add_session_error_handlers,
)

if TYPE_CHECKING:
    from reactor_runtime.transport.livekit.room import RoomSession

_PREFIX = "/sessions/{sid}/transport/livekit"

# Every route is guarded by require_session_running, and addresses one room
# connection the session knows.
_GUARD_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorDetail},
    404: {"model": ErrorDetail},
}

# Opening a room connection adds a 409 when the session already has one, and a
# 503 when the session's connection-id space is used up.
_OPEN_RESPONSES: dict[int | str, dict[str, Any]] = {
    **_GUARD_RESPONSES,
    409: {"model": ErrorDetail},
    503: {"model": ErrorDetail},
}


def livekit_available() -> bool:
    """Whether the LiveKit SDK, the runtime's ``livekit`` extra, is installed."""
    try:
        return importlib.util.find_spec("livekit.rtc") is not None
    except ModuleNotFoundError:
        return False


class RoomConnectionRequest(BaseModel):
    """Where the room is, and the credential the session's participant joins it with.

    The token is the access token of the session's participant: the room, the
    identity, and the grants come from it. It is never logged or journalled,
    and a refused request does not repeat it.
    """

    url: str
    token: SecretStr

    @field_validator("url")
    @classmethod
    def _url_is_a_websocket_url(cls, url: str) -> str:
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("url must be a ws:// or wss:// URL")
        return url


class RoomConnectionAccepted(BaseModel):
    """The room connection is accepted and joining in the background; poll for the outcome."""

    connection_id: int


class RoomConnectionStatus(BaseModel):
    """How a room connection's join ended, and whether the connection has closed since.

    ``open`` once the join succeeded and the room connection is a connection
    of the session; ``failed`` when the join did not succeed; ``closed`` once
    the room was left or lost. ``reason`` names the cause of a failure or a
    close.
    """

    connection_id: int
    state: Literal["open", "failed", "closed"]
    reason: str | None = None


class _UnechoedValidationRoute(APIRoute):
    """A route whose 422 names the fields at fault without repeating what was sent.

    The default 422 carries each failing value, and the whole body for a
    missing field. A room connection's body holds a credential, so its
    refusal carries only where and why validation failed.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as invalid:
                detail = [
                    {"type": error["type"], "loc": list(error["loc"]), "msg": error["msg"]}
                    for error in invalid.errors()
                ]
                return JSONResponse(status_code=422, content={"detail": detail})

        return route


class LiveKitRouter(TransportRouter):
    """Mount the LiveKit routes and join one room per room connection.

    Holds the session's room connection between requests: one at a time,
    the latest a caller opened.
    """

    protocol = TRANSPORT_NAME
    version = TRANSPORT_VERSION

    def __init__(self) -> None:
        self._room: RoomSession | None = None

    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        """Register the LiveKit route group against *app*, bound to *runner*."""
        add_session_error_handlers(app)

        async def open_room_connection(
            sid: str, req: RoomConnectionRequest
        ) -> RoomConnectionAccepted:
            runner.require_session_running(sid)
            if self._room is not None and self._room.active:
                raise HTTPException(
                    status_code=409, detail="the session already has a room connection"
                )
            from reactor_runtime.transport.livekit.room import RoomSession

            conn_id = runner.new_conn_id(self.protocol)
            runner.offer_admitted(conn_id)
            room = RoomSession(conn_id, req.url, req.token.get_secret_value(), runner)
            self._room = room
            room.start()
            return RoomConnectionAccepted(connection_id=conn_id)

        async def room_connection_status(sid: str, cid: int) -> Response:
            runner.require_session_running(sid)
            room = self._room_for(cid)
            if room.joining:
                return Response(status_code=202)
            status = RoomConnectionStatus.model_validate(
                {"connection_id": cid, "state": room.state.value, "reason": room.reason}
            )
            return JSONResponse(status_code=200, content=status.model_dump(exclude_none=True))

        async def leave_room(sid: str, cid: int) -> Response:
            runner.require_session_running(sid)
            await self._room_for(cid).leave()
            return Response(status_code=204)

        # Every route answers a malformed request without repeating it, since
        # the body that opens a room connection holds a credential.
        app.router.add_api_route(
            f"{_PREFIX}/connections",
            open_room_connection,
            methods=["POST"],
            status_code=202,
            responses=_OPEN_RESPONSES,
            route_class_override=_UnechoedValidationRoute,
        )
        app.router.add_api_route(
            f"{_PREFIX}/connections/{{cid}}",
            room_connection_status,
            methods=["GET"],
            responses={
                200: {"model": RoomConnectionStatus},
                202: {"description": "The join is still in flight; poll again."},
                **_GUARD_RESPONSES,
            },
            route_class_override=_UnechoedValidationRoute,
        )
        app.router.add_api_route(
            f"{_PREFIX}/connections/{{cid}}",
            leave_room,
            methods=["DELETE"],
            status_code=204,
            responses=_GUARD_RESPONSES,
            route_class_override=_UnechoedValidationRoute,
        )

    def _room_for(self, cid: int) -> RoomSession:
        """Return the room connection *cid* names, or answer 404."""
        room = self._room
        if room is None or room.conn_id != cid:
            raise HTTPException(status_code=404, detail="Unknown connection")
        return room
