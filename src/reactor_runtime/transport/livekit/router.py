"""The LiveKit transport router.

Mounts the ``/sessions/{sid}/transport/livekit`` route group, shaped like the
WebRTC one. A caller opens a room connection with the room's URL and an access
token for the session's participant, and gets its connection id at once: the
join runs in the background, as a WebRTC negotiation does. The caller may
choose the id, as a WebRTC client chooses the id it offers on, or leave the
runtime to mint it. The caller polls the connection for the outcome of the
join, and leaves the room by deleting it. A session has at most one room
connection at a time.

The LiveKit SDK ships as the runtime's ``livekit`` extra. This module imports
without it, but the room it joins needs it, so a runtime mounts these routes
only when :func:`livekit_available` reports the extra installed.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Literal, cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, SecretStr, field_validator

from reactor_runtime.core import ConnId, LiveKitConfig
from reactor_runtime.transport.livekit.version import TRANSPORT_NAME, TRANSPORT_VERSION
from reactor_runtime.transport.router import (
    CHOSEN_CONN_ID_MIN,
    CONN_ID_MAX,
    ConnectionIdClaimedError,
    ErrorDetail,
    IceServerEntry,
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

# Opening a room connection adds a 409 when the session already has one or the
# id the caller chose is in use, and a 503 when the session's connection-id
# space is used up.
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

    ``connection_id`` lets the caller choose the room connection's id: any id
    from 1002 to 9999. Absent, the runtime mints one from its own range, 5000
    to 9999, so a caller that mints its ids itself keeps to 1002 to 4999. An id
    the session already uses, through this transport or another, is refused
    with a 409, since a room connection is never opened again on the same id.

    ``ice_servers`` lets the caller supply the STUN/TURN servers the room
    connection gathers against, for a network where the servers the LiveKit
    server offers do not reach it. Given, they replace those servers; absent or
    empty, the room connection gathers against the LiveKit server's own. Their
    TURN credentials are treated like the token: never logged or journalled,
    and not repeated by a refused request.
    """

    url: str
    token: SecretStr
    connection_id: int | None = Field(default=None, ge=CHOSEN_CONN_ID_MIN, le=CONN_ID_MAX)
    ice_servers: list[IceServerEntry] | None = None

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
    close, and never repeats the room's token.
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

    Args:
        config: The manifest's ``runtime.livekit`` block, which maps the
            room's own signals onto the model's commands and messages.
    """

    protocol = TRANSPORT_NAME
    version = TRANSPORT_VERSION

    def __init__(self, config: LiveKitConfig | None = None) -> None:
        self._config = config or LiveKitConfig()
        # Each session's latest room connection, by session id: the room, the
        # session epoch it was opened in, and the id the caller addresses it by.
        self._rooms: dict[str, tuple[RoomSession, int, int]] = {}
        # One open at a time: retiring an ended session's room yields, and a
        # second open must not pass the check meanwhile.
        self._opening = asyncio.Lock()

    def mount(self, app: FastAPI, runner: SessionControl) -> None:
        """Register the LiveKit route group against *app*, bound to *runner*."""
        add_session_error_handlers(app)

        async def open_room_connection(
            sid: str, req: RoomConnectionRequest
        ) -> RoomConnectionAccepted:
            runner.require_session_running(sid)
            async with self._opening:
                self._forget_settled()
                epoch = runner.session_epoch(sid)
                held = self._rooms.get(sid)
                previous = held[0] if held is not None else None
                if previous is not None and held is not None and held[1] == epoch:
                    if previous.active:
                        raise HTTPException(
                            status_code=409, detail="the session already has a room connection"
                        )
                    if not previous.settled:
                        # A join that timed out or was left still holds the room until it settles.
                        raise HTTPException(
                            status_code=409,
                            detail="the session's last room connection is still leaving the room",
                        )
                elif previous is not None and previous.active:
                    # Admitted by a session that has ended: its room is retired, not reused.
                    await previous.retire()
                from reactor_runtime.transport.livekit.room import RoomSession

                cid = self._claim(runner, sid, req.connection_id)
                conn_id = runner.transport_conn_id(sid, cid)
                try:
                    runner.offer_admitted(conn_id)
                    room = RoomSession(
                        conn_id,
                        req.url,
                        req.token.get_secret_value(),
                        # Forwards every other member of the surface to the runtime's own.
                        cast(SessionControl, _SessionControl(runner, sid)),
                        self._config,
                        ice_servers=req.ice_servers or (),
                    )
                    room.start()
                except BaseException:
                    runner.release_conn_id(conn_id, self.protocol)
                    raise
                self._rooms[sid] = (room, epoch, cid)
            return RoomConnectionAccepted(connection_id=cid)

        async def room_connection_status(sid: str, cid: int) -> Response:
            runner.require_session_running(sid)
            room = self._room_for(sid, cid, runner.session_epoch(sid))
            if room.joining:
                return Response(status_code=202)
            status = RoomConnectionStatus.model_validate(
                {"connection_id": cid, "state": room.state.value, "reason": room.reason}
            )
            return JSONResponse(status_code=200, content=status.model_dump(exclude_none=True))

        async def leave_room(sid: str, cid: int) -> Response:
            runner.require_session_running(sid)
            await self._room_for(sid, cid, runner.session_epoch(sid)).leave()
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

    def _claim(self, runner: SessionControl, sid: str, chosen: int | None) -> int:
        """Return the room connection's id in session *sid*, claimed for this transport.

        *chosen*, or one minted within the session. A chosen id must be new to
        the session. One another transport holds is refused, and so is one this
        transport already holds: a room participant's, or an earlier room
        connection's.
        """
        if chosen is None:
            return int(runner.new_conn_id(self.protocol, sid))
        conn_id = runner.transport_conn_id(sid, chosen)
        try:
            claimed_now = runner.claim_conn_id(conn_id, self.protocol)
        except ConnectionIdClaimedError as claimed:
            raise HTTPException(status_code=409, detail=str(claimed)) from None
        if not claimed_now:
            raise HTTPException(
                status_code=409, detail=f"connection {chosen} is already in use in this session"
            )
        return chosen

    def _room_for(self, sid: str, cid: int, epoch: int) -> RoomSession:
        """Return session *sid*'s room connection *cid* names, or answer 404.

        A room an ended session opened is not the live session's connection,
        whatever its id.
        """
        held = self._rooms.get(sid)
        if held is None or held[2] != cid or held[1] != epoch:
            raise HTTPException(status_code=404, detail="Unknown connection")
        return held[0]

    def _forget_settled(self) -> None:
        """Drop the room connections that are over, so ended sessions hold none."""
        for sid, (room, _, _) in list(self._rooms.items()):
            if room.settled:
                del self._rooms[sid]


class _SessionControl:
    """The session-control surface as one session's room sees it.

    A room mints an id for every participant that becomes a client. In a
    runtime hosting several sessions those ids are minted within the room's own
    session and keyed for the transport, as the room connection's id is; every
    other call is the runtime's own.
    """

    def __init__(self, runner: SessionControl, sid: str) -> None:
        self._runner = runner
        self._sid = sid

    def new_conn_id(self, transport: str, sid: str | None = None) -> ConnId:
        local = self._runner.new_conn_id(transport, self._sid)
        return self._runner.transport_conn_id(self._sid, int(local))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)
