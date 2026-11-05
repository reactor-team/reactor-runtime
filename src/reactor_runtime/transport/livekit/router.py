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
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, SecretStr, field_validator

from reactor_runtime.core import ConnId, Health, HealthStatus, LiveKitConfig
from reactor_runtime.log import get_logger
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

logger = get_logger(__name__)

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
    the latest a caller opened. A room connection a later one replaces, in
    this session or the next, is kept until it has left its room, so its
    leave is awaited at shutdown (:meth:`settle`) and reported by
    :meth:`health` when the room stops answering.

    Args:
        config: The manifest's ``runtime.livekit`` block, which maps the
            room's own signals onto the model's commands and messages.
    """

    protocol = TRANSPORT_NAME
    version = TRANSPORT_VERSION

    def __init__(self, config: LiveKitConfig | None = None) -> None:
        self._config = config or LiveKitConfig()
        self._room: RoomSession | None = None
        self._room_epoch = 0
        # Replaced room connections that have not left their room yet.
        self._leaving: set[RoomSession] = set()
        # When the LiveKit server last answered a join or leave of this process.
        self._answered_at: float | None = None
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
                epoch = runner.session_epoch()
                previous = self._room
                if previous is not None and self._room_epoch == epoch:
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

                conn_id = self._claim(runner, req.connection_id)
                try:
                    runner.offer_admitted(conn_id)
                    room = RoomSession(
                        conn_id,
                        req.url,
                        req.token.get_secret_value(),
                        runner,
                        self._config,
                        on_answer=self._server_answered,
                        ice_servers=req.ice_servers or (),
                    )
                    room.start()
                except BaseException:
                    runner.release_conn_id(conn_id, self.protocol)
                    raise
                if previous is not None and not previous.settled:
                    self._leaving.add(previous)
                self._leaving = {r for r in self._leaving if not r.settled}
                self._room = room
                self._room_epoch = epoch
            return RoomConnectionAccepted(connection_id=conn_id)

        async def room_connection_status(sid: str, cid: int) -> Response:
            runner.require_session_running(sid)
            room = self._room_for(cid, runner.session_epoch())
            if room.joining:
                return Response(status_code=202)
            status = RoomConnectionStatus.model_validate(
                {"connection_id": cid, "state": room.state.value, "reason": room.reason}
            )
            return JSONResponse(status_code=200, content=status.model_dump(exclude_none=True))

        async def leave_room(sid: str, cid: int) -> Response:
            runner.require_session_running(sid)
            await self._room_for(cid, runner.session_epoch()).leave()
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

    async def settle(self) -> None:
        """Wait for every room connection this router holds to leave its room.

        Leaves are never cancelled here: a leave cut short may never reach the
        server, and the participant would stay in the room. Cancelled at the
        process's bound, it logs each room connection still leaving and for
        how long.
        """
        rooms = self._rooms()
        if not rooms:
            return
        try:
            await asyncio.gather(*(room.settle() for room in rooms))
        except asyncio.CancelledError:
            for room in self._rooms():
                age = room.leave_age()
                logger.warning(
                    "a LiveKit room connection was still leaving at shutdown",
                    conn_id=room.conn_id,
                    leaving_s=None if age is None else round(age, 1),
                )
            raise

    def health(self) -> Health:
        """Report unhealthy when a room connection is stuck while the LiveKit server answers others.

        A room connection is stuck once it has been leaving its room, or a
        timed-out join has held it, for :data:`~.room.LEAVE_STUCK_SECONDS`.
        One stuck while the server has answered a join or a leave of this
        process since it began means this process is losing room connections,
        and the deployment should replace it. A server that answers no one
        leaves every process healthy, however many are stuck: replacing them
        would end live sessions and fix nothing. ``/health`` answers 503 while
        this is unhealthy.
        """
        from reactor_runtime.transport.livekit.room import LEAVE_STUCK_SECONDS

        now = time.monotonic()
        ages = [age for room in self._rooms() if (age := room.leave_age()) is not None]
        stuck = [age for age in ages if age > LEAVE_STUCK_SECONDS]
        answered = self._answered_at
        if stuck and answered is not None and answered > now - max(stuck):
            return Health(
                HealthStatus.UNHEALTHY,
                f"{len(stuck)} LiveKit room connection(s) stuck leaving for over "
                f"{LEAVE_STUCK_SECONDS:.0f} s (the oldest {max(stuck):.0f} s) while the server "
                f"answered others, {now - answered:.0f} s ago",
            )
        return Health.healthy()

    def _server_answered(self) -> None:
        self._answered_at = time.monotonic()

    def _rooms(self) -> list[RoomSession]:
        """Return the room connections not yet settled, the current one included."""
        rooms = [room for room in self._leaving if not room.settled]
        if self._room is not None and self._room not in self._leaving and not self._room.settled:
            rooms.append(self._room)
        return rooms

    def _claim(self, runner: SessionControl, chosen: int | None) -> ConnId:
        """Return the room connection's id, claimed for this transport: *chosen*, or a minted one.

        A chosen id must be new to the session. One another transport holds
        is refused, and so is one this transport already holds: a room
        participant's, or an earlier room connection's.
        """
        if chosen is None:
            return runner.new_conn_id(self.protocol)
        conn_id = ConnId(chosen)
        try:
            claimed_now = runner.claim_conn_id(conn_id, self.protocol)
        except ConnectionIdClaimedError as claimed:
            raise HTTPException(status_code=409, detail=str(claimed)) from None
        if not claimed_now:
            raise HTTPException(
                status_code=409, detail=f"connection {conn_id} is already in use in this session"
            )
        return conn_id

    def _room_for(self, cid: int, epoch: int) -> RoomSession:
        """Return the live session's room connection *cid* names, or answer 404.

        A room an ended session opened is not the live session's connection,
        whatever its id.
        """
        room = self._room
        if room is None or room.conn_id != cid or self._room_epoch != epoch:
            raise HTTPException(status_code=404, detail="Unknown connection")
        return room
