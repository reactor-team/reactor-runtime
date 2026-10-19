import asyncio
import importlib.util
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from reactor_runtime.core import (
    ClientStatsBatch,
    Connection,
    ConnId,
    InputFrame,
    LiveKitConfig,
    LiveKitRpcMethod,
    ReplyRoute,
)
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport import SessionNotRunningError, UnknownSessionError
from reactor_runtime.transport.livekit import LiveKitRouter, livekit_available
from reactor_runtime.transport.livekit import room as room_module
from reactor_runtime.transport.livekit.room import RoomState

_SID = "s1"
_CONNECTIONS = f"/sessions/{_SID}/transport/livekit/connections"
_TOKEN = "eyJ.a-credential.sig"
_ROOM = {"url": "wss://rooms.example", "token": _TOKEN}


class FakeRunner:
    """A SessionControl that mints ids and records what the router asks of it."""

    def __init__(self, *, running: bool = True) -> None:
        self._running = running
        self.minted: list[tuple[ConnId, str]] = []
        self.admitted: list[ConnId] = []
        self.epoch = 1

    def require_session_running(self, sid: str) -> None:
        if not self._running:
            raise SessionNotRunningError
        if sid != _SID:
            raise UnknownSessionError

    def new_conn_id(self, transport: str) -> ConnId:
        conn_id = ConnId(4711 + len(self.minted))
        self.minted.append((conn_id, transport))
        return conn_id

    def claim_conn_id(self, conn_id: ConnId, transport: str) -> bool:
        return True

    def release_conn_id(self, conn_id: ConnId, transport: str) -> None:
        pass

    def connection_failed(self, conn_id: ConnId, transport: str, reason: str) -> None:
        pass

    def session_epoch(self) -> int:
        return self.epoch

    def offer_admitted(self, conn_id: ConnId) -> None:
        self.admitted.append(conn_id)

    def track_map(self) -> dict[str, Any]:
        return {}

    def connection_opened(self, conn: Connection) -> bool:
        return True

    def connection_closed(self, conn_id: ConnId) -> None:
        pass

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        pass

    def command_received(
        self,
        conn_id: ConnId,
        name: str,
        args: Mapping[str, Any],
        request_id: str,
        reply: ReplyRoute = ReplyRoute.PROTOCOL,
    ) -> None:
        pass

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        pass

    def keepalive(self, conn_id: ConnId) -> None:
        pass

    def resume_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def pause_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def publish_requested(self, conn_id: ConnId, name: str, request_id: str) -> None:
        pass

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        pass

    def schema_requested(self, conn_id: ConnId, request_id: str) -> None:
        pass

    def clip_requested(self, conn_id: ConnId, duration_seconds: float, request_id: str) -> None:
        pass

    def recording_requested(self, conn_id: ConnId, request_id: str) -> None:
        pass

    def client_stats_received(self, conn_id: ConnId, batch: ClientStatsBatch) -> None:
        pass

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        pass


class FakeRoomSession:
    """Stands in for a room connection: records how it was built and driven."""

    def __init__(
        self,
        conn_id: ConnId,
        url: str,
        token: str,
        control: Any,
        config: LiveKitConfig | None = None,
    ) -> None:
        self.conn_id = conn_id
        self.config = config
        self.url = url
        self.token = token
        self.state = RoomState.JOINING
        self.reason: str | None = None
        self.started = False
        self.left = False
        self.retired = False
        # An abandoned join still connecting or leaving.
        self.leaving = False
        _built.append(self)

    @property
    def joining(self) -> bool:
        return self.state is RoomState.JOINING

    @property
    def active(self) -> bool:
        return self.state in (RoomState.JOINING, RoomState.OPEN)

    @property
    def settled(self) -> bool:
        return not self.active and not self.leaving

    def start(self) -> None:
        self.started = True

    async def leave(self) -> None:
        self.left = True
        self.state = RoomState.CLOSED
        self.reason = "the room was left on request"

    async def retire(self) -> None:
        self.retired = True
        self.state = RoomState.CLOSED
        self.reason = "the session ended"


_built: list[FakeRoomSession] = []


@pytest.fixture
def rooms(monkeypatch: pytest.MonkeyPatch) -> list[FakeRoomSession]:
    _built.clear()
    monkeypatch.setattr(room_module, "RoomSession", FakeRoomSession)
    return _built


def _client(runner: FakeRunner, config: LiveKitConfig | None = None) -> TestClient:
    app = FastAPI()
    LiveKitRouter(config).mount(app, runner)
    return TestClient(app)


def test_the_router_names_its_transport() -> None:
    assert (LiveKitRouter.protocol, LiveKitRouter.version) == ("livekit", "1.0")


def test_opening_a_room_connection_mints_its_id_and_starts_the_join(
    rooms: list[FakeRoomSession],
) -> None:
    runner = FakeRunner()

    response = _client(runner).post(_CONNECTIONS, json=_ROOM)

    assert response.status_code == 202
    assert response.json() == {"connection_id": 4711}
    assert runner.minted == [(ConnId(4711), "livekit")]
    assert runner.admitted == [ConnId(4711)]
    (room,) = rooms
    assert (room.conn_id, room.url, room.token, room.started) == (4711, *_ROOM.values(), True)


def test_the_room_connection_maps_the_rooms_signals_as_the_manifest_says(
    rooms: list[FakeRoomSession],
) -> None:
    signals = LiveKitConfig(rpc=(LiveKitRpcMethod(method="lk.stop", command="stop"),))

    _client(FakeRunner(), signals).post(_CONNECTIONS, json=_ROOM)

    assert rooms[0].config is signals


def test_a_second_room_connection_is_refused_while_one_is_active(
    rooms: list[FakeRoomSession],
) -> None:
    client = _client(FakeRunner())
    client.post(_CONNECTIONS, json=_ROOM)

    refused = client.post(_CONNECTIONS, json=_ROOM)

    assert refused.status_code == 409
    assert refused.json() == {"detail": "the session already has a room connection"}
    assert len(rooms) == 1


def test_a_join_from_an_ended_session_is_retired_by_the_next_session(
    rooms: list[FakeRoomSession],
) -> None:
    runner = FakeRunner()
    client = _client(runner)
    client.post(_CONNECTIONS, json=_ROOM)
    runner.epoch = 2

    again = client.post(_CONNECTIONS, json=_ROOM)

    assert again.status_code == 202
    assert again.json() == {"connection_id": 4712}
    stale, fresh = rooms
    assert (stale.retired, stale.state) == (True, RoomState.CLOSED)
    assert fresh.started
    assert client.post(_CONNECTIONS, json=_ROOM).status_code == 409


async def test_concurrent_opens_while_an_ended_sessions_room_retires_make_one_room(
    rooms: list[FakeRoomSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRunner()
    app = FastAPI()
    LiveKitRouter().mount(app, runner)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as client:
        await client.post(_CONNECTIONS, json=_ROOM)
        runner.epoch = 2
        stale = rooms[0]
        retiring = asyncio.Event()

        async def slow_retire() -> None:
            await retiring.wait()
            stale.state = RoomState.CLOSED

        monkeypatch.setattr(stale, "retire", slow_retire)
        first = asyncio.create_task(client.post(_CONNECTIONS, json=_ROOM))
        second = asyncio.create_task(client.post(_CONNECTIONS, json=_ROOM))
        await asyncio.sleep(0.05)
        retiring.set()
        statuses = sorted([(await first).status_code, (await second).status_code])

    assert statuses == [202, 409]
    assert len(rooms) == 2


def test_a_room_whose_abandoned_join_still_runs_keeps_the_session_at_409(
    rooms: list[FakeRoomSession],
) -> None:
    client = _client(FakeRunner())
    client.post(_CONNECTIONS, json=_ROOM)
    rooms[0].state = RoomState.FAILED
    rooms[0].leaving = True

    refused = client.post(_CONNECTIONS, json=_ROOM)
    rooms[0].leaving = False
    again = client.post(_CONNECTIONS, json=_ROOM)

    assert refused.status_code == 409
    assert refused.json() == {
        "detail": "the session's last room connection is still leaving the room"
    }
    assert again.status_code == 202


def test_a_room_connection_can_open_again_once_the_last_one_ended(
    rooms: list[FakeRoomSession],
) -> None:
    client = _client(FakeRunner())
    client.post(_CONNECTIONS, json=_ROOM)
    rooms[0].state = RoomState.FAILED

    again = client.post(_CONNECTIONS, json=_ROOM)

    assert again.status_code == 202
    assert again.json() == {"connection_id": 4712}


def test_the_status_is_pending_while_the_join_runs_then_reports_its_outcome(
    rooms: list[FakeRoomSession],
) -> None:
    client = _client(FakeRunner())
    client.post(_CONNECTIONS, json=_ROOM)

    assert client.get(f"{_CONNECTIONS}/4711").status_code == 202

    rooms[0].state = RoomState.OPEN
    opened = client.get(f"{_CONNECTIONS}/4711")
    assert (opened.status_code, opened.json()) == (200, {"connection_id": 4711, "state": "open"})

    rooms[0].state = RoomState.FAILED
    rooms[0].reason = "the LiveKit room could not be joined: unauthorized"
    failed = client.get(f"{_CONNECTIONS}/4711")
    assert failed.json() == {
        "connection_id": 4711,
        "state": "failed",
        "reason": "the LiveKit room could not be joined: unauthorized",
    }


def test_deleting_the_room_connection_leaves_the_room(rooms: list[FakeRoomSession]) -> None:
    client = _client(FakeRunner())
    client.post(_CONNECTIONS, json=_ROOM)

    response = client.delete(f"{_CONNECTIONS}/4711")

    assert response.status_code == 204
    assert rooms[0].left


@pytest.mark.parametrize("method", ["get", "delete"])
def test_a_room_an_ended_session_opened_is_not_found_in_the_next(
    rooms: list[FakeRoomSession], method: str
) -> None:
    runner = FakeRunner()
    client = _client(runner)
    client.post(_CONNECTIONS, json=_ROOM)
    runner.epoch = 2

    response = client.request(method.upper(), f"{_CONNECTIONS}/4711")

    assert response.status_code == 404
    assert not rooms[0].left


@pytest.mark.parametrize("method", ["get", "delete"])
def test_an_unknown_room_connection_is_not_found(rooms: list[FakeRoomSession], method: str) -> None:
    client = _client(FakeRunner())
    client.post(_CONNECTIONS, json=_ROOM)

    response = client.request(method.upper(), f"{_CONNECTIONS}/9999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown connection"}


def test_every_route_answers_400_without_a_running_session(rooms: list[FakeRoomSession]) -> None:
    client = _client(FakeRunner(running=False))

    assert client.post(_CONNECTIONS, json=_ROOM).status_code == 400
    assert client.get(f"{_CONNECTIONS}/4711").status_code == 400
    assert client.delete(f"{_CONNECTIONS}/4711").status_code == 400
    assert rooms == []


def test_another_session_id_is_not_found(rooms: list[FakeRoomSession]) -> None:
    response = _client(FakeRunner()).post(
        "/sessions/other/transport/livekit/connections", json=_ROOM
    )

    assert response.status_code == 404
    assert rooms == []


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"token": _TOKEN}, "url"),
        ({"url": "https://rooms.example", "token": _TOKEN}, "url"),
        ({"url": "wss://rooms.example", "token": [_TOKEN]}, "token"),
        ({"url": 7, "token": _TOKEN}, "url"),
    ],
)
def test_a_malformed_body_is_refused_without_repeating_the_token(
    rooms: list[FakeRoomSession], body: dict[str, Any], field: str
) -> None:
    response = _client(FakeRunner()).post(_CONNECTIONS, json=body)

    assert response.status_code == 422
    assert _TOKEN not in response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["body", field]]
    assert rooms == []


def test_livekit_is_unavailable_when_the_sdk_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

    monkeypatch.setattr(importlib.util, "find_spec", missing)

    assert livekit_available() is False


def test_livekit_is_available_in_the_development_environment() -> None:
    assert livekit_available() is True
