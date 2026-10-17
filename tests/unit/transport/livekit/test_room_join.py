import asyncio
import gc
import weakref
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from tests.unit.transport.livekit.test_livekit_router import FakeRunner as RouterRunner

from reactor_runtime.core import ConnId, TrackDirection, TrackKind
from reactor_runtime.transport.livekit import LiveKitRouter
from reactor_runtime.transport.livekit import room as room_module
from reactor_runtime.transport.livekit.room import RoomSession, RoomState

_VIDEO_OUT = {
    "main_video": {
        "kind": TrackKind.VIDEO.value,
        "direction": TrackDirection.OUT.value,
        "rate": 30.0,
    }
}


class Gate:
    """A step the test lets finish, or fail, when it chooses."""

    def __init__(self, *, open_: bool = True) -> None:
        self._open = asyncio.Event()
        self.error: Exception | None = None
        self.entered = False
        self.cancelled = False
        if open_:
            self._open.set()

    def release(self, error: Exception | None = None) -> None:
        self.error = error
        self._open.set()

    async def pass_through(self) -> None:
        self.entered = True
        try:
            await self._open.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error


class FakeParticipant:
    identity = "reactor-model"

    def __init__(self, publish: Gate) -> None:
        self._publish = publish

    async def publish_track(self, track: Any, options: Any) -> None:
        await self._publish.pass_through()


class FakeRoom:
    """Stands in for ``rtc.Room``: connect and publish wait on gates."""

    name = "the-room"

    def __init__(self, *, connect: Gate | None = None, publish: Gate | None = None) -> None:
        self.connect_gate = connect or Gate()
        self.local_participant = FakeParticipant(publish or Gate())
        self.remote_participants: dict[str, Any] = {}
        self.disconnects = 0
        self.handlers: dict[str, Any] = {}
        self.stream_handlers: dict[str, Any] = {}

    def on(self, event: str, callback: Any) -> None:
        self.handlers[event] = callback

    def register_byte_stream_handler(self, topic: str, handler: Any) -> None:
        self.stream_handlers[topic] = handler

    async def connect(self, url: str, token: str) -> None:
        await self.connect_gate.pass_through()

    async def disconnect(self) -> None:
        self.disconnects += 1


class FakeControl:
    def __init__(self, tracks: dict[str, Any] | None = None) -> None:
        self._tracks = tracks or {}
        self.opened: list[Any] = []
        self.closed: list[ConnId] = []
        self.failed: list[tuple[ConnId, str, str]] = []

    def track_map(self) -> dict[str, Any]:
        return self._tracks

    def connection_opened(self, conn: Any) -> bool:
        self.opened.append(conn)
        return True

    def connection_closed(self, conn_id: ConnId) -> None:
        self.closed.append(conn_id)

    def connection_failed(self, conn_id: ConnId, transport: str, reason: str) -> None:
        self.failed.append((conn_id, transport, reason))


def _session(room: FakeRoom, control: FakeControl) -> RoomSession:
    return RoomSession(
        ConnId(4711), "wss://rooms.example", "a-token", cast(Any, control), room=cast(Any, room)
    )


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


@pytest.fixture
def short_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(room_module, "JOIN_TIMEOUT_SECONDS", 0.05)


async def test_a_join_that_connects_and_publishes_opens_the_connection() -> None:
    room, control = FakeRoom(), FakeControl()
    session = _session(room, control)

    session.start()
    await _settle()

    assert session.state is RoomState.OPEN
    (conn,) = control.opened
    assert conn.id == 4711
    await session.leave()
    assert (control.closed, room.disconnects) == ([ConnId(4711)], 1)


async def test_a_refused_connect_fails_the_join_and_leaves_the_room() -> None:
    room, control = FakeRoom(), FakeControl()
    room.connect_gate.release(RuntimeError("unauthorized"))
    session = _session(room, control)

    session.start()
    await _settle()

    assert session.state is RoomState.FAILED
    assert session.reason == "the LiveKit room could not be joined: unauthorized"
    assert control.failed == [(ConnId(4711), "livekit", session.reason)]
    assert room.disconnects == 1


@pytest.mark.usefixtures("short_deadline")
async def test_a_publish_that_never_completes_times_the_join_out() -> None:
    publish = Gate(open_=False)
    room, control = FakeRoom(publish=publish), FakeControl(_VIDEO_OUT)
    session = _session(room, control)

    session.start()
    await asyncio.sleep(0.1)

    assert session.state is RoomState.FAILED
    assert session.reason == "joining the LiveKit room timed out"
    assert control.failed == [(ConnId(4711), "livekit", "joining the LiveKit room timed out")]
    assert control.opened == []
    # The publish is still in flight: it is never cancelled, and the room is not yet left.
    assert (publish.cancelled, room.disconnects) == (False, 0)

    publish.release()
    await _settle()

    assert room.disconnects == 1
    assert control.opened == []


@pytest.mark.usefixtures("short_deadline")
async def test_a_timed_out_attempt_that_later_fails_is_left_without_a_lost_error() -> None:
    connect = Gate(open_=False)
    room, control = FakeRoom(connect=connect), FakeControl()
    session = _session(room, control)
    lost: list[dict[str, Any]] = []
    asyncio.get_running_loop().set_exception_handler(lambda _, context: lost.append(context))

    session.start()
    await asyncio.sleep(0.1)
    connect.release(RuntimeError("the server went away"))
    await _settle()

    assert room.disconnects == 1
    assert len(control.failed) == 1
    gc.collect()
    assert lost == []


async def test_leaving_while_joining_waits_for_the_connect_before_leaving_the_room() -> None:
    connect = Gate(open_=False)
    room, control = FakeRoom(connect=connect), FakeControl()
    session = _session(room, control)
    session.start()
    await _settle()
    assert connect.entered

    await session.leave()

    assert (session.state, session.reason) == (RoomState.CLOSED, "the room was left on request")
    assert (connect.cancelled, room.disconnects) == (False, 0)

    connect.release()
    await _settle()

    assert room.disconnects == 1
    assert control.opened == []
    assert control.failed == []


async def test_retiring_a_join_ends_it_with_the_session() -> None:
    connect = Gate(open_=False)
    room, control = FakeRoom(connect=connect), FakeControl()
    session = _session(room, control)
    session.start()
    await _settle()

    await session.retire()
    connect.release()
    await _settle()

    assert (session.state, session.reason) == (RoomState.CLOSED, "the session ended")
    assert room.disconnects == 1
    assert control.opened == []


@pytest.mark.usefixtures("short_deadline")
async def test_a_timed_out_join_is_settled_only_once_its_attempt_has_left() -> None:
    publish = Gate(open_=False)
    room, control = FakeRoom(publish=publish), FakeControl(_VIDEO_OUT)
    session = _session(room, control)

    session.start()
    await asyncio.sleep(0.1)
    assert session.state is RoomState.FAILED
    assert not session.settled

    publish.release()
    await _settle()

    assert room.disconnects == 1
    assert session.settled


@pytest.mark.usefixtures("short_deadline")
async def test_a_timed_out_join_is_unsettled_from_its_attempt_ending_until_the_room_is_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish, leave = Gate(open_=False), Gate(open_=False)
    room, control = FakeRoom(publish=publish), FakeControl(_VIDEO_OUT)

    async def disconnect() -> None:
        await leave.pass_through()
        room.disconnects += 1

    monkeypatch.setattr(room, "disconnect", disconnect)
    session = _session(room, control)
    leave_when_settled = session._leave_when_settled
    seen_as_the_attempt_ended: list[bool] = []

    def spy(attempt: asyncio.Task[Any]) -> None:
        seen_as_the_attempt_ended.append(session.settled)
        leave_when_settled(attempt)

    monkeypatch.setattr(session, "_leave_when_settled", spy)
    session.start()
    await asyncio.sleep(0.1)
    publish.release()
    await _settle()

    assert seen_as_the_attempt_ended == [False]
    assert leave.entered
    assert not session.settled

    leave.release()
    await _settle()

    assert room.disconnects == 1
    assert session.settled


@pytest.mark.usefixtures("short_deadline")
async def test_a_join_holds_no_room_connection_once_it_has_left_the_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish = Gate(open_=False)
    room, control = FakeRoom(publish=publish), FakeControl(_VIDEO_OUT)
    session = _session(room, control)
    built: list[weakref.ref[Any]] = []
    publish_tracks = session._publish

    async def spy(tracks: Any) -> Any:
        connection = await publish_tracks(tracks)
        built.append(weakref.ref(connection))
        return connection

    monkeypatch.setattr(session, "_publish", spy)
    session.start()
    await asyncio.sleep(0.1)  # past the deadline
    publish.release()  # the abandoned attempt succeeds late
    await _settle()

    assert session.settled
    gc.collect()
    assert [ref() for ref in built] == [None]


async def test_a_left_room_holds_no_room_connection() -> None:
    room, control = FakeRoom(), FakeControl(_VIDEO_OUT)
    session = _session(room, control)
    session.start()
    await _settle()
    (connection,) = control.opened
    ref = weakref.ref(connection)

    await session.leave()
    control.opened.clear()
    del connection
    gc.collect()

    assert ref() is None


async def test_a_join_left_mid_connect_is_settled_only_once_it_has_left() -> None:
    connect = Gate(open_=False)
    room, control = FakeRoom(connect=connect), FakeControl()
    session = _session(room, control)
    session.start()
    await _settle()

    await session.leave()
    assert not session.settled

    connect.release()
    await _settle()

    assert room.disconnects == 1
    assert session.settled


@pytest.mark.usefixtures("short_deadline")
async def test_the_session_reopens_a_timed_out_room_only_once_the_first_join_has_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = Gate(open_=False)
    first = FakeRoom(connect=connect)
    rooms = iter([first, FakeRoom()])
    monkeypatch.setattr(room_module.rtc, "Room", lambda: next(rooms))
    app = FastAPI()
    LiveKitRouter().mount(app, cast(Any, RouterRunner()))
    path = "/sessions/s1/transport/livekit/connections"
    body = {"url": "wss://rooms.example", "token": "a-token"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as client:
        assert (await client.post(path, json=body)).status_code == 202
        await asyncio.sleep(0.1)  # past the deadline: the join failed, its connect still runs

        while_leaving = await client.post(path, json=body)
        connect.release()
        await _settle()
        after = await client.post(path, json=body)

    assert while_leaving.status_code == 409
    assert first.disconnects == 1
    assert after.status_code == 202
