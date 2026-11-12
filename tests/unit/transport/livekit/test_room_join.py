import asyncio
import gc
import logging
import weakref
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from tests.unit.transport.livekit.test_livekit_router import FakeRunner as RouterRunner

from reactor_runtime.core import ConnId, HealthStatus, TrackDirection, TrackKind
from reactor_runtime.transport.livekit import LiveKitRouter
from reactor_runtime.transport.livekit import room as room_module
from reactor_runtime.transport.livekit.room import RoomSession, RoomState, _masked
from reactor_runtime.transport.livekit.router import IceForwarderEntry, ice_forwarder_available
from reactor_runtime.transport.router import IceServerEntry, TurnCredentials

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
        self.methods: dict[str, Any] = {}

    def register_rpc_method(self, method: str, handler: Any) -> None:
        self.methods[method] = handler

    def unregister_rpc_method(self, method: str) -> None:
        self.methods.pop(method, None)

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
        self.connect_options: list[Any] = []
        self.handlers: dict[str, Any] = {}
        self.stream_handlers: dict[str, Any] = {}

    def on(self, event: str, callback: Any) -> None:
        self.handlers[event] = callback

    def register_byte_stream_handler(self, topic: str, handler: Any) -> None:
        self.stream_handlers[topic] = handler

    async def connect(self, url: str, token: str, options: Any = None) -> None:
        self.connect_options.append(options)
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

    def media_received(self, conn_id: ConnId, track: str, frame: Any) -> None:
        pass

    def claim_input_track(self, conn_id: ConnId, name: str) -> bool:
        return True

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        pass


_TURN = IceServerEntry(
    uris=["turn:turn.example:3478?transport=udp"],
    credentials=TurnCredentials(username="turn-user", password="turn-pass-word"),
)
_STUN = IceServerEntry(uris=["stun:stun.example:3478"])


_FORWARDER = IceForwarderEntry.model_validate(
    {
        "addresses": ["203.0.113.10:443", "[2001:db8::1]:443"],
        "ice_credentials": {
            "ufrag": "z04yKzdOjjjhPoE5bQE0/NbtKYLwqvo",
            "pwd": "5H4joxyoYh0QO3jpq+iQVPqCuDWPsp4I",
        },
        "port": 50001,
    }
)


def _session(
    room: FakeRoom,
    control: FakeControl,
    ice_servers: tuple[IceServerEntry, ...] = (),
    ice_forwarder: IceForwarderEntry | None = None,
) -> RoomSession:
    return RoomSession(
        ConnId(4711),
        "wss://rooms.example",
        "a-token",
        cast(Any, control),
        room=cast(Any, room),
        ice_servers=ice_servers,
        ice_forwarder=ice_forwarder,
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


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (
            "failed to connect to wss://rooms.example/rtc?access_token=a-token&auto_subscribe=1",
            "failed to connect to wss://rooms.example/rtc?access_token=[redacted]&auto_subscribe=1",
        ),
        ("the server refused a-token", "the server refused [redacted]"),
        (
            "dialled ws://other.example/rtc?access_token=another-token",
            "dialled ws://other.example/rtc?access_token=[redacted]",
        ),
    ],
)
async def test_a_refused_connect_keeps_the_token_out_of_the_reason(error: str, reason: str) -> None:
    room, control = FakeRoom(), FakeControl()
    room.connect_gate.release(RuntimeError(error))
    session = _session(room, control)

    session.start()
    await _settle()

    assert session.reason == f"the LiveKit room could not be joined: {reason}"
    assert control.failed == [(ConnId(4711), "livekit", session.reason)]


async def test_the_join_gathers_against_the_caller_s_ice_servers() -> None:
    room, control = FakeRoom(), FakeControl()
    room.connect_gate.release()
    session = _session(room, control, ice_servers=(_TURN, _STUN))

    session.start()
    await _settle()

    (options,) = room.connect_options
    servers = [(list(s.urls), s.username, s.password) for s in options.rtc_config.ice_servers]
    assert servers == [
        (["turn:turn.example:3478?transport=udp"], "turn-user", "turn-pass-word"),
        (["stun:stun.example:3478"], "", ""),
    ]
    await session.leave()


async def test_without_ice_servers_the_join_passes_no_options() -> None:
    room, control = FakeRoom(), FakeControl()
    room.connect_gate.release()
    session = _session(room, control)

    session.start()
    await _settle()

    assert room.connect_options == [None]
    await session.leave()


async def test_a_refused_connect_keeps_the_turn_credentials_out_of_the_reason() -> None:
    room, control = FakeRoom(), FakeControl()
    room.connect_gate.release(RuntimeError("turn-user:turn-pass-word was refused"))
    session = _session(room, control, ice_servers=(_TURN,))

    session.start()
    await _settle()

    assert (
        session.reason == "the LiveKit room could not be joined: [redacted]:[redacted] was refused"
    )


def test_the_installed_sdk_can_send_media_through_an_ice_forwarder() -> None:
    assert ice_forwarder_available()


async def test_the_join_sends_its_media_through_the_caller_s_forwarder() -> None:
    room, control = FakeRoom(), FakeControl()
    room.connect_gate.release()
    session = _session(room, control, ice_forwarder=_FORWARDER)

    session.start()
    await _settle()

    (options,) = room.connect_options
    fwd = options.rtc_config.ice_forwarder
    assert fwd.addresses == ["203.0.113.10:443", "[2001:db8::1]:443"]
    assert (fwd.credentials.ufrag, fwd.credentials.pwd) == (
        "z04yKzdOjjjhPoE5bQE0/NbtKYLwqvo",
        "5H4joxyoYh0QO3jpq+iQVPqCuDWPsp4I",
    )
    assert fwd.local_port == 50001
    assert options.rtc_config.ice_servers == []
    await session.leave()


async def test_a_refused_connect_keeps_the_forwarder_password_out_of_the_reason() -> None:
    room, control = FakeRoom(), FakeControl()
    room.connect_gate.release(RuntimeError("bad pwd 5H4joxyoYh0QO3jpq+iQVPqCuDWPsp4I"))
    session = _session(room, control, ice_forwarder=_FORWARDER)

    session.start()
    await _settle()

    assert session.reason == "the LiveKit room could not be joined: bad pwd [redacted]"


def test_masking_a_secret_that_contains_another_leaves_no_part_of_it() -> None:
    assert _masked("refused turn-secret", "turn", "turn-secret") == "refused [redacted]"
    assert _masked("refused turn-secret", "turn-secret", "turn") == "refused [redacted]"


async def test_a_refused_connect_masks_a_password_that_contains_the_username() -> None:
    room, control = FakeRoom(), FakeControl()
    room.connect_gate.release(RuntimeError("credentials turn-secret were refused"))
    overlapping = IceServerEntry(
        uris=["turn:turn.example:3478"],
        credentials=TurnCredentials(username="turn", password="turn-secret"),
    )
    session = _session(room, control, ice_servers=(overlapping,))

    session.start()
    await _settle()

    assert (
        session.reason
        == "the LiveKit room could not be joined: credentials [redacted] were refused"
    )


def test_masking_leaves_text_without_a_token_as_it_is() -> None:
    assert _masked("unauthorized", "a-token") == "unauthorized"
    assert _masked("wss://rooms.example/rtc?room=a", "a-token") == "wss://rooms.example/rtc?room=a"


def test_masking_with_no_token_still_masks_an_access_token_parameter() -> None:
    assert _masked("ws://h/rtc?access_token=x.y.z", "") == "ws://h/rtc?access_token=[redacted]"


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


@pytest.fixture
def short_leave(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(room_module, "LEAVE_TIMEOUT_SECONDS", 0.05)


def _slow_leave(monkeypatch: pytest.MonkeyPatch, room: FakeRoom) -> Gate:
    """Make *room*'s leave wait on a gate the test opens."""
    leave = Gate(open_=False)

    async def disconnect() -> None:
        await leave.pass_through()
        room.disconnects += 1

    monkeypatch.setattr(room, "disconnect", disconnect)
    return leave


@pytest.mark.usefixtures("short_leave")
async def test_a_slow_leave_ends_the_close_at_the_bound_and_runs_on_uncancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room, control = FakeRoom(), FakeControl(_VIDEO_OUT)
    leave = _slow_leave(monkeypatch, room)
    session = _session(room, control)
    session.start()
    await _settle()
    assert session.state is RoomState.OPEN

    await asyncio.wait_for(session.leave(), timeout=1.0)

    assert session.state is RoomState.CLOSED
    assert control.closed == [ConnId(4711)]
    assert leave.entered
    assert not leave.cancelled
    assert not session.settled

    leave.release()
    await _settle()

    assert not leave.cancelled
    assert room.disconnects == 1
    assert session.settled


@pytest.mark.usefixtures("short_leave")
async def test_a_leave_that_fails_settles_the_room_connection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    room, control = FakeRoom(), FakeControl(_VIDEO_OUT)
    leave = _slow_leave(monkeypatch, room)
    session = _session(room, control)
    session.start()
    await _settle()

    with caplog.at_level(logging.INFO, logger="reactor_runtime.transport.livekit.room"):
        await asyncio.wait_for(session.leave(), timeout=1.0)
        leave.release(RuntimeError("the room is gone, token=a-token"))
        await _settle()

    assert session.settled
    (failed,) = [r for r in caplog.records if r.getMessage() == "leaving the LiveKit room failed"]
    fields = failed.__dict__["reactor_fields"]
    assert "a-token" not in fields["error"]
    assert fields["error_type"] == "RuntimeError"
    assert fields["leave_s"] >= 0


@pytest.mark.usefixtures("short_deadline", "short_leave")
async def test_the_session_reopens_a_room_only_once_a_slow_leave_has_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = Gate(open_=False)
    first = FakeRoom(connect=connect)
    leave = _slow_leave(monkeypatch, first)
    rooms = iter([first, FakeRoom()])
    monkeypatch.setattr(room_module.rtc, "Room", lambda: next(rooms))
    app = FastAPI()
    LiveKitRouter().mount(app, cast(Any, RouterRunner()))
    path = "/sessions/s1/transport/livekit/connections"
    body = {"url": "wss://rooms.example", "token": "a-token"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as client:
        assert (await client.post(path, json=body)).status_code == 202
        await asyncio.sleep(0.1)  # past the deadline: the join failed, its connect still runs
        connect.release()
        await asyncio.sleep(0.1)  # past the leave bound: the leave still runs

        while_leaving = await client.post(path, json=body)
        leave.release()
        await _settle()
        after = await client.post(path, json=body)

    assert leave.entered
    assert not leave.cancelled
    assert while_leaving.status_code == 409
    assert first.disconnects == 1
    assert after.status_code == 202


async def test_a_leave_that_fails_within_the_bound_is_not_logged_as_left(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    room, control = FakeRoom(), FakeControl(_VIDEO_OUT)

    async def disconnect() -> None:
        raise RuntimeError("refused")

    monkeypatch.setattr(room, "disconnect", disconnect)
    session = _session(room, control)
    session.start()
    await _settle()
    with caplog.at_level(logging.INFO, logger="reactor_runtime.transport.livekit.room"):
        await session.leave()
        await _settle()

    messages = [r.getMessage() for r in caplog.records]
    assert "leaving the LiveKit room failed" in messages
    assert "left the LiveKit room" not in messages
    assert session.settled


@pytest.mark.usefixtures("short_leave")
async def test_a_slow_leave_logs_its_late_end_with_how_long_it_took(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    room, control = FakeRoom(), FakeControl(_VIDEO_OUT)
    leave = _slow_leave(monkeypatch, room)
    session = _session(room, control)
    session.start()
    await _settle()
    with caplog.at_level(logging.INFO, logger="reactor_runtime.transport.livekit.room"):
        await asyncio.wait_for(session.leave(), timeout=1.0)
        assert session.leave_age() is not None
        leave.release()
        await _settle()

    (late,) = [r for r in caplog.records if r.getMessage() == "left the LiveKit room late"]
    assert late.__dict__["reactor_fields"]["leave_s"] >= room_module.LEAVE_TIMEOUT_SECONDS
    assert session.leave_age() is None


@pytest.mark.usefixtures("short_leave")
async def test_settle_waits_for_a_slow_leave_and_cancels_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room, control = FakeRoom(), FakeControl(_VIDEO_OUT)
    leave = _slow_leave(monkeypatch, room)
    session = _session(room, control)
    session.start()
    await _settle()
    await asyncio.wait_for(session.leave(), timeout=1.0)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(session.settle(), timeout=0.1)
    assert not leave.cancelled
    settling = asyncio.create_task(session.settle())
    leave.release()
    await asyncio.wait_for(settling, timeout=1.0)

    assert session.settled
    assert room.disconnects == 1


@pytest.mark.usefixtures("short_leave")
async def test_a_room_still_leaving_when_the_next_session_opens_is_kept_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeRoom()
    leave = _slow_leave(monkeypatch, first)
    rooms = iter([first, FakeRoom()])
    monkeypatch.setattr(room_module.rtc, "Room", lambda: next(rooms))
    monkeypatch.setattr(room_module, "LEAVE_STUCK_SECONDS", 0.1)
    runner = RouterRunner()
    router = LiveKitRouter()
    app = FastAPI()
    router.mount(app, cast(Any, runner))
    path = "/sessions/s1/transport/livekit/connections"
    body = {"url": "wss://rooms.example", "token": "a-token"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as client:
        assert (await client.post(path, json=body)).status_code == 202
        await _settle()
        runner.epoch = 2  # the first session ended; the next one opens its room
        opened = await client.post(path, json=body)
        assert opened.status_code == 202
        await _settle()

        assert router.health().status is HealthStatus.HEALTHY
        await asyncio.sleep(0.2)  # past the bound and the stuck limit: the first leave still runs
        health = router.health()
        assert health.status is HealthStatus.UNHEALTHY
        # the next session's room joined after the first began to leave: the server answers us
        assert "1 LiveKit room connection(s) stuck leaving" in str(health.detail)

        settling = asyncio.create_task(router.settle())
        await asyncio.sleep(0.05)
        assert not settling.done()
        leave.release()
        await asyncio.wait_for(settling, timeout=1.0)
        cid = opened.json()["connection_id"]
        assert (await client.delete(f"{path}/{cid}")).status_code == 204

    assert not leave.cancelled
    assert first.disconnects == 1
    assert router.health().status is HealthStatus.HEALTHY


@pytest.mark.usefixtures("short_leave")
async def test_a_stuck_leave_is_unhealthy_only_once_the_server_answers_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = FakeRoom(), FakeRoom()
    leave = _slow_leave(monkeypatch, first)
    queue = iter([first, second])
    monkeypatch.setattr(room_module.rtc, "Room", lambda: next(queue))
    monkeypatch.setattr(room_module, "LEAVE_STUCK_SECONDS", 0.1)
    runner = RouterRunner()
    router = LiveKitRouter()
    app = FastAPI()
    router.mount(app, cast(Any, runner))
    path = "/sessions/s1/transport/livekit/connections"
    body = {"url": "wss://rooms.example", "token": "a-token"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as client:
        opened = await client.post(path, json=body)
        await _settle()
        assert (await client.delete(f"{path}/{opened.json()['connection_id']}")).status_code == 204
        await asyncio.sleep(0.2)
        # stuck, and the server has answered nothing since: an outage, not this process
        assert router.health().status is HealthStatus.HEALTHY

        runner.epoch = 2
        last = await client.post(path, json=body)
        await _settle()
        # the server answers the next join while the first room connection is still stuck: a leak
        health = router.health()
        assert health.status is HealthStatus.UNHEALTHY
        assert "while the server answered others" in str(health.detail)

        leave.release()
        await asyncio.wait_for(router.settle(), timeout=1.0)
        assert router.health().status is HealthStatus.HEALTHY
        assert (await client.delete(f"{path}/{last.json()['connection_id']}")).status_code == 204


@pytest.mark.usefixtures("short_deadline")
async def test_settle_waits_for_a_timed_out_join_and_the_leave_after_it() -> None:
    connect = Gate(open_=False)
    room, control = FakeRoom(connect=connect), FakeControl(_VIDEO_OUT)
    session = _session(room, control)
    session.start()
    await asyncio.sleep(0.1)  # past the deadline: the join failed, its connect still runs
    assert not session.settled
    assert session.leave_age() is not None  # the abandoned attempt holds the room

    settling = asyncio.create_task(session.settle())
    await asyncio.sleep(0.05)
    assert not settling.done()
    connect.release()
    await asyncio.wait_for(settling, timeout=1.0)

    assert session.settled
    assert room.disconnects == 1
    assert session.leave_age() is None


@pytest.mark.usefixtures("short_leave")
async def test_a_leave_whose_caller_gave_up_still_logs_its_late_end(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    room, control = FakeRoom(), FakeControl(_VIDEO_OUT)
    leave = _slow_leave(monkeypatch, room)
    session = _session(room, control)
    session.start()
    await _settle()
    with caplog.at_level(logging.INFO, logger="reactor_runtime.transport.livekit.room"):
        leaving = asyncio.create_task(session.leave())
        await asyncio.sleep(0.01)
        leaving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leaving
        leave.release()
        await _settle()

    assert not leave.cancelled
    assert "left the LiveKit room late" in [r.getMessage() for r in caplog.records]


@pytest.mark.usefixtures("short_leave")
async def test_a_settle_cut_off_at_shutdown_names_each_room_still_leaving(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    room = FakeRoom()
    leave = _slow_leave(monkeypatch, room)
    monkeypatch.setattr(room_module.rtc, "Room", lambda: room)
    router = LiveKitRouter()
    app = FastAPI()
    router.mount(app, cast(Any, RouterRunner()))
    path = "/sessions/s1/transport/livekit/connections"
    body = {"url": "wss://rooms.example", "token": "a-token"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as client:
        opened = await client.post(path, json=body)
        await _settle()
        assert (await client.delete(f"{path}/{opened.json()['connection_id']}")).status_code == 204

    with (
        caplog.at_level(logging.WARNING, logger="reactor_runtime.transport.livekit.router"),
        pytest.raises(TimeoutError),
    ):
        await asyncio.wait_for(router.settle(), timeout=0.05)

    (record,) = [
        r
        for r in caplog.records
        if r.getMessage() == "a LiveKit room connection was still leaving at shutdown"
    ]
    assert record.__dict__["reactor_fields"]["leaving_s"] >= 0
    assert not leave.cancelled
    leave.release()
    await _settle()


async def test_settle_returns_at_once_for_a_room_connection_still_open() -> None:
    room, control = FakeRoom(), FakeControl(_VIDEO_OUT)
    session = _session(room, control)
    session.start()
    await _settle()
    assert session.state is RoomState.OPEN

    await asyncio.wait_for(session.settle(), timeout=0.5)

    await session.leave()
