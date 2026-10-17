import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tests.unit.transport.livekit.test_participant_connection import FakeReader
from tests.unit.transport.livekit.test_room_join import FakeControl, FakeRoom

from reactor_runtime.core import ConnId
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport import ConnectionsExhaustedError
from reactor_runtime.transport.livekit.participant import CONTROL_TOPIC, DATA_TOPIC
from reactor_runtime.transport.livekit.room import RoomSession, RoomState


class ParticipantControl(FakeControl):
    """A session that mints participant ids and records the messages it receives."""

    def __init__(self, *, ids: int = 100, refuse: set[ConnId] | None = None) -> None:
        super().__init__()
        self._ids = iter(range(5000, 5000 + ids))
        self._refuse = refuse or set()
        self.received: list[tuple[ConnId, bytes, Channel]] = []

    def new_conn_id(self, transport: str) -> ConnId:
        assert transport == "livekit"
        try:
            return ConnId(next(self._ids))
        except StopIteration:
            raise ConnectionsExhaustedError from None

    def connection_opened(self, conn: Any) -> bool:
        self.opened.append(conn)
        return conn.id not in self._refuse

    def message_received(
        self, conn_id: ConnId, payload: bytes, version: ProtocolVersion, channel: Channel
    ) -> None:
        assert version is ProtocolVersion.V1
        self.received.append((conn_id, payload, channel))


def _person(identity: str) -> SimpleNamespace:
    return SimpleNamespace(identity=identity)


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


_sessions: list[RoomSession] = []


@pytest.fixture(autouse=True)
async def _leave_every_room() -> AsyncIterator[None]:
    """Leave every room a test opened, so no room pacer outlives its test."""
    yield
    while _sessions:
        await _sessions.pop().leave()


def _session(room: FakeRoom, control: ParticipantControl) -> RoomSession:
    session = RoomSession(
        ConnId(4711), "wss://rooms.example", "a-token", cast(Any, control), room=cast(Any, room)
    )
    _sessions.append(session)
    return session


async def _open(room: FakeRoom, control: ParticipantControl) -> RoomSession:
    session = _session(room, control)
    session.start()
    await _settle()
    assert session.state is RoomState.OPEN
    return session


def _join(room: FakeRoom, identity: str) -> None:
    room.handlers["participant_connected"](_person(identity))


def _leave(room: FakeRoom, identity: str) -> None:
    room.handlers["participant_disconnected"](_person(identity))


def _packet(room: FakeRoom, identity: str, data: bytes, topic: str = DATA_TOPIC) -> None:
    room.handlers["data_received"](
        SimpleNamespace(topic=topic, participant=_person(identity), data=data)
    )


def _ids(control: ParticipantControl) -> list[ConnId]:
    return [conn.id for conn in control.opened]


async def test_participants_in_the_room_when_it_opens_become_clients() -> None:
    room, control = FakeRoom(), ParticipantControl()
    room.remote_participants = {"alice": _person("alice"), "bob": _person("bob")}

    await _open(room, control)

    assert _ids(control) == [4711, 5000, 5001]
    assert [conn.identity for conn in control.opened[1:]] == ["alice", "bob"]


async def test_a_participant_is_no_client_before_the_room_connection_opens() -> None:
    room, control = FakeRoom(), ParticipantControl()
    room.connect_gate.release(RuntimeError("unauthorized"))
    session = _session(room, control)
    session.start()
    _join(room, "early")
    await _settle()

    _join(room, "late")

    assert session.state is RoomState.FAILED
    assert control.opened == []


async def test_a_refused_room_connection_makes_no_participant_a_client() -> None:
    room, control = FakeRoom(), ParticipantControl(refuse={ConnId(4711)})
    room.remote_participants = {"alice": _person("alice")}
    session = _session(room, control)

    session.start()
    await _settle()
    _join(room, "bob")

    assert _ids(control) == [4711]


async def test_a_participant_the_session_refuses_gets_no_messages_through() -> None:
    room, control = FakeRoom(), ParticipantControl(refuse={ConnId(5000)})
    await _open(room, control)

    _join(room, "alice")
    _packet(room, "alice", b"command")

    assert _ids(control) == [4711, 5000]
    assert control.received == []


async def test_a_participant_past_the_connection_ids_is_left_out() -> None:
    room, control = FakeRoom(), ParticipantControl(ids=1)
    await _open(room, control)

    _join(room, "alice")
    _join(room, "bob")
    _packet(room, "bob", b"command")

    assert [conn.identity for conn in control.opened[1:]] == ["alice"]
    assert control.received == []


async def test_messages_reach_the_session_as_their_senders_on_their_channels() -> None:
    room, control = FakeRoom(), ParticipantControl()
    await _open(room, control)
    _join(room, "alice")
    _join(room, "bob")

    _packet(room, "bob", b"from-bob")
    _packet(room, "alice", b"from-alice", CONTROL_TOPIC)
    _packet(room, "stranger", b"from-nobody")
    _packet(room, "alice", b"other-topic", "chat")

    assert control.received == [
        (ConnId(5001), b"from-bob", Channel.DATA),
        (ConnId(5000), b"from-alice", Channel.CONTROL),
    ]


async def test_a_participant_that_leaves_is_closed_and_its_messages_stop() -> None:
    room, control = FakeRoom(), ParticipantControl()
    await _open(room, control)
    _join(room, "alice")

    _leave(room, "alice")
    _packet(room, "alice", b"late")

    assert control.closed == [ConnId(5000)]
    assert control.received == []


async def test_a_stream_from_a_participant_that_rejoined_is_not_delivered_as_the_new_one() -> None:
    room, control = FakeRoom(), ParticipantControl()
    await _open(room, control)
    _join(room, "alice")
    reader = FakeReader([b"from-the-first-visit"])
    room.stream_handlers[DATA_TOPIC](reader, "alice")
    await _settle()

    _leave(room, "alice")
    _join(room, "alice")
    reader.arrived.set()
    await _settle()

    assert _ids(control)[-1] == 5001
    assert control.received == []
    assert reader.closed


async def test_a_stream_from_someone_who_is_not_a_client_is_closed_unread() -> None:
    room, control = FakeRoom(), ParticipantControl()
    await _open(room, control)
    reader = FakeReader([b"x"])

    room.stream_handlers[DATA_TOPIC](reader, "stranger")

    assert reader.closed


async def test_ending_the_room_closes_every_participant_before_leaving() -> None:
    room, control = FakeRoom(), ParticipantControl()
    session = await _open(room, control)
    _join(room, "alice")
    _join(room, "bob")

    await session.leave()

    assert control.closed == [ConnId(5000), ConnId(5001), ConnId(4711)]
    assert room.disconnects == 1
    _join(room, "carol")
    assert len(control.opened) == 3
