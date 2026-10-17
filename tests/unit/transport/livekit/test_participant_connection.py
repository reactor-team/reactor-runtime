import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from livekit import rtc

from reactor_runtime.core import Connection, ConnectionOrigin, ConnId
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport.livekit import participant as participant_module
from reactor_runtime.transport.livekit.participant import (
    CONTROL_TOPIC,
    DATA_TOPIC,
    MAX_OPEN_STREAMS,
    MAX_PACKET_BYTES,
    MAX_STREAMED_MESSAGE_BYTES,
    ParticipantConnection,
)


class FakeLocalParticipant:
    """Records what the session's participant sends, in the order it leaves."""

    def __init__(self, *, stream_delay: float = 0.0) -> None:
        self.sent: list[tuple[str, str, bytes, list[str]]] = []
        self.stream_delay = stream_delay
        self.blocked = asyncio.Event()
        self.blocked.set()

    async def publish_data(
        self,
        payload: bytes,
        *,
        reliable: bool = True,
        destination_identities: list[str],
        topic: str,
    ) -> None:
        assert reliable
        await self.blocked.wait()
        self.sent.append(("packet", topic, payload, destination_identities))

    async def send_bytes(
        self, data: bytes, *, topic: str, destination_identities: list[str]
    ) -> None:
        await asyncio.sleep(self.stream_delay)
        self.sent.append(("stream", topic, data, destination_identities))


class FakeReader:
    """A byte stream from a participant, whose chunks arrive when the test lets them."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        topic: str = DATA_TOPIC,
        size: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.info = SimpleNamespace(topic=topic, size=size)
        self.arrived = asyncio.Event()
        # Set by the test; until then, chunks after the first wait.
        self.rest_arrived = asyncio.Event()
        self.rest_arrived.set()
        self.closed = False
        self._chunks = chunks
        self._error = error

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._read()

    async def _read(self) -> AsyncIterator[bytes]:
        await self.arrived.wait()
        for index, chunk in enumerate(self._chunks):
            if index:
                await self.rest_arrived.wait()
            yield chunk
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.closed = True


class Received:
    """Records each message the connection hands on, with its channel."""

    def __init__(self) -> None:
        self.messages: list[tuple[bytes, Channel]] = []

    def __call__(self, data: bytes, channel: Channel) -> None:
        self.messages.append((data, channel))

    @property
    def data(self) -> list[bytes]:
        return [data for data, _ in self.messages]


def make(
    local: FakeLocalParticipant | None = None, received: Received | None = None
) -> tuple[ParticipantConnection, FakeLocalParticipant]:
    local = local or FakeLocalParticipant()
    conn = ParticipantConnection(
        ConnId(5120),
        "viewer",
        cast(Any, local),
        asyncio.get_running_loop(),
        received or Received(),
    )
    return conn, local


def stream(conn: ParticipantConnection, reader: FakeReader, channel: Channel) -> None:
    conn.receive_stream(cast(Any, reader), channel)


async def _settle(times: int = 10) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def test_a_participant_is_a_connection_that_carries_messages_only() -> None:
    conn, _ = make()

    assert isinstance(conn, Connection)
    assert conn.protocol_version is ProtocolVersion.V1
    assert not conn.capabilities.carries_video
    assert not conn.capabilities.carries_audio
    assert conn.origin == ConnectionOrigin(transport="livekit", identity="viewer")
    conn.drop()


async def test_data_and_control_messages_go_to_that_participant_on_their_topics() -> None:
    conn, local = make()

    conn.send_message(b"data-frame")
    conn.send_control(b"control-frame")
    await _settle()

    assert local.sent == [
        ("packet", DATA_TOPIC, b"data-frame", ["viewer"]),
        ("packet", CONTROL_TOPIC, b"control-frame", ["viewer"]),
    ]
    conn.drop()


async def test_a_message_above_the_packet_limit_goes_as_a_byte_stream() -> None:
    conn, local = make()
    big = b"x" * (MAX_PACKET_BYTES + 1)

    conn.send_message(b"x" * MAX_PACKET_BYTES)
    conn.send_message(big)
    await _settle()

    assert [(kind, len(data)) for kind, _, data, _ in local.sent] == [
        ("packet", MAX_PACKET_BYTES),
        ("stream", MAX_PACKET_BYTES + 1),
    ]
    conn.drop()


async def test_messages_leave_in_the_order_they_were_sent() -> None:
    # A byte stream takes longer to send than a packet; the next message waits for it.
    conn, local = make(FakeLocalParticipant(stream_delay=0.05))

    conn.send_message(b"1")
    conn.send_message(b"2" * (MAX_PACKET_BYTES + 1))
    conn.send_message(b"3")
    await asyncio.sleep(0.2)

    assert [data[:1] for _, _, data, _ in local.sent] == [b"1", b"2", b"3"]
    conn.drop()


async def test_messages_sent_from_another_thread_reach_the_participant() -> None:
    conn, local = make()

    await asyncio.to_thread(conn.send_message, b"from-the-model-thread")
    await _settle()

    assert local.sent == [("packet", DATA_TOPIC, b"from-the-model-thread", ["viewer"])]
    conn.drop()


async def test_closing_sends_what_is_queued_first() -> None:
    conn, local = make(FakeLocalParticipant(stream_delay=0.05))
    conn.send_message(b"a" * (MAX_PACKET_BYTES + 1))
    conn.send_control(b"session-ended")

    await conn.close()

    assert [data[:1] for _, _, data, _ in local.sent] == [b"a", b"s"]


async def test_nothing_is_sent_after_the_connection_closes() -> None:
    conn, local = make()
    await conn.close()

    conn.send_message(b"late")
    await _settle()

    assert local.sent == []


async def test_a_participant_that_left_is_dropped_without_draining() -> None:
    conn, local = make(FakeLocalParticipant(stream_delay=1.0))
    conn.send_message(b"a" * (MAX_PACKET_BYTES + 1))
    await _settle()

    conn.drop()
    await _settle()

    assert local.sent == []


async def test_a_packet_waits_for_a_byte_stream_that_arrived_before_it() -> None:
    received = Received()
    conn, _ = make(received=received)
    reader = FakeReader([b"big-", b"command"])

    stream(conn, reader, Channel.DATA)
    conn.receive_packet(b"small", Channel.DATA)
    await _settle()
    assert received.data == []

    reader.arrived.set()
    await _settle()

    assert received.messages == [(b"big-command", Channel.DATA), (b"small", Channel.DATA)]
    assert reader.closed
    conn.drop()


async def test_a_stream_on_one_topic_does_not_hold_back_the_other() -> None:
    received = Received()
    conn, _ = make(received=received)

    stream(conn, FakeReader([b"pending"]), Channel.DATA)
    conn.receive_packet(b"control", Channel.CONTROL)
    await _settle()

    assert received.messages == [(b"control", Channel.CONTROL)]
    conn.drop()


async def test_a_stream_past_the_open_limit_is_closed_unread() -> None:
    received = Received()
    conn, _ = make(received=received)
    readers = [FakeReader([bytes([i])]) for i in range(MAX_OPEN_STREAMS + 1)]

    for reader in readers:
        stream(conn, reader, Channel.DATA)
    await _settle()

    assert readers[-1].closed
    assert not any(reader.closed for reader in readers[:-1])
    for reader in readers:
        reader.arrived.set()
    await _settle()

    assert received.data == [bytes([i]) for i in range(MAX_OPEN_STREAMS)]
    assert all(reader.closed for reader in readers)
    conn.drop()


async def test_a_stream_that_declares_an_oversized_message_is_closed_unread() -> None:
    received = Received()
    conn, _ = make(received=received)
    reader = FakeReader([b"x"], size=MAX_STREAMED_MESSAGE_BYTES + 1)

    stream(conn, reader, Channel.DATA)
    conn.receive_packet(b"next", Channel.DATA)
    await _settle()

    assert reader.closed
    assert received.data == [b"next"]
    conn.drop()


async def test_a_stream_that_grows_past_the_message_limit_is_dropped_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(participant_module, "MAX_STREAMED_MESSAGE_BYTES", 8)
    received = Received()
    conn, _ = make(received=received)
    reader = FakeReader([b"12345", b"67890", b"never-read"])

    stream(conn, reader, Channel.DATA)
    conn.receive_packet(b"next", Channel.DATA)
    reader.arrived.set()
    await _settle()

    assert reader.closed
    assert received.data == [b"next"]
    assert conn._inbound_bytes == 0
    conn.drop()


async def test_messages_waiting_behind_a_stream_are_held_to_the_inbound_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(participant_module, "MAX_INBOUND_BYTES", 10)
    received = Received()
    conn, _ = make(received=received)
    blocker = FakeReader([b"held"])

    stream(conn, blocker, Channel.DATA)
    conn.receive_packet(b"waits", Channel.DATA)
    conn.receive_packet(b"over-budget", Channel.DATA)
    blocker.arrived.set()
    await _settle()

    assert received.data == [b"held", b"waits"]
    assert conn._inbound_bytes == 0
    conn.drop()


async def test_a_stream_that_would_overrun_the_inbound_budget_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(participant_module, "MAX_INBOUND_BYTES", 10)
    received = Received()
    conn, _ = make(received=received)
    first, second = FakeReader([b"123456", b"7"]), FakeReader([b"abcdef"])
    first.rest_arrived.clear()

    stream(conn, first, Channel.DATA)
    stream(conn, second, Channel.DATA)
    first.arrived.set()
    await _settle()
    # The first holds six bytes while it waits for the rest; the second does not fit beside it.
    second.arrived.set()
    await _settle()
    first.rest_arrived.set()
    await _settle()

    assert received.data == [b"1234567"]
    assert first.closed
    assert second.closed
    assert conn._inbound_bytes == 0
    conn.drop()


async def test_a_stream_that_does_not_finish_in_time_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(participant_module, "STREAM_READ_TIMEOUT_SECONDS", 0.05)
    received = Received()
    conn, _ = make(received=received)
    stalled = FakeReader([b"never"])

    stream(conn, stalled, Channel.DATA)
    conn.receive_packet(b"after", Channel.DATA)
    await asyncio.sleep(0.1)

    assert stalled.closed
    assert received.data == [b"after"]
    conn.drop()


async def test_a_stream_the_sender_aborts_is_dropped() -> None:
    received = Received()
    conn, _ = make(received=received)
    aborted = FakeReader([b"part"], error=rtc.StreamError("aborted"))

    stream(conn, aborted, Channel.DATA)
    conn.receive_packet(b"after", Channel.DATA)
    aborted.arrived.set()
    await _settle()

    assert aborted.closed
    assert received.data == [b"after"]
    conn.drop()


async def test_a_stream_still_reading_when_the_participant_leaves_is_never_delivered() -> None:
    received = Received()
    conn, _ = make(received=received)
    reader = FakeReader([b"late"])
    stream(conn, reader, Channel.DATA)
    await _settle()

    conn.drop()
    reader.arrived.set()
    await _settle()

    assert reader.closed
    assert received.data == []


async def test_messages_past_the_outbox_budget_are_dropped_while_the_sender_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(participant_module, "MAX_OUTBOX_BYTES", 10)
    local = FakeLocalParticipant()
    local.blocked.clear()
    conn, _ = make(local)

    for message in (b"aaaa", b"bbbb", b"cccc"):
        conn.send_message(message)
    await _settle()
    assert conn._outbox_bytes == 8

    local.blocked.set()
    await _settle()
    conn.send_message(b"dddd")
    await _settle()

    assert [data for _, _, data, _ in local.sent] == [b"aaaa", b"bbbb", b"dddd"]
    assert conn._outbox_bytes == 0
    conn.drop()
