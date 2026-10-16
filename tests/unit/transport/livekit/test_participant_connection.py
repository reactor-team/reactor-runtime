import asyncio
import json
from typing import Any, cast

import pytest
from livekit import rtc

from reactor_runtime.core import Connection, ConnectionOrigin, ConnId, MessageObserver
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.transport.livekit.participant import (
    CONTROL_TOPIC,
    DATA_TOPIC,
    MAX_PACKET_BYTES,
    ParticipantConnection,
)


class FakeLocalParticipant:
    """Records what the session's participant sends, in the order it leaves."""

    def __init__(self, *, stream_delay: float = 0.0) -> None:
        self.sent: list[tuple[str, str, bytes, list[str]]] = []
        self.stream_delay = stream_delay
        self.calls: list[tuple[str, str, str]] = []
        self.rpc_errors: list[rtc.RpcError] = []

    async def perform_rpc(self, *, destination_identity: str, method: str, payload: str) -> str:
        self.calls.append((destination_identity, method, payload))
        if self.rpc_errors:
            raise self.rpc_errors.pop(0)
        return ""

    async def publish_data(
        self,
        payload: bytes,
        *,
        reliable: bool = True,
        destination_identities: list[str],
        topic: str,
    ) -> None:
        assert reliable
        self.sent.append(("packet", topic, payload, destination_identities))

    async def send_bytes(
        self, data: bytes, *, topic: str, destination_identities: list[str]
    ) -> None:
        await asyncio.sleep(self.stream_delay)
        self.sent.append(("stream", topic, data, destination_identities))


OUTBOUND = {"playback_finished": ["lk.playback_finished"]}


def make(
    local: FakeLocalParticipant | None = None,
) -> tuple[ParticipantConnection, FakeLocalParticipant]:
    local = local or FakeLocalParticipant()
    conn = ParticipantConnection(
        ConnId(5120), "viewer", cast(Any, local), asyncio.get_running_loop(), OUTBOUND
    )
    return conn, local


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


# --- RPC: answers to calls the participant made, and calls it receives ------


async def test_an_rpc_call_is_answered_with_its_reply_messages_fields() -> None:
    conn, _ = make()
    assert isinstance(conn, MessageObserver)
    reply = conn.expect_reply("rpc-1")

    await asyncio.to_thread(conn.message_sent, "dropped", {"seconds": 1.4}, "rpc-1")

    assert await asyncio.wait_for(reply, 1) == '{"seconds": 1.4}'
    conn.drop()


async def test_an_rpc_call_whose_command_returned_nothing_is_answered_empty() -> None:
    conn, _ = make()
    reply = conn.expect_reply("rpc-1")

    conn.command_acknowledged("rpc-1")

    assert await asyncio.wait_for(reply, 1) == ""
    conn.drop()


async def test_an_rpc_call_whose_command_failed_fails_with_the_code() -> None:
    conn, _ = make()
    reply = conn.expect_reply("rpc-1")

    conn.command_failed("rpc-1", "not_now", "Not now.")

    with pytest.raises(rtc.RpcError) as failed:
        await asyncio.wait_for(reply, 1)
    assert failed.value.code == rtc.RpcError.ErrorCode.APPLICATION_ERROR
    assert (failed.value.message, failed.value.data) == ("Not now.", "not_now")
    conn.drop()


async def test_an_answer_for_no_waiting_call_is_ignored() -> None:
    conn, local = make()

    conn.command_acknowledged("protocol-req")
    conn.message_sent("echoed", {"text": "hi"}, "protocol-req")
    await _settle()

    assert local.calls == []
    conn.drop()


async def test_a_mapped_message_calls_its_method_with_its_fields() -> None:
    conn, local = make()

    await asyncio.to_thread(
        conn.message_sent,
        "playback_finished",
        {"playback_position": 3.2, "interrupted": False},
        None,
    )
    await _settle()

    assert local.calls == [
        ("viewer", "lk.playback_finished", '{"playback_position": 3.2, "interrupted": false}')
    ]
    conn.drop()


async def test_a_message_that_is_not_mapped_calls_nothing() -> None:
    conn, local = make()

    conn.message_sent("echoed", {"text": "hi"}, None)
    await _settle()

    assert local.calls == []
    conn.drop()


async def test_a_participant_without_the_method_is_not_called_again() -> None:
    conn, local = make()
    local.rpc_errors.append(
        rtc.RpcError(rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD, "no such method")
    )

    conn.message_sent("playback_finished", {}, None)
    await _settle()

    assert len(local.calls) == 1
    conn.drop()


async def test_a_call_that_may_not_have_arrived_is_tried_again() -> None:
    conn, local = make()
    local.rpc_errors.append(rtc.RpcError(rtc.RpcError.ErrorCode.CONNECTION_TIMEOUT, "timed out"))

    conn.message_sent("playback_finished", {}, None)
    await asyncio.sleep(0.5)

    assert len(local.calls) == 2
    conn.drop()


async def test_calls_go_out_in_the_order_the_messages_were_sent() -> None:
    conn, local = make()

    for position in range(5):
        conn.message_sent("playback_finished", {"playback_position": position}, None)
    await _settle(20)

    assert [json.loads(payload)["playback_position"] for _, _, payload in local.calls] == list(
        range(5)
    )
    conn.drop()


async def test_a_waiting_call_fails_when_the_participant_leaves() -> None:
    conn, _ = make()
    reply = conn.expect_reply("rpc-1")

    conn.drop()

    with pytest.raises(rtc.RpcError):
        await asyncio.wait_for(reply, 1)
