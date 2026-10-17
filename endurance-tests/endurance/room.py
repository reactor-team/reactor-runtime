"""A room participant for the room churn, speaking to the session three ways.

The participant joins the session's LiveKit room, as every client of a room
session does, and runs one command through each path a room offers: the
Reactor protocol on its data topics, an RPC method, and a data topic. It
closes once each is answered, or has had the time to be.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from google.protobuf import struct_pb2
from google.protobuf.message import DecodeError
from livekit import api, rtc

from reactor_wire.v1 import common_pb2, control_pb2, data_pb2, model_pb2, platform_pb2

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
"""The LiveKit server the room churn runs against; without it, the scenario is skipped."""

_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")

DATA_TOPIC = "reactor.data"
CONTROL_TOPIC = "reactor.control"

RPC_METHOD = "endurance.set_intensity"
"""The RPC method the endurance model maps to its ``set_intensity`` command."""

COMMAND_TOPIC = "endurance.set_effect"
"""A data topic that runs the endurance model's ``set_effect`` command."""

_REQUEST = common_pb2.MessageKind.MESSAGE_KIND_REQUEST
_RESPONSE = common_pb2.MessageKind.MESSAGE_KIND_RESPONSE
# How long the participant waits to be a client of the session, and for each answer.
_JOIN_WAIT_S = 10.0
_REPLY_WAIT_S = 10.0
# How long it listens after the topic command, for a reply it must not get.
_QUIET_S = 0.5


def token(room: str, identity: str) -> str:
    """Return a token for *identity* in *room*, signed for the dev LiveKit server."""
    grants = api.VideoGrants(
        room_join=True, room=room, can_publish=True, can_subscribe=True, can_publish_data=True
    )
    return (
        api.AccessToken(_API_KEY, _API_SECRET).with_identity(identity).with_grants(grants).to_jwt()
    )


class RoomParticipant:
    """One participant of the session's room, for one churn cycle.

    :attr:`unanswered` lists what went wrong in :meth:`run`: a request with
    no reply or an error reply, a failed RPC call, or a reply to the topic
    command, which must answer nothing.
    """

    def __init__(self, room: str, identity: str, model: str) -> None:
        """Create the participant *identity* in *room*, speaking to the session as *model*."""
        self._room_name = room
        self._identity = identity
        self._model = model
        self._room = rtc.Room()
        # Each reply by request id, as (topic, payload name).
        self._replies: dict[str, tuple[str, str | None]] = {}
        self._unexpected: list[str] = []
        self._readers: set[asyncio.Task[None]] = set()
        self.unanswered: list[str] = []
        self._room.on("data_received", self._on_data)
        for topic in (DATA_TOPIC, CONTROL_TOPIC):
            self._room.register_byte_stream_handler(topic, self._on_stream)

    async def run(self) -> None:
        """Join, run a command each way, and leave."""
        await self._room.connect(LIVEKIT_URL, token(self._room_name, self._identity))
        try:
            if not await self._become_client():
                self.unanswered.append("the session never answered the schema request")
                return
            await self._protocol_command()
            await self._rpc_command()
            await self._topic_command()
        finally:
            for reader in self._readers:
                reader.cancel()
            await self._room.disconnect()
        self.unanswered.extend(self._unexpected)

    async def _become_client(self) -> bool:
        """Ask for the schema until it is answered: then the session serves this participant."""
        deadline = time.monotonic() + _JOIN_WAIT_S
        attempt = 0
        while time.monotonic() < deadline:
            request_id = f"schema-{attempt}"
            attempt += 1
            message = control_pb2.ControlClientMessage(
                request_id=request_id, kind=_REQUEST, request_schema=platform_pb2.RequestSchema()
            )
            await self._send(CONTROL_TOPIC, message.SerializeToString())
            if await self._reply(request_id, within=0.5) is not None:
                return True
        return False

    async def _protocol_command(self) -> None:
        args = struct_pb2.Struct()
        args.update({"effect": "invert"})
        message = data_pb2.DataClientMessage(
            request_id="command",
            kind=_REQUEST,
            command=model_pb2.Command(type="set_effect", data=args),
        )
        await self._send(DATA_TOPIC, message.SerializeToString())
        reply = await self._reply("command", within=_REPLY_WAIT_S)
        if reply is None:
            self.unanswered.append("set_effect (protocol): no reply")
        elif reply[1] == "error":
            self.unanswered.append("set_effect (protocol): an error reply")

    async def _rpc_command(self) -> None:
        try:
            await self._room.local_participant.perform_rpc(
                destination_identity=self._model,
                method=RPC_METHOD,
                payload=json.dumps({"intensity": 0.5}),
                response_timeout=_REPLY_WAIT_S,
            )
        except rtc.RpcError as error:
            self.unanswered.append(f"set_intensity (rpc): {error.code} {error.message}")

    async def _topic_command(self) -> None:
        await self._room.local_participant.publish_data(
            json.dumps({"effect": "none"}).encode(),
            reliable=True,
            destination_identities=[self._model],
            topic=COMMAND_TOPIC,
        )
        await asyncio.sleep(_QUIET_S)

    async def _send(self, topic: str, payload: bytes) -> None:
        await self._room.local_participant.publish_data(
            payload, reliable=True, destination_identities=[self._model], topic=topic
        )

    async def _reply(self, request_id: str, *, within: float) -> tuple[str, str | None] | None:
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            if request_id in self._replies:
                return self._replies[request_id]
            await asyncio.sleep(0.02)
        return None

    def _on_data(self, packet: rtc.DataPacket) -> None:
        if packet.topic in (DATA_TOPIC, CONTROL_TOPIC):
            self._take(packet.topic, bytes(packet.data))

    def _on_stream(self, reader: rtc.ByteStreamReader, _sender: str) -> None:
        task = asyncio.get_running_loop().create_task(self._read(reader))
        self._readers.add(task)
        task.add_done_callback(self._readers.discard)

    async def _read(self, reader: rtc.ByteStreamReader) -> None:
        data = bytearray()
        async for chunk in reader:
            data.extend(chunk)
        self._take(reader.info.topic, bytes(data))

    def _take(self, topic: str, data: bytes) -> None:
        message: Any = (
            control_pb2.ControlServerMessage()
            if topic == CONTROL_TOPIC
            else data_pb2.DataServerMessage()
        )
        try:
            message.ParseFromString(data)
        except DecodeError:
            self._unexpected.append(f"an undecodable message on {topic}")
            return
        if topic == DATA_TOPIC and message.kind != _RESPONSE:
            return
        request_id = message.request_id
        if not request_id:
            return
        if request_id.startswith("schema-") or request_id == "command":
            self._replies[request_id] = (topic, message.WhichOneof("payload"))
        else:
            # The RPC and topic commands answer off the protocol topics.
            self._unexpected.append(f"a reply on {topic} to a request it never sent: {request_id}")
