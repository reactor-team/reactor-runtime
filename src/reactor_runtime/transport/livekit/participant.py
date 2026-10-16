"""A participant of the room, as a client of the session.

Every remote participant is a connection of its own, on a connection id minted
for the ``livekit`` transport, and its journal entries name its identity in
the room. It carries messages only: the session's media reaches the room once,
through the room connection.

The Reactor protocol travels as data messages on two topics, which follow the
protocol's two message families, as the WebRTC transport's two data channels
do. A message larger than :data:`MAX_PACKET_BYTES` goes as a byte stream on
the same topic. Messages to one participant leave in the order they were sent,
through one queue, so a reply never overtakes the message sent before it.

The connection also takes what it is sent as plain data
(:class:`~reactor_runtime.core.MessageObserver`). That is how a command an RPC
call ran answers the caller, and how a model message mapped to an RPC method
calls it on the participant. RPC calls go out through a queue of their own, in
order, so a slow answer never holds up the protocol's messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from livekit import rtc

from reactor_runtime.core import ConnectionCapabilities, ConnectionOrigin, ConnId, MediaChunk
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport.livekit.version import TRANSPORT_NAME

logger = logging.getLogger(__name__)

DATA_TOPIC = "reactor.data"
"""The topic of the Reactor protocol's data messages: commands, replies, model messages."""

CONTROL_TOPIC = "reactor.control"
"""The topic of the Reactor protocol's control messages."""

TOPIC_CHANNELS = {DATA_TOPIC: Channel.DATA, CONTROL_TOPIC: Channel.CONTROL}
"""Each protocol topic and the message family it carries."""

MAX_PACKET_BYTES = 15_000
"""The largest message sent as a single data packet; anything larger goes as a byte stream.

The SDK cuts byte streams into chunks of this size. A packet must also fit the
message size each peer negotiates with the server (64,000 bytes measured
against ``livekit-server`` 1.13), so this keeps a margin for every client.
"""

# How long a closing connection waits for its queued messages to leave, so a
# notice sent just before the session ends still reaches the participant.
_CLOSE_DRAIN_SECONDS = 2.0

# The RPC failures worth another try: the call may simply not have arrived.
_TRANSIENT_RPC_ERRORS = frozenset(
    {
        rtc.RpcError.ErrorCode.CONNECTION_TIMEOUT,
        rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT,
        rtc.RpcError.ErrorCode.SEND_FAILED,
    }
)
_RPC_ATTEMPTS = 3
_RPC_RETRY_SECONDS = 0.25


class ParticipantConnection:
    """One remote participant of the session's room, as a connection of the session."""

    def __init__(
        self,
        conn_id: ConnId,
        identity: str,
        participant: rtc.LocalParticipant,
        loop: asyncio.AbstractEventLoop,
        outbound: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Bind the connection to the participant it speaks to.

        Args:
            conn_id: The id the session gave this connection.
            identity: The participant's identity in the room.
            participant: The session's own participant, which sends to it.
            loop: The loop the room runs on.
            outbound: Each model message that calls RPC methods, and the methods.
        """
        self.id = conn_id
        self.identity = identity
        self.origin = ConnectionOrigin(transport=TRANSPORT_NAME, identity=identity)
        self.capabilities = ConnectionCapabilities()
        self._participant = participant
        self._loop = loop
        # Messages to send, in order; None ends the queue.
        self._outbox: asyncio.Queue[tuple[str, bytes] | None] = asyncio.Queue()
        self._pump = loop.create_task(self._send_all(), name=f"room-participant-{conn_id}")
        self._outbound = dict(outbound or {})
        # RPC calls to make, in order: (method, payload).
        self._calls: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._caller = loop.create_task(self._call_all(), name=f"room-rpc-{conn_id}")
        # The RPC calls this participant made that wait for their command's answer.
        self._replies: dict[str, asyncio.Future[str]] = {}
        self._closed = False
        self._refused_warned = False
        self._failed_methods: set[str] = set()

    @property
    def protocol_version(self) -> ProtocolVersion:
        """Participants speak protocol version 1."""
        return ProtocolVersion.V1

    def send_message(self, payload: bytes | str) -> None:
        """Send a data message to this participant, from any thread."""
        self._enqueue(DATA_TOPIC, payload)

    def send_control(self, payload: bytes | str) -> None:
        """Send a control message to this participant, from any thread."""
        self._enqueue(CONTROL_TOPIC, payload)

    def send_media(self, chunk: MediaChunk) -> None:
        """Do nothing: the session's media reaches the room through the room connection."""

    def flush_media(self, *, hold: bool = False) -> None:
        """Do nothing: no media is queued here."""

    def set_media_rate(self, fps: float) -> None:
        """Do nothing: no media is paced here."""

    def set_media_depth(self, depth: int) -> None:
        """Do nothing: no media is queued here."""

    def resume_track(self, name: str) -> None:
        """Do nothing: the room's server manages who receives each track."""

    def pause_track(self, name: str) -> None:
        """Do nothing: the room's server manages who receives each track."""

    def expect_reply(self, request_id: str) -> asyncio.Future[str]:
        """Return the future the answer to command *request_id* resolves, for an RPC call.

        It resolves with the reply message's fields as JSON, or an empty string
        for an acknowledgement, and fails with an :class:`rtc.RpcError` that
        carries the failure's code as its data. Call on the room's loop.
        """
        future: asyncio.Future[str] = self._loop.create_future()
        self._replies[request_id] = future
        future.add_done_callback(lambda _: self._replies.pop(request_id, None))
        return future

    def message_sent(self, name: str, data: Mapping[str, Any], request_id: str | None) -> None:
        """Answer the RPC call a reply belongs to, or call the methods a message maps to."""
        payload = json.dumps(dict(data))
        if request_id is not None and request_id in self._replies:
            self._resolve(request_id, payload)
            return
        for method in self._outbound.get(name, ()):
            self._call_soon(method, payload)

    def command_acknowledged(self, request_id: str) -> None:
        """Answer the RPC call whose command returned nothing with an empty string."""
        self._resolve(request_id, "")

    def command_failed(self, request_id: str, code: str, message: str) -> None:
        """Fail the RPC call whose command failed, with the failure's code as the error's data."""
        error = rtc.RpcError(rtc.RpcError.ErrorCode.APPLICATION_ERROR, message, code)
        self._loop.call_soon_threadsafe(self._settle, request_id, None, error)

    async def close(self) -> None:
        """Send what is queued, then stop. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._caller.cancel()
        self._fail_replies()
        # Through the loop's callbacks, so the end follows every send scheduled before it.
        self._loop.call_soon(self._outbox.put_nowait, None)
        try:
            await asyncio.wait_for(asyncio.shield(self._pump), timeout=_CLOSE_DRAIN_SECONDS)
        except TimeoutError:
            self._pump.cancel()

    def drop(self) -> None:
        """Stop at once, for a participant that has left: nothing queued can reach it."""
        self._closed = True
        self._pump.cancel()
        self._caller.cancel()
        self._fail_replies()

    def _resolve(self, request_id: str, payload: str) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._settle, request_id, payload, None)

    def _settle(self, request_id: str, payload: str | None, error: Exception | None) -> None:
        future = self._replies.get(request_id)
        if future is None or future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(payload or "")

    def _fail_replies(self) -> None:
        for future in list(self._replies.values()):
            if not future.done():
                future.set_exception(
                    rtc.RpcError(rtc.RpcError.ErrorCode.APPLICATION_ERROR, "The session ended.")
                )

    def _call_soon(self, method: str, payload: str) -> None:
        if self._closed or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._calls.put_nowait, (method, payload))

    async def _call_all(self) -> None:
        while True:
            method, payload = await self._calls.get()
            await self._call(method, payload)

    async def _call(self, method: str, payload: str) -> None:
        """Call *method* on the participant, retrying a failure that may be transient."""
        for attempt in range(_RPC_ATTEMPTS):
            try:
                await self._participant.perform_rpc(
                    destination_identity=self.identity, method=method, payload=payload
                )
                return
            except rtc.RpcError as error:
                if error.code == rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD:
                    return
                if error.code in _TRANSIENT_RPC_ERRORS and attempt + 1 < _RPC_ATTEMPTS:
                    await asyncio.sleep(_RPC_RETRY_SECONDS * (attempt + 1))
                    continue
                self._call_failed(method, f"{error.code}: {error.message}")
                return
            except Exception as error:
                self._call_failed(method, str(error) or type(error).__name__)
                return

    def _call_failed(self, method: str, reason: str) -> None:
        level = logging.DEBUG if method in self._failed_methods else logging.WARNING
        self._failed_methods.add(method)
        logger.log(level, "calling %r on participant %r failed: %s", method, self.identity, reason)

    def _enqueue(self, topic: str, payload: bytes | str) -> None:
        if self._closed:
            return
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._outbox.put_nowait, (topic, data))

    async def _send_all(self) -> None:
        while True:
            item = await self._outbox.get()
            if item is None:
                return
            topic, data = item
            try:
                await self._send_one(topic, data)
            except asyncio.CancelledError:
                raise
            except Exception:
                level = logging.DEBUG if self._refused_warned else logging.WARNING
                self._refused_warned = True
                logger.log(
                    level,
                    "sending to participant %r failed (%d bytes on %s)",
                    self.identity,
                    len(data),
                    topic,
                    exc_info=True,
                )

    async def _send_one(self, topic: str, data: bytes) -> None:
        if len(data) <= MAX_PACKET_BYTES:
            await self._participant.publish_data(
                data, reliable=True, destination_identities=[self.identity], topic=topic
            )
            return
        await self._participant.send_bytes(
            data, topic=topic, destination_identities=[self.identity]
        )
