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

Messages from a participant reach the session in the order they arrived on
their topic, as on a WebRTC data channel: a packet that arrives while an
earlier byte stream on the same topic is still being read waits for it. Every
stream belongs to the connection it arrived on, and is never handed to a later
connection of the same identity. What one participant can hold is bounded:
:data:`MAX_OPEN_STREAMS` streams at once, :data:`MAX_INBOUND_BYTES` of
messages being read or waiting their turn, and :data:`MAX_OUTBOX_BYTES` of
messages waiting to be sent to it. A message past a bound is dropped and
logged.

The connection also takes what it is sent as plain data
(:class:`~reactor_runtime.core.MessageObserver`). That is how a command an RPC
call ran answers the caller, and how a model message mapped to an RPC method
calls it on the participant. RPC calls go out through a queue of their own, in
order, so a slow answer never holds up the protocol's messages. At most
:data:`MAX_PENDING_CALLS` calls wait; past that, a new call is dropped and
logged. A method the participant answers is unsupported is not called on it
again.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

MAX_STREAMED_MESSAGE_BYTES = 8 * 1024 * 1024
"""The largest protocol message a participant may send as a byte stream; a larger one is dropped."""

MAX_OPEN_STREAMS = 8
"""How many byte streams one participant may have open at once; a stream past it is refused."""

MAX_INBOUND_BYTES = 2 * MAX_STREAMED_MESSAGE_BYTES
"""How many bytes of one participant's messages may be held while read or waiting their turn."""

MAX_OUTBOX_BYTES = 2 * MAX_STREAMED_MESSAGE_BYTES
"""How many bytes may wait to be sent to one participant; a message past it is dropped."""

MAX_PENDING_CALLS = 64
"""How many RPC calls may wait to be made on one participant; a call past it is dropped."""

STREAM_READ_TIMEOUT_SECONDS = 30.0
"""How long one byte stream may take to arrive before it is dropped."""

# How long a closing connection waits for its queued messages to leave, so a
# notice sent just before the session ends still reaches the participant.
_CLOSE_DRAIN_SECONDS = 2.0

# The RPC failures worth another try: the call did not reach the participant.
# A response timeout is not one: the participant acknowledged the call, so its
# handler ran, and calling again would deliver the message twice.
_TRANSIENT_RPC_ERRORS = frozenset(
    {
        rtc.RpcError.ErrorCode.CONNECTION_TIMEOUT,
        rtc.RpcError.ErrorCode.SEND_FAILED,
    }
)
_RPC_ATTEMPTS = 3
_RPC_RETRY_SECONDS = 0.25


@dataclass
class _Inbound:
    """One message from a participant, in its place in its topic's order."""

    data: bytes | None = None
    done: bool = False


class _Warning:
    """Log a recurring problem once as a warning, then quietly, until it clears."""

    def __init__(self) -> None:
        self._raised = False

    def log(self, message: str, *args: object, exc_info: bool = False) -> None:
        level = logging.DEBUG if self._raised else logging.WARNING
        self._raised = True
        logger.log(level, message, *args, exc_info=exc_info)

    def clear(self) -> None:
        self._raised = False


class ParticipantConnection:
    """One remote participant of the session's room, as a connection of the session."""

    def __init__(
        self,
        conn_id: ConnId,
        identity: str,
        participant: rtc.LocalParticipant,
        loop: asyncio.AbstractEventLoop,
        on_message: Callable[[bytes, Channel], None],
        outbound: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Bind the connection to the participant it speaks to.

        Args:
            conn_id: The id the session gave this connection.
            identity: The participant's identity in the room.
            participant: The session's own participant, which sends to it.
            loop: The loop the room runs on.
            on_message: Called on the loop with each message the participant
                sends, in order, while the connection is open.
            outbound: Each model message that calls RPC methods, and the methods.
        """
        self.id = conn_id
        self.identity = identity
        self.origin = ConnectionOrigin(transport=TRANSPORT_NAME, identity=identity)
        self.capabilities = ConnectionCapabilities()
        self._participant = participant
        self._loop = loop
        self._on_message = on_message
        # Messages to send, in order; None ends the queue.
        self._outbox: asyncio.Queue[tuple[str, bytes] | None] = asyncio.Queue()
        self._outbox_bytes = 0
        self._pump = loop.create_task(self._send_all(), name=f"room-participant-{conn_id}")
        self._outbound = dict(outbound or {})
        # RPC calls to make, in order: (method, payload); None ends the queue.
        self._calls: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        self._caller = loop.create_task(self._call_all(), name=f"room-rpc-{conn_id}")
        # The RPC calls this participant made that wait for their command's answer.
        self._replies: dict[str, asyncio.Future[str]] = {}
        self._closed = False
        # Messages received, in arrival order per topic, until their turn comes.
        self._inbound: dict[Channel, deque[_Inbound]] = {c: deque() for c in Channel}
        self._inbound_bytes = 0
        self._readers: set[asyncio.Task[None]] = set()
        self._send_failed = _Warning()
        self._outbox_full = _Warning()
        self._inbound_full = _Warning()
        self._failed_methods: set[str] = set()
        self._unsupported: set[str] = set()
        self._calls_full = _Warning()

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

    def receive_packet(self, data: bytes, channel: Channel) -> None:
        """Take a message the participant sent as a data packet, on the loop."""
        if self._closed:
            return
        if not self._hold(len(data)):
            self._log_over_budget()
            return
        self._inbound[channel].append(_Inbound(data, done=True))
        self._deliver(channel)

    def receive_stream(self, reader: rtc.ByteStreamReader, channel: Channel) -> None:
        """Take a message the participant sent as a byte stream, on the loop.

        The stream is read at once, and its message waits its turn behind
        what arrived before it on the same topic. A stream past the bounds is
        closed unread.
        """
        declared = reader.info.size or 0
        if self._closed or len(self._readers) >= MAX_OPEN_STREAMS:
            reader.close()
            if not self._closed:
                self._inbound_full.log(
                    "refused a byte stream from participant %r: %d are already open",
                    self.identity,
                    MAX_OPEN_STREAMS,
                )
            return
        if declared > MAX_STREAMED_MESSAGE_BYTES:
            reader.close()
            self._log_oversized()
            return
        slot = _Inbound()
        self._inbound[channel].append(slot)
        task = self._loop.create_task(self._read_stream(reader, slot, channel))
        self._readers.add(task)
        task.add_done_callback(self._readers.discard)

    async def close(self) -> None:
        """Send what is queued, then stop. Idempotent.

        Messages from the participant that have not reached the session yet
        are dropped.
        """
        if self._closed:
            return
        self._closed = True
        self._stop_reading()
        self._fail_replies()
        # Through the loop's callbacks, so each end follows every send and call
        # scheduled before it.
        self._loop.call_soon(self._outbox.put_nowait, None)
        self._loop.call_soon(self._calls.put_nowait, None)
        pumps = (self._pump, self._caller)
        _, late = await asyncio.wait(
            [asyncio.shield(pump) for pump in pumps], timeout=_CLOSE_DRAIN_SECONDS
        )
        if late:
            for pump in pumps:
                pump.cancel()

    def drop(self) -> None:
        """Stop at once, for a participant that has left: nothing queued can reach it."""
        self._closed = True
        self._stop_reading()
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
        self._loop.call_soon_threadsafe(self._admit_call, method, payload)

    def _admit_call(self, method: str, payload: str) -> None:
        """Queue an RPC call, on the loop, unless the method is unsupported or the queue is full."""
        if method in self._unsupported:
            return
        if self._calls.qsize() >= MAX_PENDING_CALLS:
            self._calls_full.log(
                "dropped a call to %r on participant %r: %d calls already wait",
                method,
                self.identity,
                MAX_PENDING_CALLS,
            )
            return
        self._calls.put_nowait((method, payload))

    async def _call_all(self) -> None:
        while True:
            item = await self._calls.get()
            if item is None:
                return
            await self._call(*item)
            if self._calls.qsize() < MAX_PENDING_CALLS // 2:
                self._calls_full.clear()

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
                    self._unsupported.add(method)
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

    def _stop_reading(self) -> None:
        for task in self._readers:
            task.cancel()
        for queue in self._inbound.values():
            queue.clear()
        self._inbound_bytes = 0

    def _hold(self, size: int) -> bool:
        """Count *size* more inbound bytes against the budget, if they fit."""
        if self._inbound_bytes + size > MAX_INBOUND_BYTES:
            return False
        self._inbound_bytes += size
        return True

    def _log_over_budget(self) -> None:
        self._inbound_full.log(
            "dropped a message from participant %r: its messages are over the limit",
            self.identity,
        )

    def _log_oversized(self) -> None:
        self._inbound_full.log(
            "dropped a message from participant %r larger than %d bytes",
            self.identity,
            MAX_STREAMED_MESSAGE_BYTES,
        )

    async def _read_stream(
        self, reader: rtc.ByteStreamReader, slot: _Inbound, channel: Channel
    ) -> None:
        data = bytearray()
        try:
            async with asyncio.timeout(STREAM_READ_TIMEOUT_SECONDS):
                async for chunk in reader:
                    if len(data) + len(chunk) > MAX_STREAMED_MESSAGE_BYTES:
                        self._log_oversized()
                        return
                    if not self._hold(len(chunk)):
                        self._log_over_budget()
                        return
                    data.extend(chunk)
            slot.data = bytes(data)
        except TimeoutError:
            self._inbound_full.log(
                "dropped a byte stream from participant %r that did not finish in %.0f s",
                self.identity,
                STREAM_READ_TIMEOUT_SECONDS,
            )
        except rtc.StreamError as error:
            logger.debug("a byte stream from participant %r ended early: %s", self.identity, error)
        finally:
            reader.close()
            slot.done = True
            if not self._closed:
                if slot.data is None:
                    self._inbound_bytes -= len(data)
                self._deliver(channel)

    def _deliver(self, channel: Channel) -> None:
        """Hand on the messages at the front of *channel* whose turn has come."""
        queue = self._inbound[channel]
        while queue and queue[0].done and not self._closed:
            data = queue.popleft().data
            if data is None:
                continue
            self._inbound_bytes -= len(data)
            if self._inbound_bytes < MAX_INBOUND_BYTES // 2:
                self._inbound_full.clear()
            self._on_message(data, channel)

    def _enqueue(self, topic: str, payload: bytes | str) -> None:
        if self._closed:
            return
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._admit, topic, data)

    def _admit(self, topic: str, data: bytes) -> None:
        """Queue a message to send, on the loop, if the outbox has room for it.

        A message sent before the connection closed is still admitted, so it
        leaves ahead of the end of the queue.
        """
        if self._outbox_bytes + len(data) > MAX_OUTBOX_BYTES:
            self._outbox_full.log(
                "dropped a message to participant %r: %d bytes already wait to be sent",
                self.identity,
                self._outbox_bytes,
            )
            return
        self._outbox_bytes += len(data)
        self._outbox.put_nowait((topic, data))

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
                self._send_failed.log(
                    "sending to participant %r failed (%d bytes on %s)",
                    self.identity,
                    len(data),
                    topic,
                    exc_info=True,
                )
            finally:
                self._outbox_bytes -= len(data)
                if self._outbox_bytes < MAX_OUTBOX_BYTES // 2:
                    self._outbox_full.clear()

    async def _send_one(self, topic: str, data: bytes) -> None:
        if len(data) <= MAX_PACKET_BYTES:
            await self._participant.publish_data(
                data, reliable=True, destination_identities=[self.identity], topic=topic
            )
            return
        await self._participant.send_bytes(
            data, topic=topic, destination_identities=[self.identity]
        )
