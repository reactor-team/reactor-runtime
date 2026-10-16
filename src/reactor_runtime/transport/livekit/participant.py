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
"""

from __future__ import annotations

import asyncio
import logging

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


class ParticipantConnection:
    """One remote participant of the session's room, as a connection of the session."""

    def __init__(
        self,
        conn_id: ConnId,
        identity: str,
        participant: rtc.LocalParticipant,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Bind the connection to the participant it speaks to.

        Args:
            conn_id: The id the session gave this connection.
            identity: The participant's identity in the room.
            participant: The session's own participant, which sends to it.
            loop: The loop the room runs on.
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
        self._closed = False
        self._refused_warned = False

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

    async def close(self) -> None:
        """Send what is queued, then stop. Idempotent."""
        if self._closed:
            return
        self._closed = True
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
