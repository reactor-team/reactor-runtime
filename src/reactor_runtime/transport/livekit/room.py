"""One room connection: the join, the published tracks, and the leave.

A :class:`RoomSession` is built for each room connection a caller opens. It
joins the room in the background, publishes one LiveKit track per output track
without waiting for a subscriber, and opens the session's
:class:`~reactor_runtime.transport.livekit.connection.RoomConnection` on the
connection id minted for it. A join that fails never opens the connection: its
failure is reported on the connection and journalled.

A room connection never ends the session. When the caller leaves the room, the
room closes, the session's participant is removed, or the connection to the
room is lost after the SDK's own reconnection attempts, the room connection
closes, and the session goes on without it. When the session ends, its
teardown closes the room connection, which leaves the room.

Every other participant is a client of the session, from the moment the room
connection is registered: one :class:`ParticipantConnection` each, on a
connection id minted for the ``livekit`` transport, opened when the participant
joins and closed when it leaves. The Reactor protocol messages a participant
sends, as data packets or byte streams on the protocol's topics, reach the
session as that participant's. When the room connection closes, every
participant connection closes before it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from livekit import rtc

from reactor_runtime.core import ConnId, TrackDirection, TrackInfo, TrackKind
from reactor_runtime.log import get_logger
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.transport.livekit.connection import (
    _INITIAL_VIDEO_SIZE,
    AUDIO_QUEUE_MS,
    RoomConnection,
)
from reactor_runtime.transport.livekit.participant import TOPIC_CHANNELS, ParticipantConnection
from reactor_runtime.transport.livekit.version import TRANSPORT_NAME
from reactor_runtime.transport.router import ConnectionsExhaustedError, SessionControl

logger = get_logger(__name__)

JOIN_TIMEOUT_SECONDS = 30.0
"""How long a join may take before it fails."""

_JOIN_REFUSED = "the LiveKit room could not be joined: {cause}"
_JOIN_TIMED_OUT = "joining the LiveKit room timed out"
_ROOM_LOST = "the LiveKit room connection closed ({reason})"
_LEFT = "the room was left on request"
_SESSION_ENDED = "the session ended"

MAX_STREAMED_MESSAGE_BYTES = 8 * 1024 * 1024
"""The largest protocol message a participant may send as a byte stream; larger ones are dropped."""


class RoomState(StrEnum):
    """Where a room connection is in its life."""

    JOINING = "joining"
    """The join runs in the background."""
    OPEN = "open"
    """The join succeeded and the room connection is a connection of the session."""
    FAILED = "failed"
    """The join failed, so the room connection never opened."""
    CLOSED = "closed"
    """The room connection opened, or was still joining, and has since closed."""


class RoomSession:
    """Join one room for one room connection, and leave it once.

    Built when a caller opens a room connection, started once, and closed
    once. Never reused. :attr:`state` and :attr:`reason` report where the
    room connection is, for the caller polling it.

    Args:
        conn_id: The connection id minted for the room connection.
        url: The LiveKit server URL.
        token: The access token for the session's participant. A credential:
            it is used to join and never logged.
        control: The runner's session-control surface.
    """

    def __init__(self, conn_id: ConnId, url: str, token: str, control: SessionControl) -> None:
        self.conn_id = conn_id
        self.state = RoomState.JOINING
        self.reason: str | None = None
        self._url = url
        self._token = token
        self._control = control
        self._room = rtc.Room()
        self._connection: RoomConnection | None = None
        self._join: asyncio.Task[None] | None = None
        self._closing: set[asyncio.Task[None]] = set()
        # The participants that are clients of the session, by identity. Empty
        # until the session registers the room connection: nobody in the room
        # is a client of a session the room is not connected to.
        self._participants: dict[str, ParticipantConnection] = {}
        self._registered = False
        self._readers: set[asyncio.Task[None]] = set()
        self._room.on("disconnected", self._on_disconnected)
        self._room.on("participant_connected", lambda p: self._add_participant(p.identity))
        self._room.on("participant_disconnected", lambda p: self._remove_participant(p.identity))
        self._room.on("data_received", self._on_data)
        for topic in TOPIC_CHANNELS:
            self._room.register_byte_stream_handler(topic, self._on_byte_stream)

    @property
    def room(self) -> rtc.Room:
        """The LiveKit room this room connection joins."""
        return self._room

    @property
    def joining(self) -> bool:
        """Whether the join is still in flight."""
        return self.state is RoomState.JOINING

    @property
    def active(self) -> bool:
        """Whether the room connection is joining or open."""
        return self.state in (RoomState.JOINING, RoomState.OPEN)

    def start(self) -> None:
        """Start joining the room in the background."""
        self._join = asyncio.get_running_loop().create_task(
            self._run_join(), name=f"livekit-join-{self.conn_id}"
        )

    async def leave(self) -> None:
        """Leave the room, closing the room connection. Idempotent.

        A join still in flight is cancelled.
        """
        await self._finish(_LEFT)

    async def _run_join(self) -> None:
        """Join the room, publish the output tracks, and open the room connection."""
        try:
            await asyncio.wait_for(
                self._room.connect(self._url, self._token), timeout=JOIN_TIMEOUT_SECONDS
            )
            connection = await self._publish(self._output_tracks())
        except TimeoutError:
            await self._fail(_JOIN_TIMED_OUT)
            return
        except Exception as error:
            await self._fail(_JOIN_REFUSED.format(cause=str(error) or type(error).__name__))
            return
        self._connection = connection
        self.state = RoomState.OPEN
        connection.start()
        if not self._control.connection_opened(connection):
            # The session it was opened for has ended; closing it leaves the room.
            return
        self._registered = True
        for participant in list(self._room.remote_participants.values()):
            self._add_participant(participant.identity)
        logger.info(
            "joined the LiveKit room",
            conn_id=self.conn_id,
            room=self._room.name,
            identity=self._room.local_participant.identity,
        )

    async def _fail(self, reason: str) -> None:
        """End a join that did not succeed, and report it on the connection."""
        self.state = RoomState.FAILED
        self.reason = reason
        await self._disconnect()
        self._control.connection_failed(self.conn_id, TRANSPORT_NAME, reason)

    async def _finish(self, reason: str) -> None:
        """Close the room connection and leave the room, once.

        The room connection is reported closed to the session before its
        media stops, so the session's fan-out stops reaching it first.
        """
        if not self.active:
            return
        joining = self.state is RoomState.JOINING
        self.state = RoomState.CLOSED
        self.reason = reason
        if joining and self._join is not None:
            self._join.cancel()
            await asyncio.gather(self._join, return_exceptions=True)
        self._registered = False
        for reader in self._readers:
            reader.cancel()
        participants = list(self._participants.values())
        self._participants.clear()
        for participant in participants:
            participant.drop()
            self._control.connection_closed(participant.id)
        connection, self._connection = self._connection, None
        if connection is not None:
            self._control.connection_closed(self.conn_id)
            await connection.close()
        await self._disconnect()
        logger.info("left the LiveKit room", conn_id=self.conn_id, reason=reason)

    async def _connection_closed(self) -> None:
        """Leave the room when the session closes the room connection itself.

        The session's teardown closes every connection. The participants'
        queued messages leave before the room does, so a notice sent just
        before the session ended still reaches them.
        """
        self._connection = None
        participants = list(self._participants.values())
        self._participants.clear()
        await asyncio.gather(*(participant.close() for participant in participants))
        await self._finish(_SESSION_ENDED)

    async def _disconnect(self) -> None:
        try:
            await self._room.disconnect()
        except Exception:
            logger.warning("leaving the LiveKit room failed", exc_info=True)

    def _output_tracks(self) -> dict[str, TrackInfo]:
        tracks: dict[str, TrackInfo] = {}
        for name, entry in self._control.track_map().items():
            if entry["direction"] != TrackDirection.OUT.value:
                continue
            tracks[name] = TrackInfo(
                name=name,
                kind=TrackKind(entry["kind"]),
                rate=float(entry["rate"]),
                direction=TrackDirection.OUT,
            )
        return tracks

    async def _publish(self, tracks: Mapping[str, TrackInfo]) -> RoomConnection:
        """Publish one LiveKit track per output track, named after it."""
        participant = self._room.local_participant
        video: dict[str, rtc.VideoSource] = {}
        audio: dict[str, rtc.AudioSource] = {}
        for name, info in tracks.items():
            local: Any
            if info.kind is TrackKind.VIDEO:
                video[name] = rtc.VideoSource(*_INITIAL_VIDEO_SIZE)
                local = rtc.LocalVideoTrack.create_video_track(name, video[name])
                source = rtc.TrackSource.SOURCE_CAMERA
            else:
                audio[name] = rtc.AudioSource(int(info.rate), 1, queue_size_ms=AUDIO_QUEUE_MS)
                local = rtc.LocalAudioTrack.create_audio_track(name, audio[name])
                source = rtc.TrackSource.SOURCE_MICROPHONE
            await participant.publish_track(local, rtc.TrackPublishOptions(source=source))
        return RoomConnection(
            self.conn_id,
            asyncio.get_running_loop(),
            video,
            audio,
            tracks,
            on_close=self._connection_closed,
        )

    def _add_participant(self, identity: str) -> None:
        """Open a connection for a participant that is in the room."""
        if not self._registered or identity in self._participants:
            return
        try:
            conn_id = self._control.new_conn_id(TRANSPORT_NAME)
        except ConnectionsExhaustedError:
            logger.warning("no connection id left for a room participant", participant=identity)
            return
        conn = ParticipantConnection(
            conn_id, identity, self._room.local_participant, asyncio.get_running_loop()
        )
        self._participants[identity] = conn
        if not self._control.connection_opened(conn):
            self._participants.pop(identity, None)

    def _remove_participant(self, identity: str) -> None:
        """Close the connection of a participant that has left."""
        conn = self._participants.pop(identity, None)
        if conn is None:
            return
        conn.drop()
        self._control.connection_closed(conn.id)

    def _on_data(self, packet: rtc.DataPacket) -> None:
        """Hand a protocol message from a participant to the session as that participant's."""
        channel = TOPIC_CHANNELS.get(packet.topic or "")
        if channel is None or packet.participant is None:
            return
        conn = self._participants.get(packet.participant.identity)
        if conn is not None:
            self._control.message_received(conn.id, packet.data, ProtocolVersion.V1, channel)

    def _on_byte_stream(self, reader: rtc.ByteStreamReader, identity: str) -> None:
        """Read a protocol message sent as a byte stream, then hand it on like a packet."""
        task = asyncio.get_running_loop().create_task(self._read_message(reader, identity))
        self._readers.add(task)
        task.add_done_callback(self._readers.discard)

    async def _read_message(self, reader: rtc.ByteStreamReader, identity: str) -> None:
        channel = TOPIC_CHANNELS[reader.info.topic]
        data = bytearray()
        async for chunk in reader:
            data.extend(chunk)
            if len(data) > MAX_STREAMED_MESSAGE_BYTES:
                logger.warning(
                    "dropped a protocol message larger than the limit",
                    participant=identity,
                    limit=MAX_STREAMED_MESSAGE_BYTES,
                )
                return
        conn = self._participants.get(identity)
        if conn is not None:
            self._control.message_received(conn.id, bytes(data), ProtocolVersion.V1, channel)

    def _on_disconnected(self, reason: Any = None) -> None:
        """Close the room connection when the room drops the session's participant."""
        if self.state is not RoomState.OPEN:
            return
        name = rtc.DisconnectReason.Name(reason) if isinstance(reason, int) else str(reason)
        task = asyncio.get_running_loop().create_task(self._finish(_ROOM_LOST.format(reason=name)))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)
