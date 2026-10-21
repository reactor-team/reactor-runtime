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
closes, and the session goes on without it; a session left with no connection
ends after the orphan timeout, as any session does. When the session ends, its
teardown closes the room connection, which leaves the room.

Every other participant is a client of the session, from the moment the room
connection is registered: one :class:`ParticipantConnection` each, on a
connection id minted for the ``livekit`` transport, opened when the participant
joins and closed when it leaves. The Reactor protocol messages a participant
sends, as data packets or byte streams on the protocol's topics, reach the
session as that participant's. When the room connection closes, every
participant connection closes before it.

The room's own signals reach the model as the manifest's ``runtime.livekit``
block maps them (:class:`RoomSignals`): an RPC call to a mapped method runs its
command and answers with the result, and a data packet on a mapped topic runs
its command and answers nothing. Neither is answered on the Reactor protocol's
topics. Participants' tracks and byte streams feed the model's input tracks
(:class:`RoomInputs`).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum
from typing import Any

from livekit import rtc

from reactor_runtime.codes import INVALID_COMMAND
from reactor_runtime.core import (
    ConnId,
    LiveKitConfig,
    ReplyRoute,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.log import get_logger
from reactor_runtime.protocol import ProtocolVersion
from reactor_runtime.transport.livekit.connection import (
    _INITIAL_VIDEO_SIZE,
    AUDIO_QUEUE_MS,
    RoomConnection,
)
from reactor_runtime.transport.livekit.media_in import RoomInputs
from reactor_runtime.transport.livekit.participant import TOPIC_CHANNELS, ParticipantConnection
from reactor_runtime.transport.livekit.signals import RoomSignals, payload_args
from reactor_runtime.transport.livekit.version import TRANSPORT_NAME
from reactor_runtime.transport.router import ConnectionsExhaustedError, SessionControl

logger = get_logger(__name__)

JOIN_TIMEOUT_SECONDS = 30.0
"""How long a join, connecting and publishing the output tracks, may take before it fails."""

_JOIN_REFUSED = "the LiveKit room could not be joined: {cause}"
_JOIN_TIMED_OUT = "joining the LiveKit room timed out"
_ROOM_LOST = "the LiveKit room connection closed ({reason})"
_LEFT = "the room was left on request"
_SESSION_ENDED = "the session ended"


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
        config: The manifest's ``runtime.livekit`` block.
        room: The room to join; a fresh ``rtc.Room`` unless one is given.
    """

    def __init__(
        self,
        conn_id: ConnId,
        url: str,
        token: str,
        control: SessionControl,
        config: LiveKitConfig | None = None,
        *,
        room: rtc.Room | None = None,
    ) -> None:
        self.conn_id = conn_id
        self.state = RoomState.JOINING
        self.reason: str | None = None
        self._url = url
        self._token = token
        self._control = control
        self._config = config or LiveKitConfig()
        self._signals = RoomSignals(self._config)
        self._inputs: RoomInputs | None = None
        self._room = room if room is not None else rtc.Room()
        self._connection: RoomConnection | None = None
        self._join: asyncio.Task[None] | None = None
        self._closing: set[asyncio.Task[None]] = set()
        # An abandoned attempt that will leave the room once it settles: set
        # when the attempt is abandoned, cleared once the room is left.
        self._leave_pending = False
        # The participants that are clients of the session, by identity. Empty
        # until the session registers the room connection: nobody in the room
        # is a client of a session the room is not connected to.
        self._participants: dict[str, ParticipantConnection] = {}
        self._registered = False
        self._room.on("disconnected", self._on_disconnected)
        self._room.on("participant_connected", lambda p: self._add_participant(p.identity))
        self._room.on("participant_disconnected", lambda p: self._remove_participant(p.identity))
        self._room.on("data_received", self._on_data)
        self._room.on("track_subscribed", self._on_track_subscribed)
        self._room.on("track_unsubscribed", self._on_track_unsubscribed)
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
    def settled(self) -> bool:
        """Whether the room connection is over and nothing of it still connects or leaves.

        A join that timed out or was left keeps its attempt running until it
        settles, and leaves the room after it: until then the room connection
        still holds the room, and its identity in it.
        """
        if self.active or self._leave_pending:
            return False
        return all(task is None or task.done() for task in (self._join, *self._closing))

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

        A join still in flight is abandoned: it is not cancelled, and the room
        is left once it settles.
        """
        await self._finish(_LEFT)

    async def retire(self) -> None:
        """End a room connection whose session has ended, as :meth:`leave` does."""
        await self._finish(_SESSION_ENDED)

    async def _run_join(self) -> None:
        """Join the room, publish the output tracks, and open the room connection.

        Connecting and publishing run as one attempt that is never cancelled:
        the SDK cannot undo a native operation cancelled half way. The deadline
        covers the whole attempt. One that passes it, or that is abandoned by
        a leave, keeps running, and the room is left once it settles.
        """
        attempt = asyncio.get_running_loop().create_task(
            self._connect_and_publish(), name=f"livekit-attempt-{self.conn_id}"
        )
        done, _ = await asyncio.wait({attempt}, timeout=JOIN_TIMEOUT_SECONDS)
        if not done:
            self._leave_pending = True
            attempt.add_done_callback(self._leave_when_settled)
            if self.joining:
                await self._fail(_JOIN_TIMED_OUT, leave=False)
            return
        if not self.joining:
            # Left while joining: the attempt has settled, so leaving is safe.
            attempt.exception()
            await self._disconnect()
            return
        error = attempt.exception()
        if error is not None:
            await self._fail(_JOIN_REFUSED.format(cause=str(error) or type(error).__name__))
            return
        connection = attempt.result()
        self._connection = connection
        self.state = RoomState.OPEN
        connection.start()
        if not self._control.connection_opened(connection):
            # The session it was opened for has ended; closing it leaves the room.
            return
        self._registered = True
        for participant in list(self._room.remote_participants.values()):
            self._add_participant(participant.identity)
            for publication in participant.track_publications.values():
                if publication.track is not None:
                    self._on_track_subscribed(publication.track, publication, participant)
        logger.info(
            "joined the LiveKit room",
            conn_id=self.conn_id,
            room=self._room.name,
            identity=self._room.local_participant.identity,
        )

    async def _connect_and_publish(self) -> RoomConnection:
        await self._room.connect(self._url, self._token)
        self._register_methods()
        self._register_inputs()
        return await self._publish(self._tracks(TrackDirection.OUT))

    def _leave_when_settled(self, attempt: asyncio.Task[RoomConnection]) -> None:
        """Leave the room once an abandoned attempt has finished, whatever its outcome."""
        if not attempt.cancelled():
            attempt.exception()  # retrieved, so an abandoned failure is not reported as lost
        task = asyncio.get_running_loop().create_task(self._leave_abandoned())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _leave_abandoned(self) -> None:
        try:
            await self._disconnect()
        finally:
            self._leave_pending = False

    async def _fail(self, reason: str, *, leave: bool = True) -> None:
        """End a join that did not succeed, and report it on the connection.

        Args:
            reason: Why the join failed.
            leave: Whether to leave the room now. A join whose attempt is
                still running leaves once the attempt settles instead.
        """
        self.state = RoomState.FAILED
        self.reason = reason
        if leave:
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
        self._registered = False
        if self._inputs is not None:
            self._inputs.close()
        participants = list(self._participants.values())
        self._participants.clear()
        for participant in participants:
            participant.drop()
            self._control.connection_closed(participant.id)
        if joining:
            # The join leaves the room once its attempt settles.
            logger.info("abandoned a LiveKit room join", conn_id=self.conn_id, reason=reason)
            return
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

    def _tracks(self, direction: TrackDirection) -> dict[str, TrackInfo]:
        """The model's declared tracks that flow in *direction*."""
        tracks: dict[str, TrackInfo] = {}
        for name, entry in self._control.track_map().items():
            if entry["direction"] != direction.value:
                continue
            tracks[name] = TrackInfo(
                name=name,
                kind=TrackKind(entry["kind"]),
                rate=float(entry["rate"]),
                direction=direction,
            )
        return tracks

    def _register_inputs(self) -> None:
        """Feed the input tracks from the byte stream topics that map to them."""
        inputs = RoomInputs(
            self._tracks(TrackDirection.IN),
            self._config.media_inputs,
            self._control.media_received,
            self._control.claim_input_track,
            self._control.unpublish_track,
        )
        self._inputs = inputs
        for topic in inputs.stream_topics:
            self._room.register_byte_stream_handler(topic, self._on_media_stream)

    def _input_freed(self, name: str) -> None:
        if self._inputs is not None:
            self._inputs.input_freed(name)

    def _on_media_stream(self, reader: rtc.ByteStreamReader, identity: str) -> None:
        conn = self._participants.get(identity)
        if conn is None or self._inputs is None:
            reader.close()
            return
        self._inputs.on_byte_stream(reader, identity, conn.id)

    def _on_track_unsubscribed(self, track: rtc.Track, publication: Any, participant: Any) -> None:
        if self._inputs is not None:
            self._inputs.on_track_gone(track.sid)

    def _on_track_subscribed(self, track: rtc.Track, publication: Any, participant: Any) -> None:
        conn = self._participants.get(participant.identity)
        if conn is not None and self._inputs is not None:
            self._inputs.on_track(track, participant.identity, conn.id)

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
            on_input_freed=self._input_freed,
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
            conn_id,
            identity,
            self._room.local_participant,
            asyncio.get_running_loop(),
            lambda data, channel: self._control.message_received(
                conn_id, data, ProtocolVersion.V1, channel
            ),
            self._signals.outbound,
        )
        self._participants[identity] = conn
        if not self._control.connection_opened(conn):
            # Refused, as during a teardown: nothing it holds may outlive it.
            self._participants.pop(identity, None)
            conn.drop()
            self._control.release_conn_id(conn_id, TRANSPORT_NAME)

    def _remove_participant(self, identity: str) -> None:
        """Close the connection of a participant that has left."""
        conn = self._participants.pop(identity, None)
        if conn is None:
            return
        if self._inputs is not None:
            self._inputs.release(identity, conn.id)
        conn.drop()
        self._control.connection_closed(conn.id)

    def _on_data(self, packet: rtc.DataPacket) -> None:
        """Hand a participant's data packet on: a protocol message, or a command."""
        if packet.participant is None:
            return
        conn = self._participants.get(packet.participant.identity)
        if conn is None:
            return
        topic = packet.topic or ""
        channel = TOPIC_CHANNELS.get(topic)
        if channel is not None:
            conn.receive_packet(packet.data, channel)
            return
        command = self._signals.topic_command(topic)
        if command is None:
            return
        args = payload_args(packet.data)
        if args is None:
            logger.info("dropped a data packet that is not a JSON object", topic=topic)
            return
        self._control.command_received(
            conn.id, command, args, f"topic-{uuid.uuid4().hex}", ReplyRoute.NONE
        )

    def _register_methods(self) -> None:
        """Register each mapped RPC method on the session's participant."""
        local = self._room.local_participant
        for method, command in self._signals.inbound.items():
            local.register_rpc_method(method, self._method_handler(command))

    def _method_handler(self, command: str) -> Callable[[rtc.RpcInvocationData], Awaitable[str]]:
        async def handle(call: rtc.RpcInvocationData) -> str:
            """Run *command* for the caller and answer with what it returned."""
            conn = self._participants.get(call.caller_identity)
            if conn is None:
                raise rtc.RpcError(
                    rtc.RpcError.ErrorCode.APPLICATION_ERROR,
                    "The session is not serving this participant.",
                    "not_a_client",
                )
            args = payload_args(call.payload)
            if args is None:
                raise rtc.RpcError(
                    rtc.RpcError.ErrorCode.APPLICATION_ERROR,
                    "The payload must be a JSON object.",
                    INVALID_COMMAND,
                )
            request_id = f"rpc-{call.request_id}"
            reply = conn.expect_reply(request_id)
            self._control.command_received(conn.id, command, args, request_id, ReplyRoute.OBSERVER)
            return await reply

        return handle

    def _on_byte_stream(self, reader: rtc.ByteStreamReader, identity: str) -> None:
        """Hand a protocol message sent as a byte stream to the sender's connection.

        The stream belongs to the connection it opened on. One from a
        participant that is not a client is closed unread.
        """
        conn = self._participants.get(identity)
        if conn is None:
            reader.close()
            return
        conn.receive_stream(reader, TOPIC_CHANNELS[reader.info.topic])

    def _on_disconnected(self, reason: Any = None) -> None:
        """Close the room connection when the room drops the session's participant."""
        if self.state is not RoomState.OPEN:
            return
        name = rtc.DisconnectReason.Name(reason) if isinstance(reason, int) else str(reason)
        task = asyncio.get_running_loop().create_task(self._finish(_ROOM_LOST.format(reason=name)))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)
