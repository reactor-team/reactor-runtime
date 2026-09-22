"""The inbound message router.

The one place where "what does this payload mean" lives. A payload that arrived
on any connection is decoded here through the wire :class:`~reactor_runtime.protocol.Codec`
— the single seam the JSON-to-protobuf swap touches — and routed by what it is:
a ping marks liveness, a command is handed onward as a decoded
:class:`InboundCommand` whose ``request_id`` is guaranteed present.

The gateway is transport-neutral: a WebSocket frame and a WebRTC data-channel
frame decode the same way. It does no validation — that belongs to the model
contract — and does not resolve uploads or submit to the model: it decodes and
emits, and the handler it is given owns what happens next.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from google.protobuf.message import DecodeError

from reactor_runtime.core import (
    ClientConnectionStat,
    ClientStatsBatch,
    ClientTrackStat,
    ConnectionSink,
    ConnId,
)
from reactor_runtime.protocol import Channel, Codec, ProtocolVersion, select
from reactor_runtime.protocol.common import struct_to_dict
from reactor_wire.v1 import control_pb2, data_pb2, platform_pb2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InboundCommand:
    """A decoded client command, ready for upload resolution and submission.

    Attributes:
        name: The command name the client invoked.
        args: The command arguments as a plain mapping, exactly as decoded — no
            validation or coercion has been applied.
        uploads: Argument name to the ``upload_id`` it references, still
            unresolved. The handler fetches the bytes for these.
        conn_id: The connection the command arrived on.
        request_id: The client's correlation id, guaranteed present — carried
            from the client when supplied, minted here only when absent.
        received_at: When the frame carrying this command reached the runtime, on
            a monotonic clock. Stamped by the caller at the transport edge, so it
            precedes the decode and the wait for the event loop.
    """

    name: str
    args: Mapping[str, Any]
    uploads: Mapping[str, str]
    conn_id: ConnId
    request_id: str
    received_at: float


CommandHandler = Callable[[InboundCommand], Awaitable[None]]
"""Receives each decoded command. Owns upload resolution and submission."""


class MessageGateway:
    """Decodes inbound frames and routes them by what they mean.

    Holds the upward sink and selects a codec per inbound frame from the version
    the connection negotiated, so one gateway decodes every client in the
    version it speaks. Pings mark liveness through the sink; commands are decoded
    into an :class:`InboundCommand` and passed to the handler. Other control
    messages decode cleanly but are not routed here — the components that serve
    them are wired by a later layer.
    """

    def __init__(
        self,
        *,
        sink: ConnectionSink,
        on_command: CommandHandler,
    ) -> None:
        """Bind the gateway to its upward sink and command handler.

        Codecs are selected per inbound frame from the negotiated version and
        cached, one per version, so repeated frames on a connection reuse one
        codec.
        """
        self._sink = sink
        self._on_command = on_command
        self._codecs: dict[ProtocolVersion, Codec] = {}

    def _codec_for(self, version: ProtocolVersion) -> Codec:
        """Return the codec for *version*, building and caching it on first use."""
        codec = self._codecs.get(version)
        if codec is None:
            codec = select(version)
            self._codecs[version] = codec
        return codec

    async def handle(
        self,
        conn_id: ConnId,
        payload: bytes | str,
        channel: Channel,
        version: ProtocolVersion,
        *,
        received_at: float,
    ) -> None:
        """Decode one inbound frame and route it.

        The frame is decoded with the codec the connection negotiated
        (*version*), so each client is read in the version it speaks. A ping
        notes liveness on the sink. A command is handed to the handler as an
        :class:`InboundCommand` with a guaranteed ``request_id``. Any other
        decoded message is left for a later layer to route. A frame that cannot
        be decoded is dropped with a warning rather than raised, so one malformed
        frame from any client cannot break the transport read loop.

        *received_at* is when the frame reached the runtime, on a monotonic clock.
        The caller stamps it at the transport edge and it rides the decoded
        command, because the gateway runs as a task and cannot see how long the
        frame waited for the event loop.
        """
        try:
            message = self._codec_for(version).decode_inbound(payload, channel)
        except (ValueError, DecodeError) as exc:
            logger.warning("MessageGateway dropped an undecodable frame on %s: %s", channel, exc)
            return

        if isinstance(message, control_pb2.ControlClientMessage):
            self._route_control(conn_id, message)
            return

        if (
            isinstance(message, data_pb2.DataClientMessage)
            and message.WhichOneof("payload") == "command"
        ):
            command = message.command
            await self._on_command(
                InboundCommand(
                    name=command.type,
                    args=struct_to_dict(command.data),
                    uploads={param: ref.upload_id for param, ref in command.uploads.items()},
                    conn_id=conn_id,
                    request_id=message.request_id or _new_request_id(),
                    received_at=received_at,
                )
            )
            return

        logger.debug("MessageGateway received an unrouted message on %s", channel)

    def _route_control(self, conn_id: ConnId, message: control_pb2.ControlClientMessage) -> None:
        """Dispatch a decoded control message by its kind.

        A ping marks liveness; the track verbs cross to the sink for the runner
        to act on — resume/pause gate one connection's outbound streams, and
        publish/unpublish drive the cross-connection publisher arbitration, with
        the publish request carrying its correlation id for the reply. A
        file-uploaded notification crosses with the id of the uploaded slot for
        the runner to resolve. A clip or recording request crosses with a
        correlation id — the client's when present, minted here when absent (the
        shipped client correlates by receipt order) — so the reply can be
        addressed back. A client-stats batch crosses as decoded sample values,
        stripped of the wire types, for the sink to log with the connection's
        identity attached. Anything else decodes cleanly but has no handler
        here yet.
        """
        which = message.WhichOneof("payload")
        if which == "ping":
            self._sink.keepalive(conn_id)
        elif which == "resume_track":
            self._sink.resume_track(conn_id, message.resume_track.name)
        elif which == "pause_track":
            self._sink.pause_track(conn_id, message.pause_track.name)
        elif which == "publish_track":
            self._sink.publish_requested(conn_id, message.publish_track.name, message.request_id)
        elif which == "unpublish_track":
            self._sink.unpublish_track(conn_id, message.unpublish_track.name)
        elif which == "file_uploaded":
            self._sink.file_uploaded(conn_id, message.file_uploaded.upload_id)
        elif which == "request_schema":
            self._sink.schema_requested(conn_id, message.request_id)
        elif which == "request_clip":
            self._sink.clip_requested(
                conn_id,
                message.request_clip.duration_seconds,
                message.request_id or _new_request_id(),
            )
        elif which == "request_recording":
            self._sink.recording_requested(conn_id, message.request_id or _new_request_id())
        elif which == "client_stats":
            self._sink.client_stats_received(conn_id, _decode_client_stats(message.client_stats))
        else:
            logger.debug("MessageGateway received an unrouted control message: %s", which)


_TRACK_KIND_NAMES = {
    platform_pb2.TrackKind.TRACK_KIND_VIDEO: "video",
    platform_pb2.TrackKind.TRACK_KIND_AUDIO: "audio",
}
_TRACK_DIRECTION_NAMES = {
    platform_pb2.TrackDirection.TRACK_DIRECTION_RECVONLY: "recvonly",
    platform_pb2.TrackDirection.TRACK_DIRECTION_SENDONLY: "sendonly",
}
_VIDEO_CODEC_NAMES = {
    platform_pb2.VideoCodec.VIDEO_CODEC_VP8: "VP8",
    platform_pb2.VideoCodec.VIDEO_CODEC_VP9: "VP9",
    platform_pb2.VideoCodec.VIDEO_CODEC_AV1: "AV1",
    platform_pb2.VideoCodec.VIDEO_CODEC_H264: "H264",
    platform_pb2.VideoCodec.VIDEO_CODEC_H265: "H265",
}
_AUDIO_CODEC_NAMES = {
    platform_pb2.AudioCodec.AUDIO_CODEC_OPUS: "opus",
}


def _decode_codec(stat: platform_pb2.ClientTrackStat) -> str:
    """Read whichever codec arm *stat* set, as a plain name.

    Empty for ``*_UNSPECIFIED`` (the browser hasn't reported a codec yet) and
    for a value this runtime doesn't recognize yet (an older runtime reading
    a newer client's codec) — never raises on either.
    """
    which = stat.WhichOneof("codec")
    if which == "video_codec":
        return _VIDEO_CODEC_NAMES.get(stat.video_codec, "")
    if which == "audio_codec":
        return _AUDIO_CODEC_NAMES.get(stat.audio_codec, "")
    return ""


def _decode_client_stats(message: platform_pb2.ClientStats) -> ClientStatsBatch:
    """Convert a decoded ``ClientStats`` batch into plain values."""
    track_stats = [
        ClientTrackStat(
            timestamp=stat.timestamp,
            track_name=stat.track_name,
            kind=_TRACK_KIND_NAMES.get(stat.kind, ""),
            direction=_TRACK_DIRECTION_NAMES.get(stat.direction, ""),
            codec=_decode_codec(stat),
            paused=stat.paused,
            metrics=dict(stat.metrics),
        )
        for stat in message.track_stats
    ]
    connection_stat = None
    if message.HasField("connection_stat"):
        connection_stat = ClientConnectionStat(
            timestamp=message.connection_stat.timestamp,
            metrics=dict(message.connection_stat.metrics),
        )
    return ClientStatsBatch(track_stats=track_stats, connection_stat=connection_stat)


def _new_request_id() -> str:
    """Mint a fallback correlation id for a client that supplied none."""
    return uuid.uuid4().hex
