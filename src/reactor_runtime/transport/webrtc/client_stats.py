"""Read the client-stats batches a WebRTC client sends on its control channel.

A client reports its own view of its receive-side quality as a ``ClientStats``
control message. The batch never reaches the model and is no session command,
so the WebRTC transport picks it out of the control traffic itself and hands
it to the sink as plain values, next to the peer stats it samples on its own
(:mod:`.stats`). Every other frame goes on to the message gateway unchanged.
"""

from __future__ import annotations

from google.protobuf.message import DecodeError

from reactor_runtime.core import (
    ClientConnectionStat,
    ClientStatsBatch,
    ClientTrackDirection,
    ClientTrackStat,
    FrameStage,
    TrackKind,
)
from reactor_runtime.protocol import Channel, Codec, ProtocolVersion, select
from reactor_wire.v1 import control_pb2, platform_pb2

_TRACK_KINDS = {
    platform_pb2.TrackKind.TRACK_KIND_VIDEO: TrackKind.VIDEO,
    platform_pb2.TrackKind.TRACK_KIND_AUDIO: TrackKind.AUDIO,
}
_TRACK_DIRECTIONS = {
    platform_pb2.TrackDirection.TRACK_DIRECTION_RECVONLY: ClientTrackDirection.RECVONLY,
    platform_pb2.TrackDirection.TRACK_DIRECTION_SENDONLY: ClientTrackDirection.SENDONLY,
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

# One codec per wire version, built on first use.
_codecs: dict[ProtocolVersion, Codec] = {}


def read_client_stats(
    payload: bytes | str, version: ProtocolVersion, channel: Channel
) -> ClientStatsBatch | None:
    """Return the client-stats batch *payload* carries, or ``None`` for any other frame.

    Only a control-channel frame can carry one. A frame that does not decode
    is not a batch either: it goes on to the gateway, which reports it.
    Control traffic is low-rate, so decoding a control frame here as well as
    in the gateway costs little.
    """
    if channel is not Channel.CONTROL:
        return None
    codec = _codecs.get(version)
    if codec is None:
        codec = select(version)
        _codecs[version] = codec
    try:
        message = codec.decode_inbound(payload, channel)
    except (ValueError, DecodeError):
        return None
    if not isinstance(message, control_pb2.ControlClientMessage):
        return None
    if message.WhichOneof("payload") != "client_stats":
        return None
    return _decode_client_stats(message.client_stats)


def _decode_codec(stat: platform_pb2.ClientTrackStat) -> str:
    """Read whichever codec arm *stat* set, as a plain name.

    Empty for ``*_UNSPECIFIED`` (the browser hasn't reported a codec yet), for
    a value this runtime doesn't recognize yet (an older runtime reading a
    newer client's codec), and for a codec arm that doesn't match ``kind`` (a
    video reading naming an audio codec or vice versa) — the oneof only
    keeps ``video_codec``/``audio_codec`` from being set at the same time, it
    doesn't tie either one to ``kind``, so a malformed batch can still name
    the wrong one. Never raises on any of these; an unreadable codec is
    reported the same as an unreported one.
    """
    which = stat.WhichOneof("codec")
    if which == "video_codec" and stat.kind == platform_pb2.TrackKind.TRACK_KIND_VIDEO:
        return _VIDEO_CODEC_NAMES.get(stat.video_codec, "")
    if which == "audio_codec" and stat.kind == platform_pb2.TrackKind.TRACK_KIND_AUDIO:
        return _AUDIO_CODEC_NAMES.get(stat.audio_codec, "")
    return ""


def _decode_client_stats(message: platform_pb2.ClientStats) -> ClientStatsBatch:
    """Convert a decoded ``ClientStats`` batch into plain values."""
    track_stats = [
        ClientTrackStat(
            timestamp=stat.timestamp,
            track_name=stat.track_name,
            kind=_TRACK_KINDS.get(stat.kind),
            direction=_TRACK_DIRECTIONS.get(stat.direction),
            codec=_decode_codec(stat),
            paused=stat.paused,
            metrics=dict(stat.metrics),
            frame_stages=tuple(
                FrameStage(name=s.name, total_ms=s.total_ms, frames=s.frames)
                for s in stat.frame_stages
            ),
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
