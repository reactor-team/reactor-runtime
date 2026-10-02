from reactor_runtime.core import (
    ClientConnectionStat,
    ClientStatsBatch,
    ClientTrackDirection,
    ClientTrackStat,
    TrackKind,
)
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.protocol.v0.codec import V0Codec
from reactor_runtime.protocol.v1.codec import V1Codec
from reactor_runtime.transport.webrtc.client_stats import read_client_stats
from reactor_wire.v1 import control_pb2, platform_pb2


def _v1_frame(stats: platform_pb2.ClientStats) -> bytes | str:
    _, frame = V1Codec().encode(control_pb2.ControlClientMessage(client_stats=stats))
    return frame


def _read(stats: platform_pb2.ClientStats) -> ClientStatsBatch:
    batch = read_client_stats(_v1_frame(stats), ProtocolVersion.V1, Channel.CONTROL)
    assert batch is not None
    return batch


def test_a_client_stats_frame_decodes_to_plain_values() -> None:
    batch = _read(
        platform_pb2.ClientStats(
            track_stats=[
                platform_pb2.ClientTrackStat(
                    timestamp=1_700_000_000_000,
                    track_name="main_video",
                    kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO,
                    direction=platform_pb2.TrackDirection.TRACK_DIRECTION_RECVONLY,
                    video_codec=platform_pb2.VideoCodec.VIDEO_CODEC_VP9,
                    paused=True,
                    metrics={"bitrate_bps": 950_000, "frames_per_second": 29.5},
                )
            ],
            connection_stat=platform_pb2.ClientConnectionStat(
                timestamp=1_700_000_000_000,
                metrics={"available_outgoing_bitrate_bps": 2_000_000, "time_to_connect_ms": 850},
            ),
        )
    )

    assert batch == ClientStatsBatch(
        track_stats=[
            ClientTrackStat(
                timestamp=1_700_000_000_000,
                track_name="main_video",
                kind=TrackKind.VIDEO,
                direction=ClientTrackDirection.RECVONLY,
                codec="VP9",
                paused=True,
                metrics={"bitrate_bps": 950_000, "frames_per_second": 29.5},
            )
        ],
        connection_stat=ClientConnectionStat(
            timestamp=1_700_000_000_000,
            metrics={"available_outgoing_bitrate_bps": 2_000_000, "time_to_connect_ms": 850},
        ),
    )


def test_a_batch_with_no_connection_stat_decodes_to_none() -> None:
    batch = _read(
        platform_pb2.ClientStats(
            track_stats=[platform_pb2.ClientTrackStat(timestamp=1_700_000_000_000)]
        )
    )
    assert batch.connection_stat is None


def test_an_audio_track_decodes_its_own_codec_arm() -> None:
    batch = _read(
        platform_pb2.ClientStats(
            track_stats=[
                platform_pb2.ClientTrackStat(
                    timestamp=1_700_000_000_000,
                    track_name="main_audio",
                    kind=platform_pb2.TrackKind.TRACK_KIND_AUDIO,
                    direction=platform_pb2.TrackDirection.TRACK_DIRECTION_SENDONLY,
                    audio_codec=platform_pb2.AudioCodec.AUDIO_CODEC_OPUS,
                )
            ]
        )
    )
    (stat,) = batch.track_stats
    assert stat.kind is TrackKind.AUDIO
    assert stat.direction is ClientTrackDirection.SENDONLY
    assert stat.codec == "opus"


def test_an_unspecified_codec_decodes_to_an_empty_string() -> None:
    batch = _read(
        platform_pb2.ClientStats(
            track_stats=[
                platform_pb2.ClientTrackStat(
                    timestamp=1_700_000_000_000,
                    track_name="main_video",
                    kind=platform_pb2.TrackKind.TRACK_KIND_VIDEO,
                )
            ]
        )
    )
    assert batch.track_stats[0].codec == ""


def test_a_codec_arm_contradicting_kind_decodes_to_an_empty_string() -> None:
    # A malformed batch: kind says audio, but the codec arm set is
    # video_codec. The oneof only keeps the two codec arms from both being set
    # at once — it doesn't tie either one to kind — so this is a schema-legal
    # message decode must still not crash on.
    batch = _read(
        platform_pb2.ClientStats(
            track_stats=[
                platform_pb2.ClientTrackStat(
                    timestamp=1_700_000_000_000,
                    track_name="main_audio",
                    kind=platform_pb2.TrackKind.TRACK_KIND_AUDIO,
                    video_codec=platform_pb2.VideoCodec.VIDEO_CODEC_VP9,
                )
            ]
        )
    )
    assert batch.track_stats[0].codec == ""


def test_an_unspecified_kind_and_direction_decode_to_none() -> None:
    batch = _read(
        platform_pb2.ClientStats(
            track_stats=[platform_pb2.ClientTrackStat(timestamp=1_700_000_000_000)]
        )
    )
    (stat,) = batch.track_stats
    assert stat.kind is None
    assert stat.direction is None


def test_any_other_control_frame_is_not_a_batch() -> None:
    _, ping = V1Codec().encode(control_pb2.ControlClientMessage(ping=platform_pb2.Ping()))
    assert read_client_stats(ping, ProtocolVersion.V1, Channel.CONTROL) is None


def test_a_frame_off_the_control_channel_is_not_a_batch() -> None:
    frame = _v1_frame(platform_pb2.ClientStats())
    assert read_client_stats(frame, ProtocolVersion.V1, Channel.DATA) is None


def test_an_undecodable_control_frame_is_not_a_batch() -> None:
    assert read_client_stats(b"\xff\xfe not a frame", ProtocolVersion.V1, Channel.CONTROL) is None


def test_a_v0_control_frame_is_not_a_batch() -> None:
    channel, ping = V0Codec().encode(control_pb2.ControlClientMessage(ping=platform_pb2.Ping()))
    assert read_client_stats(ping, ProtocolVersion.V0, channel) is None
