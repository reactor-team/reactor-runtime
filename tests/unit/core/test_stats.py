from dataclasses import fields

from reactor_runtime.core import PeerStats, TrackDirection, TrackStat

# The fields every caller could already pass by position. New fields go after
# them, so a positional call keeps its meaning.
_TRACK_STAT_FIELDS = (
    "name",
    "direction",
    "packets_sent",
    "packets_received",
    "packets_lost",
    "retransmitted_packets_sent",
    "bytes_sent",
    "bytes_received",
    "frames_sent",
    "frames_decoded",
    "frames_dropped",
    "nacks",
    "keyframe_requests",
    "jitter",
    "rtt_seconds",
    "loss_ratio",
)
_PEER_STATS_FIELDS = ("rtt_seconds", "available_outgoing_bitrate_bps", "tracks", "media")


def test_a_track_stat_built_by_position_keeps_its_meaning() -> None:
    names = tuple(f.name for f in fields(TrackStat))
    assert names[: len(_TRACK_STAT_FIELDS)] == _TRACK_STAT_FIELDS

    stat = TrackStat("main_video", TrackDirection.OUT, 10)

    assert stat.packets_sent == 10
    assert stat.kind is None


def test_peer_stats_built_by_position_keeps_its_meaning() -> None:
    names = tuple(f.name for f in fields(PeerStats))
    assert names[: len(_PEER_STATS_FIELDS)] == _PEER_STATS_FIELDS

    stats = PeerStats(0.05, 2_000_000.0)

    assert stats.available_outgoing_bitrate_bps == 2_000_000.0
    assert stats.taken_at is None
