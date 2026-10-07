from reactor_runtime.core import (
    ClientTrackDirection,
    TrackKind,
    TransportReading,
    TransportTrackReading,
)


def test_a_reading_with_nothing_measured_is_empty() -> None:
    reading = TransportReading()

    assert reading.metrics == {}
    assert reading.tracks == ()


def test_a_track_reading_holds_its_facts_beside_its_metrics() -> None:
    track = TransportTrackReading(
        track_name="main_video",
        kind=TrackKind.VIDEO,
        direction=ClientTrackDirection.RECVONLY,
        codec="VP9",
        metrics={"bitrate_bps": 1_500_000.0},
    )
    reading = TransportReading(metrics={"connection_rtt_ms": 40.0}, tracks=(track,))

    assert reading.tracks[0].direction is ClientTrackDirection.RECVONLY
    assert reading.tracks[0].metrics["bitrate_bps"] == 1_500_000.0
