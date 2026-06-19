import numpy as np

from reactor_runtime.core import (
    ConnectionCapabilities,
    ConnId,
    Health,
    HealthStatus,
    InputFrame,
    MediaBundle,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)


def test_conn_id_is_an_int() -> None:
    conn_id = ConnId(7)
    assert conn_id == 7
    assert isinstance(conn_id, int)


def test_input_frame_carries_payload_and_pts() -> None:
    data = np.zeros((2, 2, 3), dtype=np.uint8)
    frame = InputFrame(data=data, pts=1.5)
    assert frame.data is data
    assert frame.pts == 1.5


def test_input_frame_defaults_pts_to_none() -> None:
    assert InputFrame(data=np.zeros((1, 1, 3), dtype=np.uint8)).pts is None


def test_input_frame_equality_is_identity() -> None:
    data = np.ones((1, 1, 3), dtype=np.uint8)
    assert InputFrame(data=data) != InputFrame(data=data.copy())
    assert len({InputFrame(data=data), InputFrame(data=data)}) == 2


def test_media_bundle_lookups() -> None:
    video = TrackData(
        info=TrackInfo(name="main_video", kind=TrackKind.VIDEO, direction=TrackDirection.OUT),
        data=np.zeros((1, 1, 3), dtype=np.uint8),
    )
    audio = TrackData(
        info=TrackInfo(name="main_audio", kind=TrackKind.AUDIO, rate=48000.0),
        data=np.zeros((1, 4), dtype=np.int16),
    )
    bundle = MediaBundle(tracks={"main_video": video, "main_audio": audio})

    assert bundle.get_track("main_video") is video
    assert bundle.get_track("missing") is None
    assert set(bundle.get_tracks()) == {video, audio}
    assert bundle.get_tracks_by_kind(TrackKind.AUDIO) == [audio]


def test_connection_capabilities_default_to_data_only() -> None:
    caps = ConnectionCapabilities()
    assert caps.carries_data is True
    assert caps.carries_video is False
    assert caps.carries_audio is False


def test_health_aggregate_keeps_the_worst_status() -> None:
    rolled = Health.aggregate(
        [
            Health.healthy(),
            Health(HealthStatus.DEGRADED, "cache cold"),
            Health(HealthStatus.UNHEALTHY, "model thread down"),
        ]
    )
    assert rolled.status is HealthStatus.UNHEALTHY
    assert rolled.detail is not None
    assert "model thread down" in rolled.detail
    assert "cache cold" in rolled.detail


def test_health_aggregate_of_nothing_is_healthy() -> None:
    rolled = Health.aggregate([])
    assert rolled.status is HealthStatus.HEALTHY
    assert rolled.detail is None


def test_health_aggregate_omits_healthy_part_details() -> None:
    rolled = Health.aggregate(
        [
            Health.healthy("warming complete"),
            Health(HealthStatus.DEGRADED, "cache cold"),
        ]
    )
    assert rolled.status is HealthStatus.DEGRADED
    assert rolled.detail == "cache cold"
