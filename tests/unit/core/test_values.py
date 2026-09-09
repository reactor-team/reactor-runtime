import numpy as np
import pytest

from reactor_runtime.core import (
    ConnectionCapabilities,
    ConnId,
    Health,
    HealthStatus,
    InputFrame,
    MediaBundle,
    RuntimeState,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.core.values import split_batch


def video_track(data: np.ndarray, metadata: bytes | list[bytes] | None = None) -> TrackData:
    info = TrackInfo(name="main_video", kind=TrackKind.VIDEO, direction=TrackDirection.OUT)
    return TrackData(info=info, data=data, metadata=metadata)


def audio_track(data: np.ndarray, metadata: bytes | list[bytes] | None = None) -> TrackData:
    info = TrackInfo(name="main_audio", kind=TrackKind.AUDIO, rate=48_000.0)
    return TrackData(info=info, data=data, metadata=metadata)


def batch(n_frames: int) -> np.ndarray:
    return np.zeros((n_frames, 2, 2, 3), dtype=np.uint8)


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


def test_track_data_carries_no_metadata_by_default() -> None:
    assert video_track(np.zeros((1, 1, 3), dtype=np.uint8)).metadata is None


def test_split_batch_gives_each_frame_its_own_metadata() -> None:
    bundle = MediaBundle(tracks={"main_video": video_track(batch(3), [b"a", b"b", b"c"])})
    assert [frame.tracks["main_video"].metadata for frame in split_batch(bundle)] == [
        b"a",
        b"b",
        b"c",
    ]


def test_split_batch_repeats_one_metadata_value_across_a_batch() -> None:
    bundle = MediaBundle(tracks={"main_video": video_track(batch(3), b"same")})
    assert [frame.tracks["main_video"].metadata for frame in split_batch(bundle)] == [b"same"] * 3


def test_split_batch_carries_metadata_alongside_a_split_audio_track() -> None:
    bundle = MediaBundle(
        tracks={
            "main_video": video_track(batch(2), [b"a", b"b"]),
            "main_audio": audio_track(np.zeros((1, 8), dtype=np.int16)),
        }
    )
    frames = split_batch(bundle)
    assert [frame.tracks["main_video"].metadata for frame in frames] == [b"a", b"b"]
    assert all(frame.tracks["main_audio"].metadata is None for frame in frames)


def test_split_batch_rejects_metadata_that_does_not_cover_the_batch() -> None:
    bundle = MediaBundle(tracks={"main_video": video_track(batch(3), [b"a", b"b"])})
    with pytest.raises(ValueError, match="2 metadata entries for 3 frames"):
        split_batch(bundle)


def test_split_batch_resolves_a_one_frame_batch_to_a_single_value() -> None:
    bundle = MediaBundle(tracks={"main_video": video_track(batch(1), [b"only"])})
    (frame,) = split_batch(bundle)
    assert frame.tracks["main_video"].metadata == b"only"


def test_split_batch_leaves_an_unbatched_bundle_untouched() -> None:
    track = video_track(np.zeros((2, 2, 3), dtype=np.uint8), b"one")
    bundle = MediaBundle(tracks={"main_video": track})
    assert split_batch(bundle) == [bundle]


def test_connection_capabilities_default_to_no_media() -> None:
    caps = ConnectionCapabilities()
    assert caps.carries_video is False
    assert caps.carries_audio is False


def test_health_status_is_two_valued() -> None:
    assert {status.value for status in HealthStatus} == {"healthy", "unhealthy"}


def test_runtime_state_names_the_four_lifecycle_words() -> None:
    assert [state.value for state in RuntimeState] == [
        "loading",
        "available",
        "serving",
        "terminated",
    ]


def test_health_aggregate_keeps_the_worst_status() -> None:
    rolled = Health.aggregate(
        [
            Health.healthy(),
            Health(HealthStatus.UNHEALTHY, "model thread down"),
            Health(HealthStatus.UNHEALTHY, "http server not started"),
        ]
    )
    assert rolled.status is HealthStatus.UNHEALTHY
    assert rolled.detail is not None
    assert "model thread down" in rolled.detail
    assert "http server not started" in rolled.detail


def test_health_aggregate_of_nothing_is_healthy() -> None:
    rolled = Health.aggregate([])
    assert rolled.status is HealthStatus.HEALTHY
    assert rolled.detail is None


def test_health_aggregate_omits_healthy_part_details() -> None:
    rolled = Health.aggregate(
        [
            Health.healthy("warming complete"),
            Health(HealthStatus.UNHEALTHY, "model thread down"),
        ]
    )
    assert rolled.status is HealthStatus.UNHEALTHY
    assert rolled.detail == "model thread down"
