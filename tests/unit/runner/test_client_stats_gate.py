from typing import Any

import pytest

from reactor_runtime.core import (
    ClientConnectionStat,
    ClientStatsBatch,
    ClientTrackDirection,
    ClientTrackStat,
    ConnId,
    FrameStage,
    TrackKind,
)
from reactor_runtime.runner.client_stats import (
    MAX_FRAME_STAGES,
    MAX_METRICS,
    MAX_NAME_LENGTH,
    MAX_TRACKS,
    MIN_INTERVAL_SECONDS,
    ClientStatsGate,
    fits,
    to_detail,
)


def _track(**overrides: Any) -> ClientTrackStat:
    fields: dict[str, Any] = {
        "timestamp": 1_700_000_000_000,
        "track_name": "main_video",
        "kind": TrackKind.VIDEO,
        "direction": ClientTrackDirection.RECVONLY,
        "codec": "VP9",
        "paused": False,
        "metrics": {"frames_per_second": 30.0},
    }
    fields.update(overrides)
    return ClientTrackStat(**fields)


def _batch(
    *tracks: ClientTrackStat, connection_metrics: dict[str, float] | None = None
) -> ClientStatsBatch:
    connection_stat = None
    if connection_metrics is not None:
        connection_stat = ClientConnectionStat(
            timestamp=1_700_000_000_000, metrics=connection_metrics
        )
    return ClientStatsBatch(track_stats=list(tracks or [_track()]), connection_stat=connection_stat)


def _gate() -> tuple[ClientStatsGate, list[float]]:
    now = [1000.0]
    return ClientStatsGate(clock=lambda: now[0]), now


def test_a_batch_that_follows_its_connections_last_too_soon_is_dropped() -> None:
    gate, now = _gate()

    assert gate.accept(ConnId(3), _batch()) is None
    now[0] += MIN_INTERVAL_SECONDS / 2
    assert gate.accept(ConnId(3), _batch()) == "sent too soon"
    # Each connection has its own clock.
    assert gate.accept(ConnId(4), _batch()) is None
    now[0] += MIN_INTERVAL_SECONDS
    assert gate.accept(ConnId(3), _batch()) is None


def test_a_dropped_batch_does_not_move_the_connections_clock() -> None:
    gate, now = _gate()
    assert gate.accept(ConnId(3), _batch()) is None
    now[0] += MIN_INTERVAL_SECONDS / 2
    assert gate.accept(ConnId(3), _batch()) == "sent too soon"

    now[0] += MIN_INTERVAL_SECONDS / 2

    assert gate.accept(ConnId(3), _batch()) is None


def test_forget_and_clear_drop_the_last_accepted_times() -> None:
    gate, _ = _gate()
    assert gate.accept(ConnId(3), _batch()) is None
    assert gate.accept(ConnId(4), _batch()) is None

    gate.forget(ConnId(3))
    assert gate.accept(ConnId(3), _batch()) is None
    assert gate.accept(ConnId(4), _batch()) == "sent too soon"

    gate.clear()
    assert gate.accept(ConnId(4), _batch()) is None


def test_a_batch_too_large_is_dropped_and_leaves_the_clock_alone() -> None:
    gate, _ = _gate()

    too_large = _batch(*[_track()] * (MAX_TRACKS + 1))

    assert gate.accept(ConnId(3), too_large) == "too large"
    assert gate.accept(ConnId(3), _batch()) is None


@pytest.mark.parametrize(
    "batch",
    [
        pytest.param(_batch(*[_track()] * (MAX_TRACKS + 1)), id="too many tracks"),
        pytest.param(
            _batch(_track(metrics={f"metric_{i}": 1.0 for i in range(MAX_METRICS + 1)})),
            id="too many metrics",
        ),
        pytest.param(
            _batch(connection_metrics={f"metric_{i}": 1.0 for i in range(MAX_METRICS + 1)}),
            id="too many connection metrics",
        ),
        pytest.param(
            _batch(_track(metrics={"m" * (MAX_NAME_LENGTH + 1): 1.0})), id="a metric name too long"
        ),
        pytest.param(
            _batch(_track(track_name="t" * (MAX_NAME_LENGTH + 1))), id="a track name too long"
        ),
        pytest.param(_batch(_track(codec="c" * (MAX_NAME_LENGTH + 1))), id="a codec name too long"),
    ],
)
def test_a_batch_larger_than_the_sdk_sends_does_not_fit(batch: ClientStatsBatch) -> None:
    assert not fits(batch)


def test_a_batch_at_the_limits_fits() -> None:
    track = _track(
        track_name="t" * MAX_NAME_LENGTH,
        metrics={f"{i:0{MAX_NAME_LENGTH}d}": 1.0 for i in range(MAX_METRICS)},
    )
    assert fits(_batch(*[track] * MAX_TRACKS, connection_metrics={"rtt_ms": 1.0}))


def test_to_detail_carries_the_readings_and_leaves_out_values_json_cannot_carry() -> None:
    batch = _batch(
        _track(
            metrics={"jitter_ms": float("nan"), "bitrate_bps": float("inf"), "packets_lost": 4.0},
            frame_stages=(
                FrameStage("jitter_buffer", total_ms=1500.0, frames=150),
                FrameStage("decode", total_ms=float("inf"), frames=150),
                FrameStage("delivery", total_ms=3.0, frames=0),
                FrameStage("", total_ms=3.0, frames=1),
            ),
        ),
        connection_metrics={"connection_rtt_ms": 25.0, "bad": float("-inf")},
    )

    assert to_detail(batch) == {
        "track_stats": [
            {
                "timestamp": 1_700_000_000_000,
                "track_name": "main_video",
                "kind": "video",
                "direction": "recvonly",
                "codec": "VP9",
                "paused": False,
                "metrics": {"packets_lost": 4.0},
                # Only a stage with a name, frames and a finite time has an average.
                "frame_stages": [{"name": "jitter_buffer", "total_ms": 1500.0, "frames": 150}],
            }
        ],
        "connection_stat": {"timestamp": 1_700_000_000_000, "metrics": {"connection_rtt_ms": 25.0}},
    }


def test_to_detail_without_a_connection_reading() -> None:
    assert to_detail(_batch())["connection_stat"] is None


def test_a_track_with_more_stages_than_a_client_sends_is_too_large() -> None:
    stage = FrameStage("decode", total_ms=1.0, frames=1)
    assert fits(_batch(_track(frame_stages=(stage,) * MAX_FRAME_STAGES)))
    assert not fits(_batch(_track(frame_stages=(stage,) * (MAX_FRAME_STAGES + 1))))
    long_name = FrameStage("x" * (MAX_NAME_LENGTH + 1), total_ms=1.0, frames=1)
    assert not fits(_batch(_track(frame_stages=(long_name,))))
