from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

import pytest

from reactor_runtime.core import (
    ConnId,
    OutboundMediaHealth,
    PeerStats,
    TrackDirection,
    TrackKind,
    TrackStat,
)
from reactor_runtime.runner.runtime_stats import ModelOutput, TransportRates, to_detail


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def test_a_declared_track_that_emitted_nothing_reads_zero(clock: _Clock) -> None:
    output = ModelOutput(clock)
    output.declare(["main_video", "main_audio"])
    clock.now += 5.0

    readings = output.take()

    assert readings == {
        "main_video": {"frames_emitted": 0.0, "frames_per_second": 0.0},
        "main_audio": {"frames_emitted": 0.0, "frames_per_second": 0.0},
    }


def test_frames_are_counted_in_frames_not_emissions(clock: _Clock) -> None:
    output = ModelOutput(clock)
    output.declare(["main_video"])
    for _ in range(5):
        clock.now += 1.0
        output.emitted("main_video", 30)

    video = output.take()["main_video"]

    assert video["frames_emitted"] == 150.0
    assert video["frames_per_second"] == pytest.approx(30.0)


def test_the_rate_and_the_longest_gap_cover_one_window(clock: _Clock) -> None:
    output = ModelOutput(clock)
    output.declare(["main_video"])
    output.emitted("main_video", 1)
    clock.now += 0.04
    output.emitted("main_video", 1)
    clock.now += 0.5  # a stall
    output.emitted("main_video", 1)
    clock.now += 0.46

    first = output.take()["main_video"]
    assert first["max_emit_interval_ms"] == pytest.approx(500.0)
    assert first["frames_per_second"] == pytest.approx(3.0)

    clock.now += 0.01
    output.emitted("main_video", 1)  # 470 ms after the last emission of the first window
    clock.now += 0.04
    output.emitted("main_video", 1)
    clock.now += 0.95

    second = output.take()["main_video"]
    # The 500 ms stall stays with the window that saw it. A gap that spans the
    # boundary is counted where it ends; the total runs for the whole session.
    assert second["max_emit_interval_ms"] == pytest.approx(470.0)
    assert second["frames_per_second"] == pytest.approx(2.0)
    assert second["frames_emitted"] == 5.0


def test_a_window_without_a_second_emission_reports_no_gap(clock: _Clock) -> None:
    output = ModelOutput(clock)
    output.emitted("main_video", 1)
    clock.now += 1.0
    output.take()
    clock.now += 1.0

    video = output.take()["main_video"]

    assert "max_emit_interval_ms" not in video
    # The track is still quiet, and how long it has been is the stall in progress.
    assert video["ms_since_last_emit"] == pytest.approx(2000.0)


def test_reset_starts_the_session_from_zero_without_a_gap_across_it(clock: _Clock) -> None:
    output = ModelOutput(clock)
    output.declare(["main_video"])
    output.emitted("main_video", 10)
    clock.now += 60.0  # the model waits for the next session's client
    output.reset()
    clock.now += 0.1
    output.emitted("main_video", 1)
    clock.now += 0.9

    video = output.take()["main_video"]

    assert video["frames_emitted"] == 1.0
    assert "max_emit_interval_ms" not in video


def test_emissions_from_several_threads_are_all_counted() -> None:
    output = ModelOutput()
    threads = [
        threading.Thread(target=lambda: [output.emitted("main_video", 1) for _ in range(5000)])
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert output.take()["main_video"]["frames_emitted"] == 20000.0


def _sample(taken_at: float | None, *, sent: int, received: int, **conn: float) -> PeerStats:
    return PeerStats(
        rtt_seconds=conn.get("rtt", 0.05),
        available_outgoing_bitrate_bps=2_000_000.0,
        tracks=(
            TrackStat(
                name="main_video",
                direction=TrackDirection.OUT,
                kind=TrackKind.VIDEO,
                codec="video/VP9",
                bytes_sent=sent,
                packets_sent=100,
                packets_lost=2,
                nacks=3,
                keyframe_requests=1,
                frames_sent=60,
                frames_per_second=30.0,
                frame_width=1280,
                frame_height=720,
                target_bitrate_bps=1_500_000.0,
                rtt_seconds=0.06,
                loss_ratio=0.01,
            ),
            TrackStat(
                name="webcam",
                direction=TrackDirection.IN,
                kind=TrackKind.VIDEO,
                codec="video/VP8",
                bytes_received=received,
                packets_received=80,
                packets_lost=0,
                frames_decoded=50,
                frames_dropped=1,
                jitter=0.004,
            ),
        ),
        media=OutboundMediaHealth(silence_frames=7, dropped_frames=2),
        taken_at=taken_at,
    )


def _detail(rates: TransportRates, sample: PeerStats) -> dict[str, Any]:
    return to_detail(
        observed_at_ms=1_700_000_000_000,
        model_output={"main_video": {"frames_emitted": 30.0}},
        output_kinds={"main_video": TrackKind.VIDEO},
        samples=[(ConnId(7), sample)],
        rates=rates,
    )


def test_a_reading_is_shaped_like_a_client_reading() -> None:
    rates = TransportRates()

    detail = _detail(rates, _sample(10.0, sent=1_000, received=500))

    assert detail["observed_at"] == 1_700_000_000_000
    assert detail["model_output"] == [
        {
            "track_name": "main_video",
            "kind": "video",
            # What the model sends, the client receives.
            "direction": "recvonly",
            "metrics": {"frames_emitted": 30.0},
        }
    ]
    (conn,) = detail["connection_stats"]
    assert conn["conn_id"] == 7
    assert conn["metrics"] == {
        "connection_rtt_ms": 50.0,
        "available_outgoing_bitrate_bps": 2_000_000.0,
        "silence_frames": 7.0,
        "dropped_samples": 0.0,
        "dropped_bundles": 0.0,
        "dropped_frames": 2.0,
    }
    out, inbound = conn["track_stats"]
    assert {k: out[k] for k in ("track_name", "kind", "direction", "codec")} == {
        "track_name": "main_video",
        "kind": "video",
        "direction": "recvonly",
        "codec": "VP9",
    }
    assert out["metrics"] == {
        "packets_lost": 2.0,
        "nack_count": 3.0,
        "keyframe_requests": 1.0,
        "frames_per_second": 30.0,
        "frame_width": 1280.0,
        "frame_height": 720.0,
        "packets_sent": 100.0,
        "frames_sent": 60.0,
        "target_bitrate_bps": 1_500_000.0,
        "round_trip_time_ms": 60.0,
        "loss_ratio": 0.01,
    }
    assert {k: inbound[k] for k in ("track_name", "direction", "codec")} == {
        "track_name": "webcam",
        # What the client sends, the model receives.
        "direction": "sendonly",
        "codec": "VP8",
    }
    assert inbound["metrics"]["jitter_ms"] == pytest.approx(4.0)
    assert inbound["metrics"]["frames_dropped"] == 1.0
    # A first sample has nothing to measure a rate against.
    assert "bitrate_bps" not in out["metrics"]
    assert "bitrate_bps" not in inbound["metrics"]


def test_the_bitrate_is_the_byte_difference_over_the_time_between_samples() -> None:
    rates = TransportRates()
    _detail(rates, _sample(10.0, sent=1_000, received=500))

    detail = _detail(rates, _sample(12.0, sent=251_000, received=50_500))

    out, inbound = detail["connection_stats"][0]["track_stats"]
    # Timed by when each sample was taken, not when the reading reads it.
    assert out["metrics"]["bitrate_bps"] == pytest.approx(1_000_000.0)
    assert inbound["metrics"]["bitrate_bps"] == pytest.approx(200_000.0)


def test_a_sample_the_transport_has_not_replaced_has_no_rate() -> None:
    rates = TransportRates()
    sample = _sample(10.0, sent=1_000, received=500)
    _detail(rates, sample)

    detail = _detail(rates, sample)

    out, _ = detail["connection_stats"][0]["track_stats"]
    assert "bitrate_bps" not in out["metrics"]


def test_a_connection_that_left_starts_its_rates_over() -> None:
    rates = TransportRates()
    _detail(rates, _sample(10.0, sent=1_000, received=500))
    to_detail(observed_at_ms=0, model_output={}, output_kinds={}, samples=[], rates=rates)

    detail = _detail(rates, _sample(12.0, sent=251_000, received=50_500))

    out, _ = detail["connection_stats"][0]["track_stats"]
    assert "bitrate_bps" not in out["metrics"]


def test_values_json_cannot_carry_are_left_out() -> None:
    sample = replace(_sample(None, sent=0, received=0), rtt_seconds=float("nan"))

    detail = _detail(TransportRates(), sample)

    assert "connection_rtt_ms" not in detail["connection_stats"][0]["metrics"]


def test_an_audio_track_carries_no_frame_counts() -> None:
    sample = PeerStats(
        tracks=(
            TrackStat(
                name="main_audio",
                direction=TrackDirection.OUT,
                kind=TrackKind.AUDIO,
                codec="audio/opus",
                packets_sent=600,
                frames_sent=0,
            ),
        ),
    )

    detail = _detail(TransportRates(), sample)

    (audio,) = detail["connection_stats"][0]["track_stats"]
    assert audio["codec"] == "opus"
    assert audio["metrics"]["packets_sent"] == 600.0
    assert "frames_sent" not in audio["metrics"]
