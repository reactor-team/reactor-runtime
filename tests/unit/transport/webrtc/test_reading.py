from dataclasses import replace

import pytest

from reactor_runtime.core import ClientTrackDirection, TrackDirection, TrackKind
from reactor_runtime.transport.webrtc.reading import transport_reading
from reactor_runtime.transport.webrtc.stats import OutboundMediaHealth, PeerStats, TrackStat


def _sample(*, sent: int, received: int) -> PeerStats:
    return PeerStats(
        rtt_seconds=0.05,
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
    )


def test_a_sample_reads_as_named_metrics_in_the_clients_terms() -> None:
    reading = transport_reading(_sample(sent=1_000, received=500), None, None)

    assert reading.metrics == {
        "connection_rtt_ms": 50.0,
        "available_outgoing_bitrate_bps": 2_000_000.0,
        "silence_frames": 7.0,
        "dropped_samples": 0.0,
        "dropped_bundles": 0.0,
        "dropped_frames": 2.0,
    }
    out, inbound = reading.tracks
    assert (out.track_name, out.kind, out.codec) == ("main_video", TrackKind.VIDEO, "VP9")
    # What the model sends, the client receives.
    assert out.direction is ClientTrackDirection.RECVONLY
    assert out.metrics == {
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
    assert (inbound.track_name, inbound.codec) == ("webcam", "VP8")
    # What the client sends, the model receives.
    assert inbound.direction is ClientTrackDirection.SENDONLY
    assert inbound.metrics["jitter_ms"] == pytest.approx(4.0)
    assert inbound.metrics["frames_dropped"] == 1.0
    # A first sample has nothing to measure a rate against.
    assert "bitrate_bps" not in out.metrics
    assert "bitrate_bps" not in inbound.metrics


def test_the_bitrate_is_the_byte_difference_over_the_time_between_samples() -> None:
    previous = _sample(sent=1_000, received=500)

    reading = transport_reading(_sample(sent=501_000, received=100_500), previous, 2.0)

    out, inbound = reading.tracks
    assert out.metrics["bitrate_bps"] == pytest.approx(2_000_000.0)
    assert inbound.metrics["bitrate_bps"] == pytest.approx(400_000.0)


def test_a_track_new_since_the_previous_sample_has_no_rate() -> None:
    previous = PeerStats(tracks=_sample(sent=1_000, received=500).tracks[1:])

    reading = transport_reading(_sample(sent=501_000, received=100_500), previous, 2.0)

    out, inbound = reading.tracks
    assert "bitrate_bps" not in out.metrics
    assert "bitrate_bps" in inbound.metrics


def test_a_total_that_went_backwards_has_no_rate() -> None:
    previous = _sample(sent=9_000, received=500)

    reading = transport_reading(_sample(sent=1_000, received=500), previous, 2.0)

    assert "bitrate_bps" not in reading.tracks[0].metrics


def test_values_that_are_not_finite_numbers_are_left_out() -> None:
    sample = replace(_sample(sent=0, received=0), rtt_seconds=float("nan"))

    reading = transport_reading(sample, None, None)

    assert "connection_rtt_ms" not in reading.metrics


def test_a_track_without_a_codec_reads_an_empty_codec() -> None:
    sample = PeerStats(tracks=(TrackStat(name="mic", direction=TrackDirection.IN),))

    (track,) = transport_reading(sample, None, None).tracks

    assert (track.codec, track.kind) == ("", None)
