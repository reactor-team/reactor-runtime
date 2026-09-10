from __future__ import annotations

import pytest

from reactor_runtime.core import TrackDirection
from reactor_runtime.metrics import RuntimeMetrics, WebRtcMetrics
from reactor_runtime.transport.webrtc.stats import OutboundMediaHealth, PeerStats, TrackStat


class _Transport:
    """A WebRTC metrics group with one connection's recorder on it."""

    def __init__(self, *, outbound: tuple[str, ...] = (), inbound: tuple[str, ...] = ()) -> None:
        self.metrics = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")
        self.group = WebRtcMetrics(self.metrics)
        self.recorder = self.group.sampler(
            outbound=outbound, inbound=inbound, allowed_tracks=(*outbound, *inbound)
        )

    def observe(
        self,
        *,
        rtt: float | None = None,
        bandwidth_bps: float | None = None,
        tracks: tuple[TrackStat, ...] = (),
        media: OutboundMediaHealth | None = None,
    ) -> None:
        self.recorder.observe(
            PeerStats(
                rtt_seconds=rtt,
                available_outgoing_bitrate_bps=bandwidth_bps,
                tracks=tracks,
                media=media or OutboundMediaHealth(),
            )
        )

    def value(self, name: str, **labels: str) -> float | None:
        return self.metrics.registry.get_sample_value(name, labels or None)


def _outbound(
    name: str,
    *,
    packets_sent: int | None = None,
    packets_lost: int | None = None,
    retransmitted_packets_sent: int | None = None,
    bytes_sent: int | None = None,
    frames_sent: int | None = None,
    rtt_seconds: float | None = None,
    loss_ratio: float | None = None,
    nacks: int | None = None,
    keyframe_requests: int | None = None,
) -> TrackStat:
    return TrackStat(
        name=name,
        direction=TrackDirection.OUT,
        packets_sent=packets_sent,
        packets_lost=packets_lost,
        retransmitted_packets_sent=retransmitted_packets_sent,
        bytes_sent=bytes_sent,
        frames_sent=frames_sent,
        rtt_seconds=rtt_seconds,
        loss_ratio=loss_ratio,
        nacks=nacks,
        keyframe_requests=keyframe_requests,
    )


def _inbound(
    name: str,
    *,
    packets_received: int | None = None,
    packets_lost: int | None = None,
    bytes_received: int | None = None,
    frames_decoded: int | None = None,
    frames_dropped: int | None = None,
    nacks: int | None = None,
    keyframe_requests: int | None = None,
    jitter: float | None = None,
) -> TrackStat:
    return TrackStat(
        name=name,
        direction=TrackDirection.IN,
        packets_received=packets_received,
        packets_lost=packets_lost,
        bytes_received=bytes_received,
        frames_decoded=frames_decoded,
        frames_dropped=frames_dropped,
        nacks=nacks,
        keyframe_requests=keyframe_requests,
        jitter=jitter,
    )


def test_seeds_a_child_for_every_track_the_connection_carries() -> None:
    # A track that stayed silent has to read zero rather than be absent, which is
    # what tells it apart from a track the model never declared.
    transport = _Transport(outbound=("main_video",), inbound=("mic",))

    assert transport.value("runtime_webrtc_packets_sent_total", track="main_video") == 0.0
    assert transport.value("runtime_webrtc_bytes_sent_total", track="main_video") == 0.0
    assert transport.value("runtime_webrtc_frames_sent_total", track="main_video") == 0.0
    assert transport.value("runtime_webrtc_packets_received_total", track="mic") == 0.0
    assert transport.value("runtime_webrtc_jitter_seconds_count", track="mic") == 0.0
    # The feedback counters are seeded per direction, like the loss they precede.
    assert transport.value("runtime_webrtc_nacks_total", track="main_video", direction="out") == 0.0
    assert transport.value("runtime_webrtc_nacks_total", track="mic", direction="in") == 0.0
    assert (
        transport.value(
            "runtime_webrtc_keyframe_requests_total", track="main_video", direction="out"
        )
        == 0.0
    )
    # Loss is seeded once per direction, because one instrument carries both and
    # the two are only told apart by the way the track flowed.
    assert (
        transport.value("runtime_webrtc_packets_lost_total", track="main_video", direction="out")
        == 0.0
    )
    assert transport.value("runtime_webrtc_packets_lost_total", track="mic", direction="in") == 0.0
    # A track this connection does not carry is absent, not zero.
    assert transport.value("runtime_webrtc_packets_sent_total", track="overlay") is None


def test_counts_the_whole_of_the_first_sample_it_sees() -> None:
    transport = _Transport(outbound=("main_video",))

    transport.observe(tracks=(_outbound("main_video", packets_sent=1200),))

    assert transport.value("runtime_webrtc_packets_sent_total", track="main_video") == 1200.0


def test_moves_a_counter_by_the_window_rather_than_by_the_total() -> None:
    # The peer reports a running total. Feeding it straight to a counter would
    # count every packet again on every sample, and a rate over it would climb
    # with the age of the connection instead of describing the last window.
    transport = _Transport(outbound=("main_video",))

    transport.observe(tracks=(_outbound("main_video", packets_sent=1200),))
    transport.observe(tracks=(_outbound("main_video", packets_sent=1500),))
    transport.observe(tracks=(_outbound("main_video", packets_sent=1800),))

    assert transport.value("runtime_webrtc_packets_sent_total", track="main_video") == 1800.0


def test_reads_a_total_that_went_backwards_as_a_fresh_start() -> None:
    transport = _Transport(outbound=("main_video",))

    transport.observe(tracks=(_outbound("main_video", packets_sent=1200),))
    transport.observe(tracks=(_outbound("main_video", packets_sent=40),))

    # The 40 count as new rather than as a negative increment, which a counter
    # would refuse outright.
    assert transport.value("runtime_webrtc_packets_sent_total", track="main_video") == 1240.0


def test_keeps_each_track_on_its_own_baseline() -> None:
    transport = _Transport(outbound=("main_video", "overlay"))

    transport.observe(
        tracks=(
            _outbound("main_video", packets_sent=1000),
            _outbound("overlay", packets_sent=10),
        )
    )
    transport.observe(
        tracks=(
            _outbound("main_video", packets_sent=1400),
            _outbound("overlay", packets_sent=25),
        )
    )

    assert transport.value("runtime_webrtc_packets_sent_total", track="main_video") == 1400.0
    assert transport.value("runtime_webrtc_packets_sent_total", track="overlay") == 25.0


def test_records_the_loss_and_the_jitter_of_an_inbound_track() -> None:
    transport = _Transport(inbound=("mic",))

    transport.observe(tracks=(_inbound("mic", packets_lost=12, jitter=0.004),))
    transport.observe(tracks=(_inbound("mic", packets_lost=30, jitter=0.06),))

    assert transport.value("runtime_webrtc_packets_lost_total", track="mic", direction="in") == 30.0
    assert transport.value("runtime_webrtc_jitter_seconds_count", track="mic") == 2.0
    assert transport.value("runtime_webrtc_jitter_seconds_sum", track="mic") == 0.064


def test_tells_the_loss_of_the_two_directions_apart() -> None:
    # Outbound loss is what the receiver reported back about what this side sent,
    # and inbound loss is what this side counted itself. They share an instrument
    # and are only separated by the direction the track flowed.
    transport = _Transport(outbound=("main_video",), inbound=("webcam",))

    transport.observe(
        tracks=(
            _outbound("main_video", packets_lost=17),
            _inbound("webcam", packets_lost=3),
        )
    )

    assert (
        transport.value("runtime_webrtc_packets_lost_total", track="main_video", direction="out")
        == 17.0
    )
    assert (
        transport.value("runtime_webrtc_packets_lost_total", track="webcam", direction="in") == 3.0
    )


def test_carries_the_volume_a_track_actually_moved() -> None:
    # The rate of these is the bitrate and the frame rate that reached the wire,
    # which is what a viewer saw rather than what the encoder was aiming at.
    transport = _Transport(outbound=("main_video",))

    transport.observe(
        tracks=(_outbound("main_video", bytes_sent=90_000, frames_sent=30, packets_sent=120),)
    )
    transport.observe(
        tracks=(_outbound("main_video", bytes_sent=180_000, frames_sent=61, packets_sent=245),)
    )

    assert transport.value("runtime_webrtc_bytes_sent_total", track="main_video") == 180_000.0
    assert transport.value("runtime_webrtc_frames_sent_total", track="main_video") == 61.0
    assert transport.value("runtime_webrtc_packets_sent_total", track="main_video") == 245.0


def test_carries_what_an_inbound_track_delivered_and_the_decoder_lost() -> None:
    transport = _Transport(inbound=("webcam",))

    transport.observe(
        tracks=(
            _inbound(
                "webcam",
                packets_received=400,
                bytes_received=52_000,
                frames_decoded=118,
                frames_dropped=6,
            ),
        )
    )

    assert transport.value("runtime_webrtc_packets_received_total", track="webcam") == 400.0
    assert transport.value("runtime_webrtc_bytes_received_total", track="webcam") == 52_000.0
    assert transport.value("runtime_webrtc_frames_decoded_total", track="webcam") == 118.0
    assert transport.value("runtime_webrtc_frames_dropped_total", track="webcam") == 6.0


def test_counts_the_retransmissions_a_lossy_path_cost() -> None:
    transport = _Transport(outbound=("main_video",))

    transport.observe(tracks=(_outbound("main_video", retransmitted_packets_sent=14),))

    assert transport.value("runtime_webrtc_packets_retransmitted_total", track="main_video") == 14.0


def test_measures_what_congestion_control_thinks_the_path_will_carry() -> None:
    # Reported in bits per second and held in bytes, which is the base unit every
    # other size in the registry uses.
    transport = _Transport()

    transport.observe(bandwidth_bps=2_400_000.0)

    assert transport.value("runtime_webrtc_bandwidth_estimate_bytes_per_second_count") == 1.0
    assert transport.value("runtime_webrtc_bandwidth_estimate_bytes_per_second_sum") == 300_000.0


def test_records_no_estimate_before_the_engine_has_one() -> None:
    transport = _Transport()

    transport.observe(rtt=0.02)

    assert transport.value("runtime_webrtc_bandwidth_estimate_bytes_per_second_count") == 0.0


def test_measures_the_round_trip_of_a_live_wire() -> None:
    transport = _Transport()

    transport.observe(rtt=0.032)
    transport.observe(rtt=0.048)

    assert transport.value("runtime_webrtc_rtt_seconds_count") == 2.0
    assert transport.value("runtime_webrtc_rtt_seconds_sum") == 0.08


def test_records_nothing_for_a_reading_the_peer_could_not_take() -> None:
    # An absent reading is not a reading of zero. A sample with nothing in it
    # must leave the histogram empty rather than pull its average to the floor.
    transport = _Transport(outbound=("main_video",), inbound=("mic",))

    transport.observe(tracks=(_outbound("main_video"), _inbound("mic")))

    assert transport.value("runtime_webrtc_rtt_seconds_count") == 0.0
    assert transport.value("runtime_webrtc_jitter_seconds_count", track="mic") == 0.0
    assert transport.value("runtime_webrtc_packets_sent_total", track="main_video") == 0.0


def test_counts_the_outbound_media_the_runtime_discarded_or_invented() -> None:
    # None of this appears in transport statistics: it is what the runtime
    # dropped or manufactured on its own side, before anything reached the wire.
    transport = _Transport()

    transport.observe(
        media=OutboundMediaHealth(
            silence_frames=10, dropped_samples=480, dropped_bundles=2, dropped_frames=7
        )
    )
    transport.observe(
        media=OutboundMediaHealth(
            silence_frames=14, dropped_samples=960, dropped_bundles=2, dropped_frames=9
        )
    )

    assert transport.value("runtime_media_silence_frames_total") == 14.0
    assert transport.value("runtime_media_dropped_samples_total") == 960.0
    assert transport.value("runtime_media_dropped_bundles_total") == 2.0
    assert transport.value("runtime_media_dropped_frames_total") == 9.0


def test_each_connection_differences_against_its_own_previous_sample() -> None:
    # A second connection starts its own baseline, so the totals its peer
    # reports are not read as a jump on top of the first connection's.
    transport = _Transport(outbound=("main_video",))
    transport.observe(tracks=(_outbound("main_video", packets_sent=5000),))

    second = transport.group.sampler(outbound=("main_video",), allowed_tracks=("main_video",))
    second.observe(PeerStats(tracks=(_outbound("main_video", packets_sent=300),)))

    assert transport.value("runtime_webrtc_packets_sent_total", track="main_video") == 5300.0


def test_measures_the_round_trip_media_actually_took() -> None:
    # The receiver measures this one on the stream itself. It is not the round
    # trip of the ICE checks, which keep succeeding on a path whose media queue
    # has grown, so the two are recorded apart.
    transport = _Transport(outbound=("main_video",))

    transport.observe(rtt=0.02, tracks=(_outbound("main_video", rtt_seconds=0.31),))

    assert transport.value("runtime_webrtc_rtt_seconds_sum") == 0.02
    assert transport.value("runtime_webrtc_media_rtt_seconds_count", track="main_video") == 1.0
    assert transport.value("runtime_webrtc_media_rtt_seconds_sum", track="main_video") == 0.31


def test_measures_the_share_of_a_track_the_receiver_never_got() -> None:
    transport = _Transport(outbound=("main_video",))

    transport.observe(tracks=(_outbound("main_video", loss_ratio=0.04),))
    transport.observe(tracks=(_outbound("main_video", loss_ratio=0.12),))

    assert transport.value("runtime_webrtc_loss_ratio_count", track="main_video") == 2.0
    assert transport.value("runtime_webrtc_loss_ratio_sum", track="main_video") == 0.16
    # A ratio is a level, so the buckets are what a query reads: one sample sat
    # under two percent and the other did not.
    assert transport.value("runtime_webrtc_loss_ratio_bucket", track="main_video", le="0.05") == 1.0


def test_records_a_clean_path_as_a_zero_rather_than_as_nothing() -> None:
    # Zero loss is a reading. Dropping it would leave the histogram holding only
    # the bad windows and every quantile reading as though loss were constant.
    transport = _Transport(outbound=("main_video",))

    transport.observe(tracks=(_outbound("main_video", loss_ratio=0.0),))

    assert transport.value("runtime_webrtc_loss_ratio_count", track="main_video") == 1.0
    assert transport.value("runtime_webrtc_loss_ratio_sum", track="main_video") == 0.0


def test_records_neither_before_the_receiver_has_reported() -> None:
    transport = _Transport(outbound=("main_video",))

    transport.observe(tracks=(_outbound("main_video", packets_sent=100),))

    assert transport.value("runtime_webrtc_media_rtt_seconds_count", track="main_video") == 0.0
    assert transport.value("runtime_webrtc_loss_ratio_count", track="main_video") == 0.0


def test_counts_the_repairs_asked_for_in_either_direction() -> None:
    # Repair traffic turns bad before the picture does: a retransmission that
    # arrives in time hides the loss that prompted it, so the requests rise while
    # the stream still plays. Both ends ask, and the direction says which did.
    transport = _Transport(outbound=("main_video",), inbound=("webcam",))

    transport.observe(tracks=(_outbound("main_video", nacks=17), _inbound("webcam", nacks=6)))
    transport.observe(tracks=(_outbound("main_video", nacks=44), _inbound("webcam", nacks=41)))

    assert (
        transport.value("runtime_webrtc_nacks_total", track="main_video", direction="out") == 44.0
    )
    assert transport.value("runtime_webrtc_nacks_total", track="webcam", direction="in") == 41.0


def test_counts_requests_and_repairs_apart() -> None:
    # A retransmission goes out in answer to a request, so the two track each
    # other while a path is merely lossy. They diverge when requests start going
    # unanswered, which is the reading that says repair is no longer keeping up —
    # and one number could not show it.
    transport = _Transport(outbound=("main_video",))

    transport.observe(tracks=(_outbound("main_video", nacks=50, retransmitted_packets_sent=31),))

    assert (
        transport.value("runtime_webrtc_nacks_total", track="main_video", direction="out") == 50.0
    )
    assert transport.value("runtime_webrtc_packets_retransmitted_total", track="main_video") == 31.0


def test_counts_the_keyframes_a_decoder_had_to_ask_for() -> None:
    # The level past a retransmission request: the decoder cannot continue at all
    # and the stream has to restart. Answering one costs a whole keyframe.
    transport = _Transport(outbound=("main_video",), inbound=("webcam",))

    transport.observe(
        tracks=(
            _outbound("main_video", keyframe_requests=3),
            _inbound("webcam", keyframe_requests=1),
        )
    )

    assert (
        transport.value(
            "runtime_webrtc_keyframe_requests_total", track="main_video", direction="out"
        )
        == 3.0
    )
    assert (
        transport.value("runtime_webrtc_keyframe_requests_total", track="webcam", direction="in")
        == 1.0
    )


@pytest.mark.parametrize("field", ["packets_lost", "nacks", "keyframe_requests"])
def test_same_name_in_both_directions_has_independent_baselines(field: str) -> None:
    transport = _Transport(outbound=("shared",), inbound=("shared",))
    for sent, received in [(10, 30), (15, 37), (2, 40)]:
        transport.observe(
            tracks=(
                TrackStat(name="shared", direction=TrackDirection.OUT, **{field: sent}),
                TrackStat(name="shared", direction=TrackDirection.IN, **{field: received}),
            )
        )
    metric = f"runtime_webrtc_{field}_total"
    assert transport.value(metric, track="shared", direction="out") == 17
    assert transport.value(metric, track="shared", direction="in") == 40


def test_unknown_tracks_keep_registry_bounded_across_connections() -> None:
    transport = _Transport()
    counts = []
    for index in range(20):
        name = f"client_{index}"
        recorder = transport.group.sampler(outbound=(name,), inbound=(name,))
        recorder.observe(
            PeerStats(
                tracks=(
                    _outbound(name, packets_sent=10, loss_ratio=0.1, rtt_seconds=0.1),
                    _inbound(name, packets_received=20, jitter=0.1),
                )
            )
        )
        counts.append(sum(len(metric.samples) for metric in transport.metrics.registry.collect()))
    assert len(set(counts)) == 1
    assert transport.value("runtime_webrtc_packets_sent_total", track="unknown") == 200
    assert transport.value("runtime_webrtc_packets_received_total", track="unknown") == 400


def test_unknown_tracks_aggregate_with_independent_baselines() -> None:
    transport = _Transport()
    for first, second in [(10, 30), (15, 37)]:
        transport.observe(
            tracks=(
                _outbound("first", packets_sent=first),
                _outbound("second", packets_sent=second),
            )
        )
    assert transport.value("runtime_webrtc_packets_sent_total", track="unknown") == 52
