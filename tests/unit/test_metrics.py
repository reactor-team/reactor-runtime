from __future__ import annotations

import pytest
from prometheus_client import Counter

from reactor_runtime.core import EndReason, SessionEvent, SessionState, TrackDirection
from reactor_runtime.metrics import (
    CONTENT_TYPE,
    MetricsRecorder,
    RuntimeMetrics,
    WebRtcMetrics,
)
from reactor_runtime.runner.state_machine import SessionStateMachine
from reactor_runtime.transport.webrtc.stats import OutboundMediaHealth, PeerStats, TrackStat


def test_publishes_the_identity_of_the_process() -> None:
    metrics = RuntimeMetrics(version="1.4.2", model="pipeline:Brightness")

    rendered = metrics.render().decode()

    assert 'runtime_info{model="pipeline:Brightness",version="1.4.2"} 1.0' in rendered


def test_renders_in_the_prometheus_text_format() -> None:
    metrics = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")

    assert CONTENT_TYPE.startswith("text/plain")
    assert metrics.render().startswith(b"# HELP runtime_info")


def test_each_holder_owns_its_own_registry() -> None:
    # Two holders in one process must not see each other's numbers. The registry
    # is a field of the holder rather than a module global, so a component that
    # observes on one leaves the other empty. Every test in this suite relies on
    # that: they build a holder each and read it back.
    first = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")
    second = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")

    Counter("runtime_test_total", "A counter one holder carries.", registry=first.registry).inc()

    assert "runtime_test_total" in first.render().decode()
    assert "runtime_test_total" not in second.render().decode()


def test_the_registry_carries_only_the_runtime_instruments() -> None:
    # A dedicated registry, not the default one: the process reports what the
    # runtime declares, and the scraper adds what it knows about the machine.
    rendered = RuntimeMetrics(version="0.0.0", model="pipeline:Echo").render().decode()

    assert "python_gc_objects_collected_total" not in rendered
    assert "process_virtual_memory_bytes" not in rendered


class _Clock:
    """A clock the test moves by hand, so a measured duration is exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class _Session:
    """A state machine with a recorder on it, plus readers for the registry."""

    def __init__(self) -> None:
        self.clock = _Clock()
        self.metrics = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")
        self.machine = SessionStateMachine()
        self.recorder = MetricsRecorder(
            self.metrics, state=self.machine.current_state, clock=self.clock
        )
        self.machine.on_transition(self.recorder.observe)

    def send(self, event: SessionEvent, **detail: object) -> None:
        assert self.machine.send(event, **detail), f"{event.name} was rejected"

    def open_session(self) -> None:
        """Drive the machine from a fresh process to a session waiting for a client."""
        self.send(SessionEvent.INITIALIZATION_SUCCESS)
        self.send(SessionEvent.START_SESSION)

    def value(self, name: str, **labels: str) -> float | None:
        return self.metrics.registry.get_sample_value(name, labels or None)

    def rendered(self) -> str:
        return self.metrics.render().decode()


def test_publishes_the_state_of_a_session_that_has_not_moved() -> None:
    session = _Session()

    # Every state has a series from the first scrape, so a query never has to
    # tell "no session yet" apart from "no metric yet".
    assert session.value("runtime_session_state", state="created") == 1.0
    assert session.value("runtime_session_state", state="ready") == 0.0
    assert session.value("runtime_session_state", state="terminated") == 0.0


def test_raises_one_state_series_at_a_time() -> None:
    session = _Session()

    session.open_session()

    assert session.value("runtime_session_state", state="waiting") == 1.0
    assert session.value("runtime_session_state", state="ready") == 0.0
    assert session.value("runtime_session_state", state="created") == 0.0


def test_a_journal_fact_leaves_the_state_alone() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.CHUNK_READY, recording_id="r", idx=1)

    assert session.value("runtime_session_state", state="waiting") == 1.0


def test_counts_a_stopped_session_and_how_long_it_ran() -> None:
    session = _Session()
    session.open_session()

    session.clock.tick(90.0)
    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    assert session.value("runtime_sessions_total", reason="stopped") == 1.0
    assert session.value("runtime_session_duration_seconds_count", reason="stopped") == 1.0
    assert session.value("runtime_session_duration_seconds_sum", reason="stopped") == 90.0
    assert session.value("runtime_session_state", state="ready") == 1.0


def test_carries_the_reason_a_session_ended() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.TIMEOUT, reason=EndReason.TIMED_OUT)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.TIMED_OUT)

    assert session.value("runtime_sessions_total", reason="timed_out") == 1.0
    assert session.value("runtime_sessions_total", reason="stopped") == 0.0


def test_measures_the_wait_for_the_first_client() -> None:
    session = _Session()
    session.open_session()

    session.clock.tick(2.5)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.clock.tick(4.0)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1003)

    # The second client is not the first one: the measurement is how long the
    # session sat empty, so it happens once per session.
    assert session.value("runtime_session_time_to_first_client_seconds_count") == 1.0
    assert session.value("runtime_session_time_to_first_client_seconds_sum") == 2.5


def test_measures_how_long_a_teardown_took() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.clock.tick(0.4)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    assert session.value("runtime_session_teardown_seconds_count") == 1.0
    assert session.value("runtime_session_teardown_seconds_sum") == 0.4


def test_counts_each_client_that_arrives_and_leaves() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1003)
    session.send(SessionEvent.CONNECTION_CLOSED, conn_id=1003)

    assert session.value("runtime_connections_opened_total") == 2.0
    assert session.value("runtime_connections_closed_total") == 1.0
    assert session.value("runtime_connections_active") == 1.0
    assert session.value("runtime_session_state", state="streaming") == 1.0


def test_the_last_client_to_leave_orphans_the_session() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.CONNECTION_CLOSED, conn_id=1002)

    assert session.value("runtime_connections_active") == 0.0
    assert session.value("runtime_session_state", state="orphaned") == 1.0


def test_teardown_clears_the_clients_it_closed_wholesale() -> None:
    session = _Session()
    session.open_session()
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1003)

    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    # Teardown closes every wire at once and reports no per-connection loss, so
    # the gauge would hold two dead clients if it only followed the close facts.
    assert session.value("runtime_connections_active") == 0.0
    # Neither client closed itself, so the two counters differ by what teardown
    # reaped. A reader subtracts them to get that number.
    assert session.value("runtime_connections_opened_total") == 2.0
    assert session.value("runtime_connections_closed_total") == 0.0


def test_counts_a_session_a_crash_evicted() -> None:
    session = _Session()
    session.open_session()

    session.clock.tick(12.0)
    session.send(SessionEvent.EVICTION, reason=EndReason.ERROR, error="the model crashed")

    assert session.value("runtime_sessions_total", reason="error") == 1.0
    assert session.value("runtime_session_duration_seconds_sum", reason="error") == 12.0
    assert session.value("runtime_session_state", state="terminated") == 1.0
    assert session.value("runtime_connections_active") == 0.0


def test_an_eviction_with_no_session_counts_nothing() -> None:
    session = _Session()
    session.send(SessionEvent.INITIALIZATION_SUCCESS)

    session.send(SessionEvent.EVICTION, reason=EndReason.EVICTED)

    # The model sat idle, so no session ran and none ended.
    assert session.value("runtime_sessions_total", reason="evicted") == 0.0
    assert session.value("runtime_session_duration_seconds_count", reason="evicted") is None


def test_a_model_that_fails_to_load_counts_no_session() -> None:
    session = _Session()

    session.send(SessionEvent.INITIALIZATION_FAIL)

    assert session.value("runtime_session_state", state="terminated") == 1.0
    for reason in EndReason:
        assert session.value("runtime_sessions_total", reason=reason.value) == 0.0


def test_counts_errors_and_keeps_the_message_out_of_the_labels() -> None:
    session = _Session()
    session.open_session()

    session.send(SessionEvent.ERROR, message="command 'set_mode' references an unresolved upload")
    session.send(SessionEvent.ERROR, message="command 'set_seed' rejected (seed: out of range)")

    assert session.value("runtime_session_errors_total") == 2.0
    # A message is unbounded text. One in a label would cost a series for every
    # distinct error the runtime ever writes.
    assert "set_mode" not in session.rendered()


def test_a_second_session_measures_itself_from_its_own_start() -> None:
    session = _Session()
    session.open_session()
    session.clock.tick(30.0)
    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    session.send(SessionEvent.START_SESSION)
    session.clock.tick(5.0)
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)

    assert session.value("runtime_sessions_total", reason="stopped") == 2.0
    assert session.value("runtime_session_duration_seconds_sum", reason="stopped") == 35.0
    assert session.value("runtime_session_time_to_first_client_seconds_sum") == 5.0
    assert session.value("runtime_session_state", state="ready") == 1.0


def test_the_machine_state_and_the_gauge_agree_after_a_full_round() -> None:
    session = _Session()
    session.open_session()
    session.send(SessionEvent.CONNECTION_OPENED, conn_id=1002)
    session.send(SessionEvent.CONNECTION_CLOSED, conn_id=1002)
    session.send(SessionEvent.TIMEOUT, reason=EndReason.TIMED_OUT)
    session.send(SessionEvent.CLEANUP_COMPLETE, reason=EndReason.TIMED_OUT)

    assert session.machine.current_state is SessionState.READY
    for member in SessionState:
        expected = 1.0 if member is SessionState.READY else 0.0
        assert session.value("runtime_session_state", state=member.name.lower()) == expected


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
