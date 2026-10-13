import math
import threading
from typing import Any

from reactor_runtime.core import FrameStage, TrackDirection, TrackKind
from reactor_runtime.transport.webrtc.frame_stages import STAGES, FrameStageWindow
from reactor_runtime.transport.webrtc.stats import PeerStats, SenderTiming, TrackStat


def _out(**fields: Any) -> TrackStat:
    return TrackStat(
        name="main_video", direction=TrackDirection.OUT, kind=TrackKind.VIDEO, **fields
    )


def _in(**fields: Any) -> TrackStat:
    return TrackStat(name="webcam", direction=TrackDirection.IN, kind=TrackKind.VIDEO, **fields)


def _timing(rtp: int) -> SenderTiming:
    return SenderTiming(rtp_timestamp=rtp, encode_wait_ms=1.0, packetize_ms=0.0, pacer_ms=3.0)


def test_added_frames_sum_per_track_and_stage() -> None:
    window = FrameStageWindow()
    window.add("main_video", "output_pacing", 30.0)
    window.add("main_video", "output_pacing", 50.0)
    window.add("main_video", "output_queue", 0.5)
    window.add("webcam", "delivery", 0.8)

    assert window.take() == {
        "main_video": (
            FrameStage("output_pacing", total_ms=80.0, frames=2),
            FrameStage("output_queue", total_ms=0.5, frames=1),
        ),
        "webcam": (FrameStage("delivery", total_ms=0.8, frames=1),),
    }


def test_a_take_starts_a_new_window() -> None:
    window = FrameStageWindow()
    window.add("main_video", "output_queue", 1.0)
    window.take()

    assert window.take() == {}


def test_what_is_no_measurement_is_left_out() -> None:
    window = FrameStageWindow()
    window.add("main_video", "output_queue", -1.0)
    window.add("main_video", "output_queue", math.nan)
    window.add("main_video", "output_queue", 1.0, frames=0)

    assert window.take() == {}


def test_stages_come_out_in_trip_order_and_an_unknown_one_last() -> None:
    window = FrameStageWindow()
    for stage in ("custom", *reversed(STAGES)):
        window.add("t", stage, 1.0)

    assert [s.name for s in window.take()["t"]] == [*STAGES, "custom"]


def test_fold_adds_what_libwebrtc_timed_between_two_samples() -> None:
    window = FrameStageWindow()
    first = PeerStats(
        tracks=(
            _out(frames_encoded=100, encode_seconds=0.2),
            _in(
                frames_decoded=90,
                decode_seconds=0.09,
                jitter_buffer_frames=90,
                jitter_buffer_seconds=1.0,
            ),
        )
    )
    second = PeerStats(
        tracks=(
            _out(frames_encoded=160, encode_seconds=0.32),
            _in(
                frames_decoded=150,
                decode_seconds=0.15,
                jitter_buffer_frames=148,
                jitter_buffer_seconds=1.725,
            ),
        )
    )

    window.fold(first, None)
    assert window.take() == {}, "a first sample has nothing to compare against"
    window.fold(second, first)
    stages = window.take()

    [encode] = stages["main_video"]
    assert (encode.name, encode.frames) == ("encode", 60)
    assert math.isclose(encode.total_ms, 120.0)
    jitter_buffer, decode = stages["webcam"]
    assert (jitter_buffer.name, jitter_buffer.frames) == ("jitter_buffer", 58)
    assert math.isclose(jitter_buffer.total_ms, 725.0)
    assert (decode.name, decode.frames) == ("decode", 60)
    assert math.isclose(decode.total_ms, 60.0)


def test_fold_skips_a_total_that_went_backwards() -> None:
    window = FrameStageWindow()
    before = PeerStats(tracks=(_out(frames_encoded=100, encode_seconds=0.2),))
    restarted = PeerStats(tracks=(_out(frames_encoded=10, encode_seconds=0.02),))

    window.fold(restarted, before)

    assert window.take() == {}


def test_fold_counts_each_timing_frame_once() -> None:
    window = FrameStageWindow()
    first = PeerStats(tracks=(_in(timing_frame=_timing(9_000)),))
    repeated = PeerStats(tracks=(_in(timing_frame=_timing(9_000)),))
    next_one = PeerStats(tracks=(_in(timing_frame=_timing(12_000)),))

    window.fold(first, None)
    window.fold(repeated, first)
    window.fold(next_one, repeated)

    assert window.take() == {
        "webcam": (
            FrameStage("encode_wait", total_ms=2.0, frames=2),
            FrameStage("packetize", total_ms=0.0, frames=2),
            FrameStage("pacer", total_ms=6.0, frames=2),
        )
    }


def test_frames_timed_from_several_threads_all_count() -> None:
    window = FrameStageWindow()

    def time_frames() -> None:
        for _ in range(1_000):
            window.add("main_video", "output_queue", 1.0)

    threads = [threading.Thread(target=time_frames) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert window.take() == {
        "main_video": (FrameStage("output_queue", total_ms=4_000.0, frames=4_000),)
    }
