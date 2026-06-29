import time

from reactor_runtime.recording.markers import MarkerBookkeeper


def test_now_marker_advances_from_construction() -> None:
    markers = MarkerBookkeeper()
    first = markers.now_marker()
    time.sleep(0.02)
    assert markers.now_marker() > first


def test_unanchored_timeline_reports_started_immediately() -> None:
    markers = MarkerBookkeeper(anchor_at_first_frame=False)
    assert markers.recording_started is True


def test_anchored_timeline_waits_for_the_first_frame() -> None:
    markers = MarkerBookkeeper(anchor_at_first_frame=True)
    assert markers.recording_started is False
    markers.mark_first_real_frame()
    assert markers.recording_started is True
    assert markers.first_real_frame_marker is not None


def test_anchoring_resets_the_origin_to_the_first_frame() -> None:
    markers = MarkerBookkeeper(anchor_at_first_frame=True)
    time.sleep(0.05)
    markers.mark_first_real_frame()
    # The origin moved to the first frame, so the timeline is near zero again.
    assert markers.now_marker() < 0.05


def test_clip_range_is_the_tail_ending_now() -> None:
    markers = MarkerBookkeeper()
    start, end = markers.compute_clip_range(10.0)
    assert end >= start
    assert start == max(0.0, end - 10.0)


def test_clip_range_clamps_to_zero_when_short() -> None:
    markers = MarkerBookkeeper()
    start, _ = markers.compute_clip_range(10_000.0)
    assert start == 0.0


def test_recording_range_starts_at_zero() -> None:
    markers = MarkerBookkeeper()
    start, end = markers.compute_recording_range()
    assert start == 0.0
    assert end >= 0.0
