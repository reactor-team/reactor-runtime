from reactor_runtime.recording.markers import MarkerBookkeeper, end_segment


def test_timeline_starts_at_zero_and_unstarted() -> None:
    markers = MarkerBookkeeper()
    assert markers.now_marker() == 0.0
    assert markers.recording_started is False
    assert markers.first_real_frame_marker is None


def test_advance_moves_the_media_timeline_and_latches_started() -> None:
    markers = MarkerBookkeeper()
    markers.advance(0.5)
    markers.advance(0.25)
    assert markers.now_marker() == 0.75
    assert markers.recording_started is True
    assert markers.first_real_frame_marker == 0.0


def test_non_positive_advance_is_ignored() -> None:
    markers = MarkerBookkeeper()
    markers.advance(0.0)
    markers.advance(-1.0)
    assert markers.now_marker() == 0.0
    assert markers.recording_started is False


def test_clip_range_is_the_tail_ending_now() -> None:
    markers = MarkerBookkeeper()
    markers.advance(30.0)
    start, end = markers.compute_clip_range(10.0)
    assert end == 30.0
    assert start == 20.0


def test_clip_range_clamps_to_zero_when_short() -> None:
    markers = MarkerBookkeeper()
    markers.advance(3.0)
    start, _ = markers.compute_clip_range(10_000.0)
    assert start == 0.0


def test_recording_range_starts_at_zero() -> None:
    markers = MarkerBookkeeper()
    markers.advance(12.0)
    start, end = markers.compute_recording_range()
    assert start == 0.0
    assert end == 12.0


def test_skip_to_boundary_moves_up_to_the_next_segment() -> None:
    markers = MarkerBookkeeper()
    markers.advance(110 / 30)
    assert markers.skip_to_boundary(4) == 1
    assert markers.now_marker() == 4.0


def test_skip_to_boundary_stays_on_a_boundary_it_already_reached() -> None:
    # A timeline summed from frame durations lands a hair off the boundary; it
    # still counts as on it, rather than skipping a whole empty segment.
    markers = MarkerBookkeeper()
    for _ in range(120):
        markers.advance(1 / 30)
    assert markers.skip_to_boundary(4) == 1
    assert markers.now_marker() == 4.0


def test_end_segment_puts_a_range_ending_on_a_boundary_in_the_segment_before_it() -> None:
    # Summed frame durations land a hair past the boundary; the range still ends
    # in segment 0, the segment a seal at that position closes.
    end = 0.0
    for _ in range(60):
        end += 2 / 30
    assert end != 4.0
    assert end_segment(end, 4) == 0
    assert end_segment(4.0, 4) == 0
    assert end_segment(4.001, 4) == 1
    assert end_segment(3.667, 4) == 0
