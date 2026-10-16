from endurance.harness import FRAMES_PER_CYCLE, assert_full_streams


def test_full_streams_pass_when_every_cycle_got_its_frames() -> None:
    assert assert_full_streams([FRAMES_PER_CYCLE] * 5).status == "ok"


def test_a_cycle_one_frame_short_fails_the_run() -> None:
    result = assert_full_streams([FRAMES_PER_CYCLE, FRAMES_PER_CYCLE - 1, FRAMES_PER_CYCLE])
    assert result.status == "fail"
    assert result.first_bad_cycle == 1
    assert "1/3 cycles" in result.detail


def test_a_cycle_with_no_frames_fails_the_run() -> None:
    assert assert_full_streams([0]).status == "fail"
