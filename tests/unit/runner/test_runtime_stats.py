from __future__ import annotations

import threading

import pytest

from reactor_runtime.runner.runtime_stats import ModelOutput


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
