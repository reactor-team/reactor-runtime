import dataclasses
from typing import Any

import pytest
from endurance import trends
from endurance.trends import (
    Sample,
    assert_always_zero,
    assert_never_grows,
    assert_no_sustained_growth,
    assert_slope_under,
    cpu_deltas,
    exact_metrics,
    retained_cycles,
    trend_metrics,
)


def _sample(cycle: int, **overrides: Any) -> Sample:
    fields: dict[str, Any] = {
        "cycle": cycle,
        "elapsed_s": float(cycle * 10),
        "rss_bytes": 300_000_000,
        "cpu_s": float(cycle),
        "cpu_percent": 20.0,
        "num_threads": 11,
        "num_fds": 15,
        "pacers_holding_output": 0,
        "pinned_batches": 0,
        "live_connections": 0,
        "live_peers": 0,
        "live_pacers": 0,
        "live_client_stats": 0,
        "tasks": 6,
    }
    fields.update(overrides)
    return Sample(**fields)


def _samples(n: int) -> list[Sample]:
    return [_sample(i) for i in range(n)]


def test_always_zero_passes_when_every_cycle_is_zero() -> None:
    result = assert_always_zero(_samples(10), field="pacers_holding_output")
    assert result.status == "ok"


def test_always_zero_fails_on_one_nonzero_cycle_and_names_it() -> None:
    samples = _samples(10)
    samples[3] = dataclasses.replace(samples[3], pacers_holding_output=1)
    result = assert_always_zero(samples, field="pacers_holding_output")
    assert result.status == "fail"
    assert result.first_bad_cycle == 3
    assert "nonzero on 1/10 cycles" in result.detail


def test_always_zero_reports_the_peak_when_the_last_cycle_is_back_to_zero() -> None:
    samples = _samples(10)
    samples[2] = dataclasses.replace(samples[2], live_connections=2)
    result = assert_always_zero(samples, field="live_connections")
    assert result.end == 2


def test_never_grows_allows_a_flat_count_and_fails_a_rise() -> None:
    flat = _samples(10)
    assert assert_never_grows(flat, field="tasks").status == "ok"
    risen = [*flat[:-1], dataclasses.replace(flat[-1], tasks=7)]
    assert assert_never_grows(risen, field="tasks").status == "fail"


def test_no_sustained_growth_passes_a_flat_series() -> None:
    result = assert_no_sustained_growth([100.0] * 12, name="rss", max_growth_ratio=0.15)
    assert result.status == "ok"


def test_no_sustained_growth_passes_a_step_that_has_levelled_off() -> None:
    values = [100.0] * 4 + [200.0] * 8
    result = assert_no_sustained_growth(values, name="rss", max_growth_ratio=0.15)
    assert result.status == "ok"


def test_no_sustained_growth_fails_a_series_still_climbing() -> None:
    values = [100.0 * 1.2**i for i in range(12)]
    result = assert_no_sustained_growth(values, name="rss", max_growth_ratio=0.15)
    assert result.status == "fail"


def test_no_sustained_growth_treats_growth_under_the_floor_as_noise() -> None:
    values = [1.0] * 4 + [1.0] * 4 + [1.5] * 4
    result = assert_no_sustained_growth(
        values, name="cpu", max_growth_ratio=0.15, min_absolute_delta=1.0
    )
    assert result.status == "ok"


def test_no_sustained_growth_needs_enough_samples() -> None:
    with pytest.raises(AssertionError, match="needed for a trend"):
        assert_no_sustained_growth([1.0] * 5, name="rss", max_growth_ratio=0.15)


def test_cpu_deltas_are_the_time_between_samples() -> None:
    samples = [_sample(0, cpu_s=1.0), _sample(1, cpu_s=1.5), _sample(2, cpu_s=3.0)]
    assert cpu_deltas(samples) == [0.5, 1.5]


def test_trend_metrics_skip_the_warm_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trends, "ENDURANCE_WARMUP_SECONDS", 50.0)
    # A climb confined to the warm-up, then flat: judged on the flat part only.
    samples = [
        _sample(i, rss_bytes=(100_000_000 * (i + 1) if i < 5 else 600_000_000)) for i in range(20)
    ]
    rss = next(m for m in trend_metrics(samples) if m.name == "rss")
    assert rss.status == "ok"


def test_trend_metrics_refuse_a_run_too_short_to_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trends, "ENDURANCE_WARMUP_SECONDS", 120.0)
    with pytest.raises(AssertionError, match="after the 120s warm-up"):
        trend_metrics(_samples(15))


def test_exact_metrics_cover_every_count() -> None:
    names = [m.name for m in exact_metrics(_samples(3))]
    assert names == [
        "pacers_holding_output",
        "pinned_batches",
        "live_connections",
        "live_peers",
        "live_pacers",
        "live_client_stats",
        "live_participants",
        "live_step_requests",
        "live_step_decoders",
        "tasks",
        "child_processes",
    ]


def test_a_step_request_left_behind_fails_its_cycle() -> None:
    samples = _samples(4)
    samples[2] = dataclasses.replace(samples[2], live_step_requests=1)

    [requests] = [m for m in exact_metrics(samples) if m.name == "live_step_requests"]

    assert requests.status == "fail"


def test_a_step_decoder_left_alive_fails_its_cycle() -> None:
    samples = _samples(4)
    samples[2] = dataclasses.replace(samples[2], live_step_decoders=1)

    [decoders] = [m for m in exact_metrics(samples) if m.name == "live_step_decoders"]

    assert decoders.status == "fail"
    assert retained_cycles(samples) == 1


def test_a_participant_left_alive_fails_its_cycle() -> None:
    samples = _samples(4)
    samples[2] = dataclasses.replace(samples[2], live_participants=1)

    [participants] = [m for m in exact_metrics(samples) if m.name == "live_participants"]

    assert participants.status == "fail"
    assert retained_cycles(samples) == 1


def test_exact_metrics_judge_shared_memory_where_it_was_measured() -> None:
    samples = [_sample(i, child_processes=1, shm_blocks=2, shm_bytes=4_000_000) for i in range(3)]
    names = [m.name for m in exact_metrics(samples, children_baseline=1)]
    assert names[-3:] == ["child_processes", "shm_blocks", "shm_bytes"]


def test_shared_memory_is_not_judged_for_a_runtime_without_workers() -> None:
    # The blocks are counted across the machine: another run's must not
    # fail a scenario whose runtime holds none.
    samples = [_sample(i, shm_blocks=2 + i, shm_bytes=4_000_000) for i in range(3)]
    names = [m.name for m in exact_metrics(samples, children_baseline=0)]
    assert "shm_blocks" not in names


def test_a_shared_memory_block_left_each_cycle_fails() -> None:
    samples = [
        _sample(i, child_processes=1, shm_blocks=2 + i, shm_bytes=4_000_000) for i in range(3)
    ]
    results = {m.name: m for m in exact_metrics(samples, children_baseline=1)}
    assert results["shm_blocks"].status == "fail"
    assert results["shm_bytes"].status == "ok"


def test_a_child_process_past_the_baseline_fails() -> None:
    samples = [_sample(i, child_processes=2) for i in range(3)]
    results = {m.name: m for m in exact_metrics(samples, children_baseline=1)}
    assert results["child_processes"].status == "fail"


def test_child_rss_is_judged_only_when_the_runtime_has_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trends, "ENDURANCE_WARMUP_SECONDS", 0.0)
    assert "child_rss" not in [m.name for m in trend_metrics(_samples(15))]
    with_children = [_sample(i, child_rss_bytes=200_000_000) for i in range(15)]
    names = [m.name for m in trend_metrics(with_children)]
    assert "child_rss" in names
    assert "child_rss_slope" in names


def test_a_steady_worker_leak_the_ratio_passes_fails_on_slope() -> None:
    # A worker starting at 135 MB and growing 30 MB an hour over a two-hour
    # run: under the 15% ratio, but a slope over the limit.
    samples = [
        _sample(i, elapsed_s=i * 60.0, child_rss_bytes=int((135 + 0.5 * i) * 1e6))
        for i in range(120)
    ]
    results = {m.name: m for m in trend_metrics(samples)}
    assert results["child_rss"].status == "ok"
    assert results["child_rss_slope"].status == "fail"


def test_the_default_executor_growing_to_its_cap_late_is_not_a_thread_leak() -> None:
    # The pool reaches its cap of 8 an hour and a half into a two-hour run,
    # as it did on a runner: every new thread is one of its workers.
    samples = [
        _sample(
            i,
            elapsed_s=i * 60.0,
            num_threads=18 + min(8, max(0, i - 90)),
            executor_threads=1 + min(8, max(0, i - 90)),
        )
        for i in range(120)
    ]
    results = {m.name: m for m in trend_metrics(samples)}
    assert results["num_threads"].status == "ok"


def test_threads_growing_outside_the_default_executor_still_fail() -> None:
    samples = [
        _sample(i, elapsed_s=i * 60.0, num_threads=18 + min(8, max(0, i - 90)), executor_threads=1)
        for i in range(120)
    ]
    results = {m.name: m for m in trend_metrics(samples)}
    assert results["num_threads"].status == "fail"


def test_retained_cycles_count_every_cycle_with_any_nonzero_count() -> None:
    samples = _samples(5)
    samples[1] = dataclasses.replace(samples[1], pinned_batches=1)
    samples[3] = dataclasses.replace(samples[3], live_peers=1, live_pacers=1)
    assert retained_cycles(samples) == 2


def test_a_leak_just_over_the_limit_reads_as_over_it() -> None:
    values = [100.0] * 4 + [100.0] * 4 + [115.4] * 4
    result = assert_no_sustained_growth(values, name="rss", max_growth_ratio=0.15)
    assert result.status == "fail"
    assert "+15.4%" in result.detail


def _hours(n: int) -> list[float]:
    # One sample a minute.
    return [i / 60 for i in range(n)]


def test_a_steady_leak_the_ratio_check_passes_fails_on_slope() -> None:
    # 300 MB growing 1 MB a minute for two hours: under 15% from the middle
    # third to the last, but 60 MB/h.
    values = [300.0 + i for i in range(120)]
    ratio = assert_no_sustained_growth(
        values, name="rss", max_growth_ratio=0.15, min_absolute_delta=5.0
    )
    slope = assert_slope_under(
        _hours(120), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0, unit="MB"
    )
    assert ratio.status == "ok"
    assert slope.status == "fail"
    assert "+60.0 MB/h" in slope.detail


def test_a_step_that_levelled_off_in_the_first_third_reads_flat_on_slope() -> None:
    values = [300.0] * 20 + [350.0] * 100
    result = assert_slope_under(
        _hours(120), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
    )
    assert result.status == "ok"
    assert "noise floor" in result.detail


def test_a_fast_slope_with_a_rise_inside_the_noise_floor_passes() -> None:
    # Two hours climbing 24 MB/h: over the limit, but the fitted rise over the
    # last eighty minutes is under a 40 MB floor.
    values = [300.0 + 0.4 * i for i in range(120)]
    result = assert_slope_under(
        _hours(120), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=40.0
    )
    assert result.status == "ok"
    assert "noise floor" in result.detail


def test_a_climb_levelling_off_late_passes_the_slope_check() -> None:
    # A climb that slows each third, 72, then 36, then 15 MB/h: the fitted
    # slope is over the limit, but the thirds rise +36 MB, then +17 MB.
    values = (
        [200.0 + 1.2 * i for i in range(40)]
        + [248.0 + 0.6 * i for i in range(40)]
        + [272.0 + 0.25 * i for i in range(40)]
    )
    result = assert_slope_under(
        _hours(120), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
    )
    assert result.status == "ok"
    assert "levelling off" in result.detail


def test_a_leak_climbing_faster_at_the_end_fails_the_slope_check() -> None:
    values = [555.0] * 40 + [575.0] * 40 + [609.0] * 40
    result = assert_slope_under(
        _hours(120), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
    )
    assert result.status == "fail"
    assert "not levelling off" in result.detail


def test_a_step_in_the_middle_third_does_not_hide_a_tail_still_climbing() -> None:
    # A one-time step of 200 MB at the start of the middle third, then a
    # steady 60 MB/h climb: the thirds rise +219.5 MB, then +40 MB, but the
    # last third climbs as fast as the middle did.
    values = [300.0] * 40 + [500.0 + i for i in range(40)] + [540.0 + i for i in range(40)]
    result = assert_slope_under(
        _hours(120), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
    )
    assert result.status == "fail"
    assert "not levelling off" in result.detail


def test_an_early_step_does_not_hide_a_steady_leak_after_it() -> None:
    # Flat at 300 MB for 20 minutes, a one-time 60 MB step, then a steady
    # 40 MB/h leak: the thirds rise +53.2 MB, then +26.7 MB, as a plateau
    # being reached would, but the last third climbs at twice the limit.
    values = [300.0] * 20 + [360.0 + i * 40 / 60 for i in range(100)]
    result = assert_slope_under(
        _hours(120), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
    )
    assert result.status == "fail"
    assert "not levelling off" in result.detail


def test_a_leak_slowing_only_slightly_still_fails_the_slope_check() -> None:
    # Thirds rising +30 MB, then +25 MB: the noise in a steady leak, not a
    # plateau being reached.
    values = [300.0] * 40 + [330.0] * 40 + [355.0] * 40
    result = assert_slope_under(
        _hours(120), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
    )
    assert result.status == "fail"


def test_memory_swinging_over_a_few_minutes_is_not_judged() -> None:
    # A five-minute run whose RSS swings by a batch: a steep fitted line, but
    # no rate to judge yet.
    values = [380.0, 410.0, 380.0, 410.0, 410.0, 410.0, 385.0, 385.0, 413.0]
    hours = [i * 13 / 3600 for i in range(len(values))]
    result = assert_slope_under(
        hours, values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
    )
    assert result.status == "not_judged"
    assert "not judged" in result.detail


def test_an_hour_long_run_reports_its_slope_without_judging_it() -> None:
    # A steep climb over forty minutes of fit is still within what a healthy
    # runtime's swing reads as over that stretch.
    values = [300.0 + i for i in range(60)]
    result = assert_slope_under(
        _hours(60), values, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
    )
    assert result.status == "not_judged"
    assert "not judged" in result.detail


def test_the_slope_check_refuses_too_few_samples() -> None:
    with pytest.raises(AssertionError, match="needed for a trend"):
        assert_slope_under(
            [0.0] * 5, [1.0] * 5, name="rss_slope", max_per_hour=20.0, min_absolute_delta=5.0
        )


def test_a_task_the_first_cycle_left_counts_against_the_pre_run_baseline() -> None:
    samples = [_sample(i, tasks=7) for i in range(3)]
    assert assert_never_grows(samples, field="tasks").status == "ok"
    result = assert_never_grows(samples, field="tasks", baseline=6)
    assert result.status == "fail"
    assert "from 6 to 7" in result.detail


def test_a_two_hour_release_run_has_its_rss_slope_judged() -> None:
    # The release run's length with the default warm-up: if a longer warm-up
    # or a shorter minimum span made the fit fall short, the slope would stop
    # being judged on every release.
    samples = [_sample(i, elapsed_s=i * 10.0) for i in range(2 * 3600 // 10)]
    slope = next(m for m in trend_metrics(samples) if m.name == "rss_slope")
    assert slope.status == "ok"


def test_a_named_zero_check_carries_its_name() -> None:
    result = assert_always_zero([_sample(0, live_peers=1)], field="live_peers", name="peers_end")
    assert result.name == "peers_end"
    assert result.status == "fail"


def test_a_named_growth_check_carries_its_name() -> None:
    result = assert_never_grows([_sample(0, tasks=9)], field="tasks", baseline=6, name="tasks_end")
    assert result.name == "tasks_end"
    assert result.status == "fail"
