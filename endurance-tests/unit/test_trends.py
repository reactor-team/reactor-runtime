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
        "tasks",
    ]


def test_a_participant_left_alive_fails_its_cycle() -> None:
    samples = _samples(4)
    samples[2] = dataclasses.replace(samples[2], live_participants=1)

    [participants] = [m for m in exact_metrics(samples) if m.name == "live_participants"]

    assert participants.status == "fail"
    assert retained_cycles(samples) == 1


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
