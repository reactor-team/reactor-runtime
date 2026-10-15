"""The pass/fail checks of the endurance suite.

Given the samples a run collected, each check returns a :class:`MetricResult`
rather than raising, so the report can show every signal even when one fails.
Nothing here needs a running runtime.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import statistics

from report import MetricResult

# Wall-clock rather than a cycle count, so the same scenario serves as a quick
# local check and as a long CI run.
ENDURANCE_DURATION_SECONDS = float(os.environ.get("ENDURANCE_DURATION_SECONDS", "300"))

# The trend checks skip the first part of a run, in seconds: the runtime's
# memory climbs for its first minute or more (encoders, buffers and allocator
# arenas reaching their working size) before it levels off, and a trend read
# over that climb reports a leak that is not there. The exact counts use every
# cycle.
ENDURANCE_WARMUP_SECONDS = float(os.environ.get("ENDURANCE_WARMUP_SECONDS", "120"))
# Fewer samples than this after the warm-up and a trend cannot be judged.
MIN_TREND_SAMPLES = 6


@dataclasses.dataclass
class Sample:
    """One reading of the runtime process, taken after a cycle has settled.

    The psutil fields measure the runtime's own process, never the test's. The
    rest come from the probe route the test launcher mounts.
    """

    cycle: int
    elapsed_s: float
    rss_bytes: int
    # Cumulative process CPU time; see cpu_deltas() for the per-cycle view.
    cpu_s: float
    # % of one core busy since the previous sample. Can exceed 100%: the
    # runtime runs the model loop, the pacers and libwebrtc on their own
    # threads.
    cpu_percent: float
    num_threads: int
    num_fds: int
    # Pacers that still reached model output right after they stopped, during
    # this cycle, or stopped pacers still alive and reaching it at the probe.
    # A stopped pacer must hold no output: whatever keeps the pacer alive,
    # garbage included, would keep that output alive too. Checked when the
    # pacer stops, so it does not depend on when a collection runs or on what
    # else holds the same output.
    pacers_holding_output: int
    # Output batches the cyclic garbage collector freed during this cycle:
    # output that nothing but reference-cycle garbage held.
    pinned_batches: int
    # Alive after the probe's collection, with no client connected: something
    # holds a strong reference to them.
    live_connections: int
    live_peers: int
    live_pacers: int
    # Running asyncio tasks on the runtime's loop.
    tasks: int


def cpu_deltas(samples: list[Sample]) -> list[float]:
    """Return the CPU time spent between consecutive samples.

    ``cpu_s`` is cumulative, so it always grows; the time per interval is what
    shows a cycle getting more expensive.
    """
    return [b.cpu_s - a.cpu_s for a, b in itertools.pairwise(samples)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def assert_no_sustained_growth(
    values: list[float],
    *,
    name: str,
    max_growth_ratio: float,
    min_absolute_delta: float = 0.0,
    use_median: bool = False,
    unit: str = "",
) -> MetricResult:
    """Fail when *values* is still climbing in the last third of the run.

    The verdict compares the last third with the middle third, not with the
    first: a one-time step that has levelled off by the last third, such as a
    buffer pool reaching its working size, reads as flat, while a value still
    climbing at the end fails. It fails only when the growth exceeds both
    *max_growth_ratio* and *min_absolute_delta*, so a near-zero baseline cannot
    turn a wobble into a large ratio. The reported start and end are the first
    and last thirds, the overall movement.

    Args:
        values: A non-cumulative measurement, one per sample.
        name: The metric's name in the report.
        max_growth_ratio: Largest allowed growth of the last third over the
            middle third, as a ratio.
        min_absolute_delta: Growth at or under this is noise, whatever the ratio.
        use_median: Average each third by its median, so one outlying sample
            does not move it.
        unit: The unit the report renders the values in.

    Returns:
        The metric's row, failed or not.

    Raises:
        AssertionError: Fewer than ``MIN_TREND_SAMPLES`` values.
    """
    n = len(values)
    if n < MIN_TREND_SAMPLES:
        raise AssertionError(f"only {n} {name} sample(s), {MIN_TREND_SAMPLES} needed for a trend")
    third = n // 3
    first, middle, last = values[:third], values[third : 2 * third], values[-third:]
    average = statistics.median if use_median else _mean
    first_avg, middle_avg, last_avg = average(first), average(middle), average(last)

    tail_delta = last_avg - middle_avg
    tail_ratio = (tail_delta / middle_avg) if middle_avg else (1.0 if tail_delta > 0 else 0.0)
    is_leak = tail_delta > min_absolute_delta and tail_ratio > max_growth_ratio
    overall_delta = last_avg - first_avg

    if is_leak:
        reason = (
            f"still climbing: the last third is {tail_ratio:+.0%} over the middle third, "
            f"above the {max_growth_ratio:.0%} limit"
        )
    elif tail_delta <= min_absolute_delta:
        reason = (
            f"the last third moved {tail_delta:+,.3f} from the middle third, "
            f"within the {min_absolute_delta:,.3g} noise floor"
        )
    else:
        reason = f"the last third grew {tail_ratio:+.0%}, under the {max_growth_ratio:.0%} limit"
    trend = f"{first_avg:,.3f} → {middle_avg:,.3f} → {last_avg:,.3f}"
    print(f"[{name}] {'LEAK?' if is_leak else 'ok'}: {trend}: {reason}")

    return MetricResult(
        name=name,
        start=first_avg,
        end=last_avg,
        change=overall_delta,
        unit=unit,
        status="fail" if is_leak else "ok",
        detail=reason,
        threshold=(
            f"last third > {max_growth_ratio:.0%} over the middle third "
            f"(min {min_absolute_delta:,.3g})"
        ),
        mid=middle_avg,
    )


def assert_always_zero(samples: list[Sample], *, field: str, unit: str = "count") -> MetricResult:
    """Fail when *field* is nonzero on any cycle.

    For a count that must return to exactly 0 after every cycle: an object
    retained on one cycle is a bug even if it is gone a few cycles later.
    """
    bad = [(s.cycle, getattr(s, field)) for s in samples if getattr(s, field) != 0]
    if bad:
        cycle, value = bad[0]
        peak = max(v for _, v in bad)
        reason = (
            f"nonzero on {len(bad)}/{len(samples)} cycles (first at cycle {cycle}: "
            f"{value}, peak {peak})"
        )
        print(f"[{field}] LEAK?: {reason}")
        # The peak, not the last value: a count back at 0 by the end would
        # otherwise read as 0 to 0.
        return MetricResult(
            name=field,
            start=0,
            end=peak,
            change=peak,
            unit=unit,
            status="fail",
            detail=reason,
            threshold="always 0",
            first_bad_cycle=cycle,
        )
    reason = f"stayed at exactly 0 across all {len(samples)} cycles"
    print(f"[{field}] ok: {reason}")
    return MetricResult(
        name=field,
        start=0,
        end=0,
        change=0,
        unit=unit,
        status="ok",
        detail=reason,
        threshold="always 0",
    )


def assert_never_grows(samples: list[Sample], *, field: str, unit: str = "count") -> MetricResult:
    """Fail when *field* ever rises above its first reading."""
    baseline = getattr(samples[0], field)
    peak = max(getattr(s, field) for s in samples)
    grew = peak > baseline
    reason = (
        f"grew from {baseline} to {peak} during the run"
        if grew
        else f"never rose above its first reading ({baseline})"
    )
    print(f"[{field}] {'LEAK?' if grew else 'ok'}: {reason}")
    return MetricResult(
        name=field,
        start=baseline,
        end=peak,
        change=peak - baseline,
        unit=unit,
        status="fail" if grew else "ok",
        detail=reason,
        threshold=f"never above {baseline}",
    )


def exact_metrics(samples: list[Sample]) -> list[MetricResult]:
    """Return the exact counts over every cycle: a retained object fails the cycle it shows in."""
    return [
        assert_always_zero(samples, field="pacers_holding_output"),
        assert_always_zero(samples, field="pinned_batches"),
        assert_always_zero(samples, field="live_connections"),
        assert_always_zero(samples, field="live_peers"),
        assert_always_zero(samples, field="live_pacers"),
        assert_never_grows(samples, field="tasks"),
    ]


def trend_metrics(samples: list[Sample]) -> list[MetricResult]:
    """Return the process-level trends over the samples after the warm-up.

    They catch the leaks the exact counts cannot name.

    Raises:
        AssertionError: Fewer than ``MIN_TREND_SAMPLES`` samples after the
            warm-up, too few to judge a trend.
    """
    warmed = [s for s in samples if s.elapsed_s >= ENDURANCE_WARMUP_SECONDS]
    # One more than a trend needs: the CPU check reads the gaps between samples.
    needed = MIN_TREND_SAMPLES + 1
    if len(warmed) < needed:
        raise AssertionError(
            f"only {len(warmed)} sample(s) after the {ENDURANCE_WARMUP_SECONDS:g}s warm-up, "
            f"{needed} needed to judge a trend: raise ENDURANCE_DURATION_SECONDS"
        )
    return [
        assert_no_sustained_growth(
            [s.rss_bytes / 1e6 for s in warmed],
            name="rss",
            unit="MB",
            max_growth_ratio=0.15,
            min_absolute_delta=5.0,
        ),
        assert_no_sustained_growth(
            cpu_deltas(warmed),
            name="cpu_s_per_cycle",
            unit="s",
            max_growth_ratio=0.5,
            min_absolute_delta=0.05,
        ),
        # Medians: a cycle can catch the previous one's native threads and
        # sockets still being torn down, a one-sample blip a mean would carry.
        assert_no_sustained_growth(
            [float(s.num_threads) for s in warmed],
            name="num_threads",
            unit="count",
            max_growth_ratio=0.15,
            min_absolute_delta=4,
            use_median=True,
        ),
        assert_no_sustained_growth(
            [float(s.num_fds) for s in warmed],
            name="num_fds",
            unit="count",
            max_growth_ratio=0.15,
            min_absolute_delta=3,
            use_median=True,
        ),
    ]
