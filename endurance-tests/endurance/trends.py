"""The pass/fail checks of the endurance suite.

Given the samples a run collected, each check returns a :class:`MetricResult`
rather than raising, so the report can show every signal even when one fails.
The one exception is a run too short to judge: the trend checks raise
``AssertionError`` when they get fewer than ``MIN_TREND_SAMPLES`` samples, and
the run is reported as ERROR. A new check follows the same contract. Nothing
here needs a running runtime.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import statistics

from endurance.report import MetricResult

# Wall-clock rather than a cycle count, so the same scenario serves as a quick
# local check and as a long CI run.
ENDURANCE_DURATION_SECONDS = float(os.environ.get("ENDURANCE_DURATION_SECONDS", "300"))
# A smoke run, as PR CI does: stop after this many cycles and judge only that
# every cycle streamed in full, which proves the launcher's hooks, the probe,
# the signaling and the control messages still work. The leak checks need a
# real run, so a smoke run never fails on them. 0 runs to the deadline.
ENDURANCE_SMOKE_CYCLES = int(os.environ.get("ENDURANCE_SMOKE_CYCLES", "0"))

# The trend checks skip the first part of a run, in seconds: the runtime's
# memory climbs for its first minute or more (encoders, buffers and allocator
# arenas reaching their working size) before it levels off, and a trend read
# over that climb reports a leak that is not there. The exact counts use every
# cycle.
ENDURANCE_WARMUP_SECONDS = float(os.environ.get("ENDURANCE_WARMUP_SECONDS", "120"))
# Fewer samples than this after the warm-up and a trend cannot be judged.
MIN_TREND_SAMPLES = 6
# Largest RSS growth rate allowed over the end of a run, in MB per hour. An
# absolute rate stays as strict on a two-hour run as on a short one, where a
# limit relative to the run's own length loosens as the run grows.
MAX_RSS_MB_PER_HOUR = 20.0
# Shortest stretch of samples, in hours, a slope is judged over. RSS swings by
# about a batch as the allocator takes and returns memory, and a line fitted
# through that swing reads as a climb or a fall that shrinks as the stretch
# grows: on a healthy two-hour run, the steepest climb was 85 MB/h over 30
# minutes, 18 MB/h over 60 and 9 MB/h over 70. So a slope is judged only over
# 75 minutes or more, which a run of about two hours reaches: the release
# runs are judged, shorter manual runs only report the rate.
MIN_SLOPE_SPAN_HOURS = 1.25


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
    # Connections the runtime still keeps client-stats state for, with no
    # client connected.
    live_client_stats: int
    # Running asyncio tasks on the runtime's loop.
    tasks: int
    # Room participant connections alive after the probe's collection, with
    # no participant in the room.
    live_participants: int = 0
    # Processes the runtime started, at any depth: a model's worker ranks and
    # the helpers they bring.
    child_processes: int = 0
    # Those processes' resident memory, summed.
    child_rss_bytes: int = 0
    # Shared-memory blocks the distributed package holds in /dev/shm, and
    # their bytes; None on a platform with no /dev/shm.
    shm_blocks: int | None = None
    shm_bytes: int | None = None


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
    # Three equal windows, the last one ending at the final sample: when n is
    # not a multiple of three, the one or two samples between the middle and
    # last windows are left out, so every window weighs the same.
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
            f"still climbing: the last third is {tail_ratio:+.1%} over the middle third, "
            f"above the {max_growth_ratio:.0%} limit"
        )
    elif tail_delta <= min_absolute_delta:
        reason = (
            f"the last third moved {tail_delta:+,.3f} from the middle third, "
            f"within the {min_absolute_delta:,.3g} noise floor"
        )
    else:
        reason = f"the last third grew {tail_ratio:+.1%}, under the {max_growth_ratio:.0%} limit"
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


def assert_slope_under(
    hours: list[float],
    values: list[float],
    *,
    name: str,
    max_per_hour: float,
    min_absolute_delta: float,
    unit: str = "",
) -> MetricResult:
    """Fail when *values* climbs faster than *max_per_hour* over the end of the run.

    Fits a least-squares line over the last two thirds of the samples, the
    windows the ratio check compares, so a one-time step that levelled off
    before them reads as flat. It fails only when the slope is over the limit
    and the line also rises more than *min_absolute_delta* across those
    samples, so on a short run a few MB of noise cannot read as a fast rate.
    A fit over less than :data:`MIN_SLOPE_SPAN_HOURS` is reported as not
    judged and never fails: memory that swings up and down by about a batch
    fits a steep line over anything shorter.

    Args:
        hours: When each sample was taken, in hours.
        values: A non-cumulative measurement, one per sample.
        name: The metric's name in the report.
        max_per_hour: Largest allowed slope, in *unit* per hour.
        min_absolute_delta: A fitted rise at or under this is noise, whatever
            the slope.
        unit: The unit the report renders the values in.

    Returns:
        The metric's row, failed or not.

    Raises:
        AssertionError: Fewer than ``MIN_TREND_SAMPLES`` values.
    """
    n = len(values)
    if n < MIN_TREND_SAMPLES:
        raise AssertionError(f"only {n} {name} sample(s), {MIN_TREND_SAMPLES} needed for a trend")
    # Everything after the first third: the middle and last windows.
    tail = n - n // 3
    xs, ys = hours[-tail:], values[-tail:]
    slope = statistics.linear_regression(xs, ys).slope if xs[-1] > xs[0] else 0.0
    span = xs[-1] - xs[0]
    rise = slope * span
    judged = span >= MIN_SLOPE_SPAN_HOURS
    is_leak = judged and slope > max_per_hour and rise > min_absolute_delta
    if not judged:
        reason = (
            f"{slope:+,.1f} {unit}/h over {span * 60:.0f} min, not judged: "
            f"a rate needs {MIN_SLOPE_SPAN_HOURS * 60:.0f} min of samples"
        )
    elif is_leak:
        reason = (
            f"climbing {slope:+,.1f} {unit}/h over the end of the run, "
            f"above the {max_per_hour:g} {unit}/h limit"
        )
    elif rise <= min_absolute_delta:
        reason = (
            f"{slope:+,.1f} {unit}/h, a fitted rise of {rise:+,.1f} {unit} "
            f"within the {min_absolute_delta:g} {unit} noise floor"
        )
    else:
        reason = f"{slope:+,.1f} {unit}/h, under the {max_per_hour:g} {unit}/h limit"
    status = "fail" if is_leak else ("ok" if judged else "not_judged")
    print(f"[{name}] {'LEAK?' if is_leak else status.replace('_', ' ')}: {reason}")
    return MetricResult(
        name=name,
        start=ys[0],
        end=ys[-1],
        change=rise,
        unit=unit,
        status=status,
        detail=reason,
        threshold=f"slope > {max_per_hour:g} {unit}/h (min rise {min_absolute_delta:g} {unit})",
    )


def assert_always_zero(
    samples: list[Sample], *, field: str, unit: str = "count", name: str | None = None
) -> MetricResult:
    """Fail when *field* is nonzero on any cycle.

    For a count that must return to exactly 0 after every cycle: an object
    retained on one cycle is a bug even if it is gone a few cycles later. The
    result is named *name*, by default *field*.
    """
    label = name or field
    bad = [(s.cycle, getattr(s, field)) for s in samples if getattr(s, field) != 0]
    if bad:
        cycle, value = bad[0]
        peak = max(v for _, v in bad)
        reason = (
            f"nonzero on {len(bad)}/{len(samples)} cycles (first at cycle {cycle}: "
            f"{value}, peak {peak})"
        )
        print(f"[{label}] LEAK?: {reason}")
        # The peak, not the last value: a count back at 0 by the end would
        # otherwise read as 0 to 0.
        return MetricResult(
            name=label,
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
    print(f"[{label}] ok: {reason}")
    return MetricResult(
        name=label,
        start=0,
        end=0,
        change=0,
        unit=unit,
        status="ok",
        detail=reason,
        threshold="always 0",
    )


def assert_never_grows(
    samples: list[Sample],
    *,
    field: str,
    baseline: float | None = None,
    unit: str = "count",
    name: str | None = None,
) -> MetricResult:
    """Fail when *field* ever rises above *baseline*, by default its first reading.

    Pass the reading from before the first cycle as *baseline* when there is
    one, so what the first cycle leaves behind counts too. The result is named
    *name*, by default *field*.
    """
    label = name or field
    if baseline is None:
        baseline = getattr(samples[0], field)
    peak = max(getattr(s, field) for s in samples)
    grew = peak > baseline
    reason = (
        f"grew from {baseline} to {peak} during the run"
        if grew
        else f"never rose above its first reading ({baseline})"
    )
    print(f"[{label}] {'LEAK?' if grew else 'ok'}: {reason}")
    return MetricResult(
        name=label,
        start=baseline,
        end=peak,
        change=peak - baseline,
        unit=unit,
        status="fail" if grew else "ok",
        detail=reason,
        threshold=f"never above {baseline}",
    )


# The counts that must be 0 after every cycle.
ZERO_FIELDS = (
    "pacers_holding_output",
    "pinned_batches",
    "live_connections",
    "live_peers",
    "live_pacers",
    "live_client_stats",
    "live_participants",
)


def retained_cycles(samples: list[Sample]) -> int:
    """Return how many cycles left something behind, by any count that must stay at 0."""
    return sum(1 for s in samples if any(getattr(s, field) for field in ZERO_FIELDS))


def process_metrics(
    samples: list[Sample], *, children_baseline: int | None = None
) -> list[MetricResult]:
    """Return the counts of what the runtime holds outside its own process.

    The child processes must never outnumber *children_baseline*, the count
    before the first cycle when known. The shared-memory blocks and bytes must
    never rise above the first reading: a block grows to its high-water mark
    on first use and is then reused. They are judged only for a runtime with
    worker processes, and only where they were measured.
    """
    results = [assert_never_grows(samples, field="child_processes", baseline=children_baseline)]
    if not children_baseline:
        # The blocks are counted across the machine: a runtime with no worker
        # processes holds none, and another run on the machine must not fail it.
        print("[shm_blocks] not judged: the runtime has no worker processes")
        return results
    if any(s.shm_blocks is None or s.shm_bytes is None for s in samples):
        print("[shm_blocks] not measured: this platform has no /dev/shm")
        return results
    return [
        *results,
        assert_never_grows(samples, field="shm_blocks"),
        assert_never_grows(samples, field="shm_bytes", unit="bytes"),
    ]


def exact_metrics(
    samples: list[Sample],
    *,
    tasks_baseline: int | None = None,
    children_baseline: int | None = None,
) -> list[MetricResult]:
    """Return the exact counts over every cycle: a retained object fails the cycle it shows in.

    *tasks_baseline* and *children_baseline* are the task and child process
    counts before the first cycle, when known.
    """
    return [
        *(assert_always_zero(samples, field=field) for field in ZERO_FIELDS),
        assert_never_grows(samples, field="tasks", baseline=tasks_baseline),
        *process_metrics(samples, children_baseline=children_baseline),
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
        assert_slope_under(
            [s.elapsed_s / 3600 for s in warmed],
            [s.rss_bytes / 1e6 for s in warmed],
            name="rss_slope",
            unit="MB",
            max_per_hour=MAX_RSS_MB_PER_HOUR,
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
        # Judged only for a runtime whose model runs in processes of its own,
        # by the same pair of checks as the runtime's own RSS.
        *(
            [
                assert_no_sustained_growth(
                    [s.child_rss_bytes / 1e6 for s in warmed],
                    name="child_rss",
                    unit="MB",
                    max_growth_ratio=0.15,
                    min_absolute_delta=5.0,
                ),
                assert_slope_under(
                    [s.elapsed_s / 3600 for s in warmed],
                    [s.child_rss_bytes / 1e6 for s in warmed],
                    name="child_rss_slope",
                    unit="MB",
                    max_per_hour=MAX_RSS_MB_PER_HOUR,
                    min_absolute_delta=5.0,
                ),
            ]
            if any(s.child_rss_bytes for s in warmed)
            else []
        ),
    ]
