"""The reports of an endurance run.

A run ends as one :class:`RunResult`, written by :func:`write_reports` as a
JSON dump, a Markdown report and a plain-text summary, all rendered from that
one result so the three always agree. While a run is going,
:class:`LiveReporter` prints a short status block every so often.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# How often the live status block prints: often enough that a CI log opened
# mid-run shows something recent, rarely enough that an hours-long run stays
# readable.
LIVE_INTERVAL_SECONDS = float(os.environ.get("ENDURANCE_LIVE_INTERVAL_SECONDS", "30"))

DEFAULT_OUTPUT_DIR = Path("endurance-results")


def now_iso() -> str:
    """Return the current UTC time in ISO 8601."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def git_commit_sha() -> str | None:
    """Return the commit under test, or None when it cannot be told."""
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def format_duration(seconds: float) -> str:
    """Return *seconds* as ``1h 05m``, ``4m 09s`` or ``42s``."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _format_value(value: float, unit: str) -> str:
    if unit == "MB":
        return f"{value:,.1f} MB"
    if unit == "%":
        return f"{value:,.1f}%"
    if unit == "count":
        return f"{value:,.0f}"
    if unit == "s":
        return f"{value:,.3f}s"
    return f"{value:,.3f}"


def _format_delta(value: float, unit: str) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{_format_value(value, unit)}"


@dataclasses.dataclass
class MetricResult:
    """One row of the report: how one signal moved over the run, and its verdict."""

    name: str
    start: float
    end: float
    change: float
    unit: str  # "MB" | "%" | "count" | "s" | ""
    status: str  # "ok" | "fail"
    detail: str
    threshold: str | None = None
    # The first cycle a count was off, for the checks that track one.
    first_bad_cycle: int | None = None
    # The middle third, for a trend check: its verdict is the move from here to
    # the end, so the table shows it rather than leaving it to be inferred.
    mid: float | None = None

    @property
    def emoji(self) -> str:
        """The status as a colored dot."""
        return "🟢" if self.status == "ok" else "🔴"

    @property
    def status_label(self) -> str:
        """The status as a word."""
        return "Stable" if self.status == "ok" else "FAILED"

    @property
    def mid_change(self) -> float | None:
        """The move from the middle third to the end, which a trend check judges."""
        return None if self.mid is None else self.end - self.mid

    def _cells(self) -> tuple[str, str, str, str]:
        mid = _format_value(self.mid, self.unit) if self.mid is not None else "—"
        move = self.mid_change
        delta = _format_delta(move, self.unit) if move is not None else "—"
        return _format_value(self.start, self.unit), mid, _format_value(self.end, self.unit), delta

    def row_markdown(self) -> str:
        """Return the row of the Markdown results table."""
        start, mid, end, delta = self._cells()
        status = f"{self.emoji} {self.status_label}"
        return f"| {self.name} | {start} | {mid} | {end} | {delta} | {status} |"

    def row_text(self) -> str:
        """Return the row of the plain-text results."""
        start, mid, end, delta = self._cells()
        return (
            f"  {self.name:<22} {start:>12} -> {mid:>12} -> {end:<12} {delta:>10}  "
            f"{self.emoji} {self.status_label}"
        )


@dataclasses.dataclass
class RunResult:
    """Everything a run's reports are rendered from."""

    test_name: str
    # What was tested, and its version: "reactor-runtime" and "3.8.3".
    subject: str
    status: str  # "PASS" | "FAIL" | "ERROR"
    duration_s: float
    elapsed_s: float
    iterations: int
    started_at: str
    ended_at: str
    metrics: list[MetricResult]
    # One line on what the scenario does, shown under the title.
    description: str | None = None
    # Cycles that went wrong without stopping the run; None when the scenario
    # does not count any.
    errors: int | None = None
    version: str | None = None
    commit_sha: str | None = None
    # Why the run stopped short, when it did.
    run_error: str | None = None
    # Five evenly spaced readings for the timeline; filled from the samples.
    checkpoints: list[dict[str, Any]] | None = None
    samples: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    @property
    def emoji(self) -> str:
        """The status as a colored dot."""
        return {"PASS": "🟢", "FAIL": "🔴", "ERROR": "🟠"}.get(self.status, "⚪")

    @property
    def tested(self) -> str:
        """What was tested, with its version when known: ``reactor-runtime 3.8.3``."""
        return f"{self.subject} {self.version}" if self.version else self.subject


def compute_checkpoints(samples: list[dict[str, Any]], *, n: int = 5) -> list[dict[str, Any]]:
    """Return *n* samples evenly spaced across the run, for the timeline.

    The results table compares thirds of the run, which cannot tell a value
    that stepped up once and held from one that climbed throughout; the
    timeline shows the shape.
    """
    if not samples:
        return []
    last_idx = len(samples) - 1
    checkpoints = []
    for i in range(n):
        pct = (i / (n - 1)) if n > 1 else 0.0
        s = samples[round(pct * last_idx)]
        checkpoints.append(
            {
                "pct": round(pct * 100),
                "cycle": s.get("cycle"),
                "elapsed_s": s.get("elapsed_s"),
                "rss_mb": s["rss_bytes"] / 1e6 if "rss_bytes" in s else None,
                "num_threads": s.get("num_threads"),
                "num_fds": s.get("num_fds"),
                "cpu_percent": s.get("cpu_percent"),
            }
        )
    return checkpoints


def _timeline_rows(checkpoints: list[dict[str, Any]]) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for c in checkpoints:
        at = f"{c['pct']}%"
        if c.get("elapsed_s") is not None:
            at += f" ({format_duration(c['elapsed_s'])})"
        rss = f"{c['rss_mb']:,.1f} MB" if c.get("rss_mb") is not None else "—"
        threads = str(c["num_threads"]) if c.get("num_threads") is not None else "—"
        fds = str(c["num_fds"]) if c.get("num_fds") is not None else "—"
        cpu = f"{c['cpu_percent']:.1f}%" if c.get("cpu_percent") is not None else "—"
        rows.append((at, rss, threads, fds, cpu))
    return rows


def _conclusion_lines(result: RunResult) -> list[str]:
    if result.status == "ERROR":
        return [
            f"The run did not complete: {result.run_error or 'an unexpected error'}.",
            "No check failed before it stopped, so this points at the test or its "
            "setup rather than at a leak. See the job log.",
        ]

    if result.status == "PASS":
        tested = ", ".join(m.name for m in result.metrics)
        lines = [
            f"{result.tested} ran {format_duration(result.elapsed_s)} of endurance testing "
            f"across {result.iterations:,} cycles with no retained objects and no sustained "
            f"growth ({tested})."
        ]
        for m in result.metrics:
            if m.unit == "count" and m.change == 0:
                lines.append(f"{m.name} stayed at {_format_value(m.end, m.unit)}.")
            else:
                lines.append(
                    f"{m.name} moved {_format_delta(m.change, m.unit)} "
                    f"({_format_value(m.start, m.unit)} → {_format_value(m.end, m.unit)}) "
                    "and held steady."
                )
        if result.errors is not None:
            lines.append(
                "Every cycle worked."
                if not result.errors
                else f"{result.errors} cycle(s) went wrong without affecting the result."
            )
        return lines

    failed = [m for m in result.metrics if m.status == "fail"]
    lines = [f"The run found a problem in: {', '.join(m.name for m in failed)}."]
    for m in failed:
        if m.threshold == "always 0":
            # How often and since when says more than a start and an end.
            lines.append(f"{m.name}: {m.detail}.")
        else:
            lines.append(
                f"{m.name} went from {_format_value(m.start, m.unit)} to "
                f"{_format_value(m.end, m.unit)} ({_format_delta(m.change, m.unit)}): {m.detail}."
            )
    if result.run_error:
        lines.append(f"The run then stopped short: {result.run_error}.")
    return lines


def _checkpoints_for(result: RunResult) -> list[dict[str, Any]]:
    if result.checkpoints is not None:
        return result.checkpoints
    return compute_checkpoints(result.samples)


def render_markdown(result: RunResult) -> str:
    """Return the Markdown report, as a CI job summary shows it."""
    lines = [f"# {result.emoji} Endurance Test Report — {result.test_name}", ""]
    if result.description:
        lines += [result.description, ""]
    lines += [
        f"{result.tested} · {format_duration(result.elapsed_s)} · "
        f"{result.iterations:,} cycles · **{result.status}**",
        "",
    ]
    if result.commit_sha:
        lines.append(f"- **Commit:** `{result.commit_sha[:12]}`")
    lines += [
        f"- **Started:** {result.started_at}",
        f"- **Ended:** {result.ended_at}",
        f"- **Target duration:** {format_duration(result.duration_s)}",
        "",
    ]

    if result.metrics:
        lines += [
            "## Results",
            "",
            "| Metric | Start | Mid | End | Δ (Mid→End) | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        lines += [m.row_markdown() for m in result.metrics]
        if result.errors is not None:
            mark = "🟢" if result.errors == 0 else "🟡"
            lines.append(f"| Errors | {result.errors} | — | {result.errors} | — | {mark} |")
        lines.append("")

    if checkpoints := _checkpoints_for(result):
        lines += [
            "## Timeline",
            "",
            "| At | RSS | Threads | FDs | CPU% |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {at} | {rss} | {threads} | {fds} | {cpu} |"
            for at, rss, threads, fds, cpu in _timeline_rows(checkpoints)
        ]
        lines.append("")

    lines += ["## Conclusion", "", f"{result.emoji} {result.status}", ""]
    lines += _conclusion_lines(result)
    lines += ["", "_Every sample is in the JSON report, uploaded with the run's artifacts._"]
    return "\n".join(lines).rstrip() + "\n"


def render_text(result: RunResult) -> str:
    """Return the plain-text summary."""
    lines = [f"Endurance Test Report — {result.test_name}"]
    if result.description:
        lines.append(result.description)
    lines += [result.tested, result.status, f"Duration: {format_duration(result.elapsed_s)}", ""]
    if result.commit_sha:
        lines.append(f"Commit:          {result.commit_sha[:12]}")
    lines += [
        f"Started:         {result.started_at}",
        f"Ended:           {result.ended_at}",
        f"Target duration: {format_duration(result.duration_s)}",
        f"Cycles:          {result.iterations:,}",
        "",
    ]

    if result.metrics:
        lines += ["Results", "-------"]
        lines += [m.row_text() for m in result.metrics]
        if result.errors is not None:
            mark = "🟢" if result.errors == 0 else "🟡"
            lines.append(f"  {'Errors':<22} {result.errors:>12}  {mark}")
        lines.append("")

    if checkpoints := _checkpoints_for(result):
        lines += ["Timeline", "--------"]
        lines += [
            f"  {at:<14} RSS {rss:>10}  threads {threads:>4}  fds {fds:>4}  cpu {cpu:>6}"
            for at, rss, threads, fds, cpu in _timeline_rows(checkpoints)
        ]
        lines.append("")

    lines += ["Conclusion", "----------", result.status, ""]
    lines += _conclusion_lines(result)
    return "\n".join(lines).rstrip() + "\n"


def write_reports(result: RunResult, out_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Write the JSON, Markdown and text reports of *result* under *out_dir*.

    Returns:
        The path of each report, keyed ``json``, ``markdown`` and ``text``.
    """
    if result.checkpoints is None and result.samples:
        result.checkpoints = compute_checkpoints(result.samples)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / f"{result.test_name}.json",
        "markdown": out_dir / f"{result.test_name}-report.md",
        "text": out_dir / f"{result.test_name}-summary.txt",
    }
    paths["json"].write_text(json.dumps(dataclasses.asdict(result), indent=2, default=str))
    paths["markdown"].write_text(render_markdown(result))
    paths["text"].write_text(render_text(result))
    return paths


def finish_run(
    *,
    test_name: str,
    subject: str,
    duration_s: float,
    started_at: str,
    samples: list[Any],
    metrics: list[MetricResult],
    iterations: int,
    errors: int | None = None,
    version: str | None = None,
    description: str | None = None,
    out_dir: Path = DEFAULT_OUTPUT_DIR,
) -> RunResult:
    """Build the run's result and write its reports.

    Call it from the scenario's ``finally`` block: an exception being handled
    there is what marks the run as stopped short. A failed check outranks that,
    so the status is FAIL when any check failed, ERROR when the run stopped
    short with none failed, and PASS otherwise. A failure to write the reports
    is printed rather than raised, so it never hides the test's own failure.
    """
    exc_type, exc_val, _ = sys.exc_info()
    run_error = f"{exc_type.__name__}: {exc_val}" if exc_type is not None else None
    failed = any(m.status == "fail" for m in metrics)
    status = "FAIL" if failed else ("ERROR" if run_error else "PASS")
    result = RunResult(
        test_name=test_name,
        subject=subject,
        status=status,
        duration_s=duration_s,
        elapsed_s=samples[-1].elapsed_s if samples else 0.0,
        iterations=iterations,
        started_at=started_at,
        ended_at=now_iso(),
        metrics=metrics,
        errors=errors,
        version=version,
        description=description,
        commit_sha=git_commit_sha(),
        run_error=run_error,
        samples=[dataclasses.asdict(s) for s in samples],
    )
    try:
        write_reports(result, out_dir)
    # Any failure here is printed, not raised: the test's own failure must surface.
    except Exception as e:
        print(f"[finish_run] writing the reports failed: {e!r}")
    return result


def finish_and_check(
    *,
    test_name: str,
    subject: str,
    description: str,
    version: str | None,
    duration_s: float,
    started_at: str,
    sampler: Any,
    live: LiveReporter,
    metrics: list[MetricResult],
    iterations: int,
    errors: int | None = None,
    out_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """End a scenario: print the last status, write the reports, and fail on a failed check.

    The last call in a scenario's ``finally`` block. *sampler* is the scenario's
    sampler, read for its ``samples``.

    Raises:
        AssertionError: A check failed.
    """
    if sampler.samples:
        live.update(sampler.samples, errors=errors, force=True)
    result = finish_run(
        test_name=test_name,
        subject=subject,
        duration_s=duration_s,
        started_at=started_at,
        samples=sampler.samples,
        metrics=metrics,
        iterations=iterations,
        errors=errors,
        version=version,
        description=description,
        out_dir=out_dir,
    )
    if result.status == "FAIL":
        names = ", ".join(m.name for m in metrics if m.status == "fail")
        raise AssertionError(f"the endurance run found a problem in: {names}; see the report")


class LiveReporter:
    """Print a short status block every ``LIVE_INTERVAL_SECONDS`` while a run goes."""

    def __init__(
        self, test_name: str, duration_s: float, *, interval_s: float | None = None
    ) -> None:
        """Report on *test_name*, a run of *duration_s* seconds."""
        self.test_name = test_name
        self.duration_s = duration_s
        self.interval_s = LIVE_INTERVAL_SECONDS if interval_s is None else interval_s
        self._last_print: float | None = None

    def update(self, samples: list[Any], *, errors: int | None = None, force: bool = False) -> None:
        """Print the block once the interval has passed since the last one, or when *force*."""
        if not samples:
            return
        now = time.monotonic()
        if not force and self._last_print is not None and now - self._last_print < self.interval_s:
            return
        self._last_print = now
        print(self._format(samples, errors=errors))

    def _format(self, samples: list[Any], *, errors: int | None) -> str:
        first, last = samples[0], samples[-1]
        elapsed = last.elapsed_s
        pct = min(100, int(elapsed / self.duration_s * 100)) if self.duration_s else 0
        rss_start_mb, rss_now_mb = first.rss_bytes / 1e6, last.rss_bytes / 1e6
        avg_cpu = sum(s.cpu_percent for s in samples) / len(samples)
        bar = "─" * 46
        lines = [
            bar,
            f"Endurance Test — {self.test_name}",
            bar,
            "",
            f"Duration       {format_duration(self.duration_s)}",
            f"Elapsed        {format_duration(elapsed)}",
            f"Progress       {pct}%",
            "",
            f"Cycles         {last.cycle + 1:,}",
        ]
        if errors is not None:
            lines.append(f"Errors         {errors}")
        lines += [
            "",
            "Resources",
            f"  RSS          {rss_start_mb:,.0f} MB → {rss_now_mb:,.0f} MB   "
            f"({_format_delta(rss_now_mb - rss_start_mb, 'MB')})",
            f"  CPU          avg {avg_cpu:.1f}%",
            f"  Threads      {first.num_threads} → {last.num_threads}",
            f"  File desc.   {first.num_fds} → {last.num_fds}",
            "",
            f"Status         {'🟡 Errors' if errors else '🟢 Healthy'}",
            bar,
        ]
        return "\n".join(lines)
