import json
from pathlib import Path

import pytest
from endurance.report import (
    LiveReporter,
    MetricResult,
    RunResult,
    finish_run,
    git_commit_sha,
    render_markdown,
    write_reports,
)
from endurance.trends import Sample


def _metric(name: str, status: str, detail: str = "") -> MetricResult:
    return MetricResult(
        name=name,
        start=0,
        end=1 if status == "fail" else 0,
        change=1 if status == "fail" else 0,
        unit="count",
        status=status,
        detail=detail,
        threshold="always 0",
        first_bad_cycle=0 if status == "fail" else None,
    )


def _samples() -> list[Sample]:
    return [
        Sample(
            cycle=i,
            elapsed_s=float(i),
            rss_bytes=300_000_000,
            cpu_s=float(i),
            cpu_percent=20.0,
            num_threads=11,
            num_fds=15,
            pacers_holding_output=0,
            pinned_batches=0,
            live_connections=0,
            live_peers=0,
            live_pacers=0,
            live_client_stats=0,
            tasks=6,
        )
        for i in range(3)
    ]


def _finish(tmp_path: Path, metrics: list[MetricResult]) -> RunResult:
    return finish_run(
        test_name="churn",
        subject="reactor-runtime",
        duration_s=3,
        started_at="2026-10-15T00:00:00+00:00",
        samples=_samples(),
        metrics=metrics,
        iterations=3,
        version="3.8.3",
        out_dir=tmp_path,
    )


def test_a_run_with_every_check_ok_passes(tmp_path: Path) -> None:
    assert _finish(tmp_path, [_metric("live_pacers", "ok")]).status == "PASS"


def test_a_failed_check_fails_the_run(tmp_path: Path) -> None:
    assert _finish(tmp_path, [_metric("live_pacers", "fail")]).status == "FAIL"


def test_a_run_that_stopped_short_with_no_failed_check_is_an_error(tmp_path: Path) -> None:
    try:
        raise AssertionError("too short")
    except AssertionError:
        result = _finish(tmp_path, [_metric("live_pacers", "ok")])
    assert result.status == "ERROR"
    assert result.run_error == "AssertionError: too short"


def test_a_failed_check_outranks_a_run_that_stopped_short(tmp_path: Path) -> None:
    try:
        raise AssertionError("too short")
    except AssertionError:
        result = _finish(tmp_path, [_metric("live_pacers", "fail")])
    assert result.status == "FAIL"
    assert "stopped short" in render_markdown(result)


def test_a_failed_count_is_reported_by_how_often_and_since_when(tmp_path: Path) -> None:
    detail = "nonzero on 14/14 cycles (first at cycle 0: 1, peak 1)"
    result = _finish(tmp_path, [_metric("pacers_holding_output", "fail", detail)])
    assert f"pacers_holding_output: {detail}." in render_markdown(result)


def test_the_reports_are_written_from_one_result(tmp_path: Path) -> None:
    result = _finish(tmp_path, [_metric("live_pacers", "ok")])
    paths = write_reports(result, tmp_path)
    data = json.loads(paths["json"].read_text())
    assert data["status"] == "PASS"
    assert len(data["samples"]) == 3
    assert "reactor-runtime 3.8.3" in paths["markdown"].read_text()
    assert "reactor-runtime 3.8.3" in paths["text"].read_text()


def test_the_live_status_says_when_a_cycle_left_something_behind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    live = LiveReporter("churn", 60)
    live.update(_samples(), retained=2, force=True)
    assert "2 cycle(s) left something behind" in capsys.readouterr().out
    live.update(_samples(), retained=0, force=True)
    assert "Nothing left behind so far" in capsys.readouterr().out


def test_a_report_that_cannot_be_written_fails_a_passing_run(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    with pytest.raises(FileExistsError):
        _finish(blocker, [_metric("live_pacers", "ok")])


def test_a_report_that_cannot_be_written_does_not_hide_a_failed_check(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    assert _finish(blocker, [_metric("live_pacers", "fail")]).status == "FAIL"


def test_the_commit_under_test_is_the_swapped_in_runtime_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("ENDURANCE_COMMIT_SHA", "b" * 40)
    assert git_commit_sha() == "b" * 40
    monkeypatch.delenv("ENDURANCE_COMMIT_SHA")
    assert git_commit_sha() == "a" * 40


def test_a_metric_that_was_not_judged_reads_apart_from_a_stable_one() -> None:
    result = MetricResult(
        name="rss_slope", start=0, end=0, change=0, unit="MB", status="not_judged", detail=""
    )
    assert (result.emoji, result.status_label) == ("⚪", "Not judged")
