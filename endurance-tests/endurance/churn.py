"""The connection churn loop: one session, a client joining and leaving over and over.

The scenarios differ only in the client they churn, so the loop, the checks
and the report live here once.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from endurance.harness import (
    BareClient,
    ResourceSampler,
    RuntimeProcess,
    V1Client,
    assert_answered,
    assert_full_streams,
    gray_frame,
)
from endurance.report import LiveReporter, MetricResult, finish_and_check, now_iso
from endurance.trends import (
    ENDURANCE_DURATION_SECONDS,
    ENDURANCE_SMOKE_CYCLES,
    exact_metrics,
    retained_cycles,
    trend_metrics,
)


async def run_churn(
    runtime: RuntimeProcess,
    *,
    test_name: str,
    description: str,
    client: Callable[[RuntimeProcess, str], BareClient],
) -> None:
    """Churn *client* connections on one session until the run's deadline, then check.

    Each cycle a new client from *client* joins, streams until its frames come
    back, and closes; the runtime is read once teardown has finished.
    """
    session = runtime.request("POST", "/start_session", {})[1]
    version = session["server_info"]["server_version"]
    # CI can run the suite against another release's runtime source, while the
    # package metadata still names this checkout's version.
    if source := os.environ.get("ENDURANCE_RUNTIME_REF"):
        version = f"from {source}"
    frame = gray_frame()
    sampler = ResourceSampler(runtime)
    live = LiveReporter(test_name, ENDURANCE_DURATION_SECONDS)
    started_at = now_iso()
    cycle = 0
    received: list[int] = []
    # Each v1 cycle's requests that went unanswered or were refused.
    unanswered: list[list[str]] = []
    metrics: list[MetricResult] = []

    def stream_checks() -> list[MetricResult]:
        checks = [assert_full_streams(received)]
        if unanswered:
            checks.append(assert_answered(unanswered))
        return checks

    try:
        while not sampler.deadline_reached():
            if ENDURANCE_SMOKE_CYCLES and cycle >= ENDURANCE_SMOKE_CYCLES:
                break
            connection = client(runtime, session["session_id"])
            received.append(await connection.run(frame))
            if isinstance(connection, V1Client):
                unanswered.append(connection.unanswered)
            reading, _ = runtime.settled_probe()
            sampler.sample(cycle=cycle, reading=reading)
            live.update(sampler.samples, retained=retained_cycles(sampler.samples))
            cycle += 1
        if not sampler.samples:
            raise AssertionError(
                f"no cycle completed in {ENDURANCE_DURATION_SECONDS:g}s: "
                "raise ENDURANCE_DURATION_SECONDS"
            )
        if ENDURANCE_SMOKE_CYCLES:
            metrics = stream_checks()
        else:
            metrics = [
                *stream_checks(),
                *exact_metrics(sampler.samples, tasks_baseline=sampler.initial_tasks),
            ]
            metrics += trend_metrics(sampler.samples)
    finally:
        # A run cut short still reports the cycles it finished.
        if not metrics and sampler.samples:
            metrics = [
                *stream_checks(),
                *exact_metrics(sampler.samples, tasks_baseline=sampler.initial_tasks),
            ]
        finish_and_check(
            test_name=test_name,
            subject="reactor-runtime",
            description=description,
            version=version,
            duration_s=ENDURANCE_DURATION_SECONDS,
            started_at=started_at,
            sampler=sampler,
            live=live,
            metrics=metrics,
            iterations=cycle,
        )
