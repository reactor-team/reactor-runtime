"""connection_churn: one session, a client connecting and leaving over and over."""

from __future__ import annotations

import os

from endurance.harness import (
    BareClient,
    ResourceSampler,
    RuntimeProcess,
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

TEST_NAME = "connection-churn"
DESCRIPTION = (
    "One long-lived session. Each cycle a client joins, streams a webcam both ways "
    "until 60 frames come back, and closes; the runtime is then read once teardown "
    "has finished."
)


async def test_connection_churn(runtime: RuntimeProcess) -> None:
    """Release everything a connection held once it has closed, every time."""
    session = runtime.request("POST", "/start_session", {})[1]
    version = session["server_info"]["server_version"]
    # CI can run the suite against another release's runtime source, while the
    # package metadata still names this checkout's version.
    if source := os.environ.get("ENDURANCE_RUNTIME_REF"):
        version = f"from {source}"
    frame = gray_frame()
    sampler = ResourceSampler(runtime)
    live = LiveReporter(TEST_NAME, ENDURANCE_DURATION_SECONDS)
    started_at = now_iso()
    cycle = 0
    received: list[int] = []
    metrics: list[MetricResult] = []
    try:
        while not sampler.deadline_reached():
            if ENDURANCE_SMOKE_CYCLES and cycle >= ENDURANCE_SMOKE_CYCLES:
                break
            received.append(await BareClient(runtime, session["session_id"]).run(frame))
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
            metrics = [assert_full_streams(received)]
        else:
            metrics = [assert_full_streams(received), *exact_metrics(sampler.samples)]
            metrics += trend_metrics(sampler.samples)
    finally:
        # A run cut short still reports the cycles it finished.
        if not metrics and sampler.samples:
            metrics = [assert_full_streams(received), *exact_metrics(sampler.samples)]
        finish_and_check(
            test_name=TEST_NAME,
            subject="reactor-runtime",
            description=DESCRIPTION,
            version=version,
            duration_s=ENDURANCE_DURATION_SECONDS,
            started_at=started_at,
            sampler=sampler,
            live=live,
            metrics=metrics,
            iterations=cycle,
        )
