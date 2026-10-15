"""connection_churn: one session, a client connecting and leaving over and over."""

from __future__ import annotations

import os

from harness import BareClient, ResourceSampler, RuntimeProcess, gray_frame
from report import LiveReporter, MetricResult, finish_and_check, now_iso
from trends import ENDURANCE_DURATION_SECONDS, exact_metrics, trend_metrics

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
    no_media_cycles = 0
    metrics: list[MetricResult] = []
    try:
        while not sampler.deadline_reached():
            received = await BareClient(runtime, session["session_id"]).run(frame)
            if received == 0:
                no_media_cycles += 1
            reading, _ = runtime.settled_probe()
            sampler.sample(cycle=cycle, reading=reading)
            live.update(sampler.samples, errors=no_media_cycles)
            cycle += 1
        metrics = exact_metrics(sampler.samples)
        metrics += trend_metrics(sampler.samples)
    finally:
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
            errors=no_media_cycles,
        )
