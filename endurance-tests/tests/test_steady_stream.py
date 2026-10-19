"""steady_stream: one client streaming for the whole run, read as it streams."""

from __future__ import annotations

import os
from typing import Any

from endurance.harness import (
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
    ZERO_FIELDS,
    Sample,
    assert_always_zero,
    assert_never_grows,
    process_metrics,
    trend_metrics,
)

DESCRIPTION = (
    "One session and one v1 client that stays connected for the whole run, streaming "
    "a webcam both ways while it pings, sends client stats and a command once a "
    "second. The runtime is read every 15 s as it streams: what one live connection "
    "holds must not grow, and every interval must get 60 frames back or more. At the "
    "end the client leaves and the runtime is read once teardown has finished."
)

# How long the client streams between two readings.
SAMPLE_INTERVAL_S = 15.0
# Counts a live connection holds for as long as it streams, read against the
# first reading taken while it streams.
_STREAMING_FIELDS = ("live_connections", "live_peers", "live_pacers", "live_client_stats", "tasks")
# Counts that are 0 while it streams: no pacer has stopped yet, and nothing
# should leave model output to the cyclic garbage collector.
_ZERO_WHILE_STREAMING = ("pacers_holding_output", "pinned_batches")


# Probes per streaming reading, and the streaming between them. A request
# in flight can hold a task for a moment, so the reading keeps the fewest
# tasks any probe saw: a task that leaks shows in every probe.
_PROBES_PER_READING = 3
_PROBE_GAP_S = 0.3


async def _streaming_probe(
    runtime: RuntimeProcess, client: V1Client, frame: tuple[bytes, int, int]
) -> dict[str, Any]:
    """Read the probe while the client streams, keeping the fewest tasks seen."""
    readings = [runtime.probe()]
    for _ in range(_PROBES_PER_READING - 1):
        await client.stream_for(frame, _PROBE_GAP_S)
        readings.append(runtime.probe())
    return {**readings[-1], "tasks": min(reading["tasks"] for reading in readings)}


def _streaming_checks(samples: list[Sample], children_baseline: int) -> list[MetricResult]:
    return [
        *(assert_always_zero(samples, field=field) for field in _ZERO_WHILE_STREAMING),
        *(assert_never_grows(samples, field=field) for field in _STREAMING_FIELDS),
        *process_metrics(samples, children_baseline=children_baseline),
    ]


def _after_close_checks(
    final: Sample, tasks_baseline: int, children_baseline: int
) -> list[MetricResult]:
    """Check the reading taken once the client has left, each count named as such."""
    return [
        *(
            assert_always_zero([final], field=field, name=f"{field}_after_close")
            for field in ZERO_FIELDS
        ),
        assert_never_grows(
            [final], field="tasks", baseline=tasks_baseline, name="tasks_after_close"
        ),
        assert_never_grows(
            [final],
            field="child_processes",
            baseline=children_baseline,
            name="child_processes_after_close",
        ),
    ]


async def test_steady_stream(runtime: RuntimeProcess) -> None:
    """Hold what one long connection needs and no more, then release it all."""
    session = runtime.request("POST", "/start_session", {})[1]
    version = session["server_info"]["server_version"]
    if source := os.environ.get("ENDURANCE_RUNTIME_REF"):
        version = f"from {source}"
    frame = gray_frame()
    sampler = ResourceSampler(runtime)
    live = LiveReporter("steady-stream", ENDURANCE_DURATION_SECONDS)
    started_at = now_iso()
    count = 0
    received: list[int] = []
    metrics: list[MetricResult] = []
    client = V1Client(runtime, session["session_id"])
    final: Sample | None = None
    streaming: list[Sample] = []

    def checks() -> list[MetricResult]:
        results = [assert_full_streams(received)]
        if streaming:
            results += _streaming_checks(streaming, sampler.initial_children)
        results.append(assert_answered([client.collect_unanswered()]))
        if final is not None:
            results += _after_close_checks(final, sampler.initial_tasks, sampler.initial_children)
        return results

    try:
        await client.open(frame)
        while not sampler.deadline_reached():
            if ENDURANCE_SMOKE_CYCLES and count >= ENDURANCE_SMOKE_CYCLES:
                break
            before = client.received_video
            await client.stream_for(frame, SAMPLE_INTERVAL_S)
            received.append(client.received_video - before)
            reading = await _streaming_probe(runtime, client, frame)
            streaming.append(sampler.sample(cycle=count, reading=reading))
            live.update(sampler.samples)
            count += 1
        if not streaming:
            raise AssertionError(
                f"no interval completed in {ENDURANCE_DURATION_SECONDS:g}s: "
                "raise ENDURANCE_DURATION_SECONDS"
            )
        await client.wait_for_replies()
        client.close()
        # Settled on the tasks too: tasks_after_close is judged on this one
        # reading, and a task still closing the connection is not a leak.
        reading, _ = runtime.settled_probe(tasks=sampler.initial_tasks)
        # Checked on its own: teardown is not another streaming interval.
        final = sampler.sample(cycle=count, reading=reading, record=False)
        # A smoke run judges what its intervals show on their own, as every
        # scenario's smoke does, so it cannot fail a pull request on a count
        # or a trend that only a long run can read.
        if ENDURANCE_SMOKE_CYCLES:
            metrics = [
                assert_full_streams(received),
                assert_answered([client.collect_unanswered()]),
            ]
        else:
            metrics = [*checks(), *trend_metrics(streaming)]
    finally:
        client.close()
        # A run cut short still reports the intervals it finished.
        if not metrics and sampler.samples:
            metrics = checks()
        finish_and_check(
            test_name="steady-stream",
            subject="reactor-runtime",
            description=DESCRIPTION,
            version=version,
            duration_s=ENDURANCE_DURATION_SECONDS,
            started_at=started_at,
            sampler=sampler,
            live=live,
            metrics=metrics,
            iterations=count,
        )
