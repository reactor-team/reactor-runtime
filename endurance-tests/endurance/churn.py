"""The cycle loop the scenarios share, and the connection churn built on it.

A scenario differs from another in what one cycle does; the loop that repeats
it until the deadline, the per-cycle reading, the checks and the report live
here once.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from endurance.harness import (
    ResourceSampler,
    RuntimeProcess,
    assert_answered,
    assert_full_streams,
    assert_released,
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


@dataclass
class CycleOutcome:
    """What one cycle's client saw.

    Attributes:
        received: The frames that came back.
        unanswered: The requests that were refused or never answered, or
            ``None`` for a client that sends none it waits on.
    """

    received: int
    unanswered: list[str] | None = None


async def run_cycles(
    runtime: RuntimeProcess,
    *,
    test_name: str,
    description: str,
    cycle: Callable[[int], Awaitable[CycleOutcome]],
    min_smoke_cycles: int = 0,
) -> None:
    """Run *cycle* until the run's deadline, reading the runtime after each, then check.

    *cycle* is called with the cycle's index. Once it returns, the runtime is
    read when its client is gone and teardown has finished, and that reading
    is the cycle's sample. A smoke run takes ``ENDURANCE_SMOKE_CYCLES``
    cycles, or *min_smoke_cycles* when that is more: a scenario whose cycles
    take turns passes their count, so a smoke run reaches every one.
    """
    smoke_cycles = max(ENDURANCE_SMOKE_CYCLES, min_smoke_cycles) if ENDURANCE_SMOKE_CYCLES else 0
    version = runtime.get("/session")["server_info"]["server_version"]
    # CI can run the suite against another release's runtime source, while the
    # package metadata still names this checkout's version.
    if source := os.environ.get("ENDURANCE_RUNTIME_REF"):
        version = f"from {source}"
    sampler = ResourceSampler(runtime)
    live = LiveReporter(test_name, ENDURANCE_DURATION_SECONDS)
    started_at = now_iso()
    count = 0
    received: list[int] = []
    # Each v1 cycle's requests that went unanswered or were refused.
    unanswered: list[list[str]] = []
    # Seconds each cycle's connection took to leave the runtime's registry.
    released: list[float | None] = []
    metrics: list[MetricResult] = []

    def stream_checks() -> list[MetricResult]:
        checks = [assert_full_streams(received), assert_released(released)]
        if unanswered:
            checks.append(assert_answered(unanswered))
        return checks

    try:
        while not sampler.deadline_reached():
            if smoke_cycles and count >= smoke_cycles:
                break
            outcome = await cycle(count)
            received.append(outcome.received)
            if outcome.unanswered is not None:
                unanswered.append(outcome.unanswered)
            reading, released_after_s = runtime.settled_probe()
            released.append(released_after_s)
            sampler.sample(cycle=count, reading=reading)
            live.update(sampler.samples, retained=retained_cycles(sampler.samples))
            count += 1
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
            iterations=count,
        )


class ChurnClient(Protocol):
    """One cycle's client, as the churn drives it."""

    @property
    def unanswered(self) -> list[str] | None:
        """The requests the cycle sent that were refused or never answered.

        ``None`` for a client that sends none it waits on.
        """
        ...

    async def run(self, frame: tuple[bytes, int, int]) -> int:
        """Connect, stream *frame*, leave, and return the frames that came back."""
        ...


async def run_churn(
    runtime: RuntimeProcess,
    *,
    test_name: str,
    description: str,
    client: Callable[[RuntimeProcess, str], ChurnClient],
) -> None:
    """Churn *client* connections on one session until the run's deadline, then check.

    Each cycle a new client from *client* joins, streams until its frames come
    back, and leaves; the runtime is read once teardown has finished.
    """
    session = runtime.request("POST", "/start_session", {})[1]
    frame = gray_frame()

    async def cycle(_: int) -> CycleOutcome:
        connection = client(runtime, session["session_id"])
        received = await connection.run(frame)
        return CycleOutcome(received=received, unanswered=connection.unanswered)

    await run_cycles(runtime, test_name=test_name, description=description, cycle=cycle)
