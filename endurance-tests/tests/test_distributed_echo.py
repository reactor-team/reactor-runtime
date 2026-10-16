"""distributed_echo: the v1 connection churn, the model's engine in a worker process."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from endurance.churn import run_churn
from endurance.harness import SUITE_DIR, RuntimeProcess, V1Client
from endurance.report import DEFAULT_OUTPUT_DIR

DESCRIPTION = (
    "One long-lived session on a model whose engine runs in a worker process behind "
    "a DistributedRunner, so every frame goes to the worker and back through shared "
    "memory. Each cycle a v1 client joins, streams until 60 frames come back, and "
    "closes. Besides what the churn checks, the worker processes and the shared-memory "
    "blocks in /dev/shm must never grow, and the workers' memory is a trend."
)


@pytest.fixture
def runtime(request: pytest.FixtureRequest) -> Iterator[RuntimeProcess]:
    """A runtime serving the distributed model, its log kept beside the reports."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = RuntimeProcess(
        DEFAULT_OUTPUT_DIR / f"{request.node.name}-runtime.log",
        model_dir=SUITE_DIR / "model-distributed",
    )
    try:
        proc.wait_ready()
        yield proc
    finally:
        proc.stop()


async def test_distributed_echo(runtime: RuntimeProcess) -> None:
    """Release everything a connection held, and hold the workers' memory flat."""
    await run_churn(runtime, test_name="distributed-echo", description=DESCRIPTION, client=V1Client)
