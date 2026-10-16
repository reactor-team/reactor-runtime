"""connection_churn: one session, a client connecting and leaving over and over."""

from __future__ import annotations

from endurance.churn import run_churn
from endurance.harness import BareClient, RuntimeProcess

DESCRIPTION = (
    "One long-lived session. Each cycle a client joins, streams a webcam both ways "
    "until 60 frames come back, and closes; the runtime is then read once teardown "
    "has finished. The client speaks the v0 JSON control protocol."
)


async def test_connection_churn(runtime: RuntimeProcess) -> None:
    """Release everything a connection held once it has closed, every time."""
    await run_churn(
        runtime, test_name="connection-churn", description=DESCRIPTION, client=BareClient
    )
