"""connection_churn_v1: the connection churn, with a client that speaks v1."""

from __future__ import annotations

from endurance.churn import run_churn
from endurance.harness import RuntimeProcess, V1Client

DESCRIPTION = (
    "One long-lived session. Each cycle a client joins over the v1 protobuf protocol, "
    "streams a webcam both ways until 60 frames come back while it requests the schema "
    "and, once a second, pings, sends client stats and a command, and closes once every "
    "request is answered; the runtime is then read once teardown has finished."
)


async def test_connection_churn_v1(runtime: RuntimeProcess) -> None:
    """Release everything a v1 connection held once it has closed, every time."""
    await run_churn(
        runtime, test_name="connection-churn-v1", description=DESCRIPTION, client=V1Client
    )
