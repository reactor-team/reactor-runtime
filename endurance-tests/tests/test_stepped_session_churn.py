"""stepped_session_churn: the connection churn, on a stepped session."""

from __future__ import annotations

from endurance.churn import run_churn
from endurance.harness import STEPS_PER_CYCLE, RuntimeProcess, StepClient

DESCRIPTION = (
    "One long-lived stepped session. Each cycle a client joins over the v1 protobuf "
    f"protocol, requests {STEPS_PER_CYCLE} steps one after another, each setting the "
    "effect and carrying one webcam frame, waits for every result, and closes; the "
    "runtime is then read once teardown has finished."
)


async def test_stepped_session_churn(runtime: RuntimeProcess) -> None:
    """Release everything a stepped client's requests held once it has closed, every time."""
    await run_churn(
        runtime,
        test_name="stepped-session-churn",
        description=DESCRIPTION,
        client=StepClient,
        start={"generation": "stepped"},
        expected=STEPS_PER_CYCLE,
        unit="step results",
    )
