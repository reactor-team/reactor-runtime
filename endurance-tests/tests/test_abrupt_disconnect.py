"""abrupt_disconnect: one session, a client killed mid-stream over and over."""

from __future__ import annotations

from endurance.abrupt import KilledClient
from endurance.churn import run_churn
from endurance.harness import RuntimeEndpoint, RuntimeProcess
from endurance.report import DEFAULT_OUTPUT_DIR

DESCRIPTION = (
    "One long-lived session. Each cycle a v1 client in its own process joins, streams "
    "a webcam both ways until 60 frames come back, streams on for a moment, and is "
    "killed with SIGKILL: it never closes its peer connection, so the runtime learns "
    "it is gone only from the wire going silent. The runtime is then read once it has "
    "dropped the connection and finished tearing it down."
)

# How long each client streams on past its burst before it is killed, in
# turn: the kill lands mid-frame, mid-command and mid-stats at different cycles.
LINGER_S = (0.0, 0.4, 1.3)


async def test_abrupt_disconnect(runtime: RuntimeProcess) -> None:
    """Release everything a killed client's connection held, every time."""
    cycle = 0
    with (DEFAULT_OUTPUT_DIR / "abrupt-disconnect-clients.log").open("w") as log:

        def client(runtime: RuntimeEndpoint, session_id: str) -> KilledClient:
            nonlocal cycle
            linger_s = LINGER_S[cycle % len(LINGER_S)]
            cycle += 1
            return KilledClient(runtime, session_id, log=log, linger_s=linger_s)

        await run_churn(
            runtime, test_name="abrupt-disconnect", description=DESCRIPTION, client=client
        )
