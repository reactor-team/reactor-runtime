"""director_generation_churn: one-shot sessions on a director model, no client connected."""

from __future__ import annotations

import time
import urllib.error
import uuid
from collections.abc import Iterator

import pytest
from endurance.churn import CycleOutcome, run_cycles
from endurance.harness import CLOSE_TIMEOUT_S, SUITE_DIR, RuntimeProcess
from endurance.report import DEFAULT_OUTPUT_DIR

DESCRIPTION = (
    "Each cycle starts a session under a fresh id on a director model, with one "
    "enqueue as its starting input and steps 1, the way a request for one generation "
    "starts one, and no client connects. The runtime's own client keeps the session "
    "live, the model builds the shot and reports it as the session's one step, and "
    "the runtime ends the session. The cycle checks that the step's folder holds "
    "output.mp4, last_frame.png and shot.json. The runtime is read once the session "
    "is ready again."
)

# The files a director model's step keeps.
STEP_FILES = {"output.mp4", "last_frame.png", "shot.json"}
# How long a saved step may take to appear after its session ended.
_SAVE_TIMEOUT_S = 10.0


@pytest.fixture
def runtime(request: pytest.FixtureRequest) -> Iterator[RuntimeProcess]:
    """A runtime serving the director model, its log kept beside the reports."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = RuntimeProcess(
        DEFAULT_OUTPUT_DIR / f"{request.node.name}-runtime.log",
        model_dir=SUITE_DIR / "model-director",
    )
    try:
        proc.wait_ready()
        yield proc
    finally:
        proc.stop()


async def test_director_generation_churn(runtime: RuntimeProcess) -> None:
    """Release everything a one-shot session held once it has ended, every time."""

    async def cycle(n: int) -> CycleOutcome:
        session_id = str(uuid.uuid4())
        failures: list[str] = []
        runtime.request(
            "POST",
            "/start_session",
            {
                "session_id": session_id,
                "starting_input": {
                    "commands": [{"command": "enqueue", "data": {"prompt": f"shot {n}"}}]
                },
                "steps": 1,
            },
        )
        try:
            runtime.wait_ready(timeout_s=CLOSE_TIMEOUT_S)
        except TimeoutError:
            failures.append(f"the session was not ready {CLOSE_TIMEOUT_S:g}s after its start")
        shots = _saved_shots(runtime, session_id, failures)
        return CycleOutcome(received=shots, stop_failures=failures)

    await run_cycles(
        runtime,
        test_name="director-generation-churn",
        description=DESCRIPTION,
        cycle=cycle,
        expected=1,
        unit="shots",
    )


def _saved_shots(runtime: RuntimeProcess, session_id: str, failures: list[str]) -> int:
    """Return 1 when the session's step 1 holds a director shot's files, else 0."""
    deadline = time.monotonic() + _SAVE_TIMEOUT_S
    while True:
        try:
            steps = runtime.get(f"/sessions/{session_id}/steps")["steps"]
        except urllib.error.HTTPError:
            steps = []
        if steps or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    if steps != [1]:
        failures.append(f"the session saved steps {steps}, not [1]")
        return 0
    result = runtime.get(f"/sessions/{session_id}/steps/1")
    names = {file["name"] for file in result["files"]}
    if result["error"] is not None or not names.issuperset(STEP_FILES):
        failures.append(f"step 1 holds {sorted(names)} with error {result['error']!r}")
        return 0
    return 1
