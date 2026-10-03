"""Lock the step results surface: the step_result_ready fact and the /steps routes.

A consumer that collects a session's step results follows ``step_result_ready``
on ``/events`` and reads each folder under the session's own id: the list of
complete steps, each step's ``result.json``, and every file it lists. The
folders stay readable after the session ends.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import av
import numpy as np
from contract_helpers import Harness, JournalReader, SteppingState, running_runtime

from reactor_runtime import ReactorApp, StepCompleted, StepOutcome
from reactor_runtime.core import RuntimeConfig, StepResultsConfig
from reactor_runtime.interface.tracks import Output, Video

_SESSION_ID = "5b6c7d8e-0000-4000-8000-0000000000bb"


class SavingOutput(Output):
    frames: Video


class SavingModel(ReactorApp):
    """A model that keeps a note with every step's frames."""

    state: SteppingState

    def load(self, config_path: Path | None) -> None: ...

    def generate(self, input: SteppingState) -> SavingOutput:
        return SavingOutput(frames=np.zeros((2, 32, 32, 3), dtype=np.uint8))

    async def process_output(self, outcome: StepOutcome) -> StepCompleted:
        return StepCompleted(output=outcome.to_output(), files={"note.txt": b"two frames"})


async def _run_two_saved_steps(harness: Harness, root: Path) -> list[dict[str, Any]]:
    """Run a two-step session with step results on, until both folders are ready."""
    assert harness.runner.step_store is not None
    harness.runner.step_store._root = root
    journal = JournalReader(harness.runner)
    try:
        response = await harness.client.post(
            "/start_session", json={"session_id": _SESSION_ID, "steps": 2}
        )
        assert response.status_code == 200
        return [await journal.expect("step_result_ready") for _ in range(2)]
    finally:
        await journal.aclose()


def _saving_config() -> RuntimeConfig:
    return RuntimeConfig(model_ref="contract:Model", step_results=StepResultsConfig(enabled=True))


async def test_each_saved_step_is_announced_and_served(tmp_path: Path) -> None:
    async with running_runtime(model_cls=SavingModel, cfg=_saving_config()) as harness:
        ready = await _run_two_saved_steps(harness, tmp_path)

        assert [envelope["detail"] for envelope in ready] == [
            {"session_id": _SESSION_ID, "step": 1, "files": ["output.mp4", "note.txt"]},
            {"session_id": _SESSION_ID, "step": 2, "files": ["output.mp4", "note.txt"]},
        ]

        listed = await harness.client.get(f"/sessions/{_SESSION_ID}/steps")
        assert listed.status_code == 200
        assert listed.json()["steps"][:2] == [1, 2]

        result = await harness.client.get(f"/sessions/{_SESSION_ID}/steps/1")
        assert result.status_code == 200
        assert result.headers["content-type"] == "application/json"
        body = result.json()
        assert body["step"] == 1
        assert body["session_id"] == _SESSION_ID
        assert [entry["name"] for entry in body["files"]] == ["output.mp4", "note.txt"]
        assert body["error"] is None
        assert body["save_error"] is None
        assert set(body["timings"]) == {"generate_s", "encode_s"}

        note = await harness.client.get(f"/sessions/{_SESSION_ID}/steps/1/note.txt")
        assert note.status_code == 200
        assert note.content == b"two frames"
        assert note.headers["content-type"].startswith("text/plain")

        video = await harness.client.get(f"/sessions/{_SESSION_ID}/steps/1/output.mp4")
        assert video.status_code == 200
        assert video.headers["content-type"] == "video/mp4"
        with av.open(io.BytesIO(video.content), mode="r") as container:
            assert sum(1 for _ in container.decode(video=0)) == 2


async def test_step_folders_stay_readable_after_the_session_ends(tmp_path: Path) -> None:
    async with running_runtime(model_cls=SavingModel, cfg=_saving_config()) as harness:
        await _run_two_saved_steps(harness, tmp_path)
        journal = JournalReader(harness.runner)
        try:
            await journal.expect("cleanup_complete")
        finally:
            await journal.aclose()

        assert harness.runner.descriptor()["state"] == "ready"
        response = await harness.client.get(f"/sessions/{_SESSION_ID}/steps/2/note.txt")
        assert response.status_code == 200


async def test_what_is_not_a_complete_step_is_404(tmp_path: Path) -> None:
    async with running_runtime(model_cls=SavingModel, cfg=_saving_config()) as harness:
        await _run_two_saved_steps(harness, tmp_path)
        unknown = "00000000-0000-4000-8000-000000000000"

        for path in (
            f"/sessions/{unknown}/steps",
            "/sessions/not-a-uuid/steps",
            f"/sessions/{_SESSION_ID}/steps/999",
            f"/sessions/{_SESSION_ID}/steps/0",
            f"/sessions/{_SESSION_ID}/steps/1/secrets.txt",
            f"/sessions/{_SESSION_ID}/steps/1/result.json",
            f"/sessions/{_SESSION_ID}/steps/1/..%2F2%2Fnote.txt",
        ):
            response = await harness.client.get(path)
            assert response.status_code == 404, path


async def test_without_step_results_there_are_no_steps_to_read(harness: Harness) -> None:
    response = await harness.client.get(f"/sessions/{_SESSION_ID}/steps")

    assert response.status_code == 404
