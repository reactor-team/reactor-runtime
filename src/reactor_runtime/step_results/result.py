"""The step-result vocabulary: what a saved step is and how a save can fail."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RESULT_FILENAME = "result.json"
"""The manifest every step folder ends with, written last: once present, the step is complete."""

MEDIA_FILENAME = "output.mp4"
"""The media file a step's ``Output`` tracks are encoded into."""


class StepResultError(Exception):
    """A step result could not be saved."""


class StepResultsDisabledError(StepResultError):
    """Step results are off for this model, or no session is live to save into."""


class StepResultCancelledError(StepResultError):
    """The save was cancelled before the folder was complete."""


@dataclass(frozen=True)
class StepResult:
    """One saved step: its number and the files in its folder.

    Attributes:
        session_id: The id the session's step folders are stored under.
        step: The step number, counting from one within the session.
        files: Every file in the folder, ``result.json`` last.
    """

    session_id: str
    step: int
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render the result as the payload the ``step_result_ready`` event carries."""
        return {
            "session_id": self.session_id,
            "step": self.step,
            "files": list(self.files),
        }
