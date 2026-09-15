"""ApplicationError and StepOutcome as an author reads them."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from reactor_runtime import ApplicationError, Output, StepOutcome, Video


class Frame(Output):
    main_video: Video


class WaitingForCamera(ApplicationError):  # noqa: N818 (a refusal reason, named for what it says)
    def __init__(self) -> None:
        super().__init__("waiting for 4 webcam frames")


def test_application_error_carries_its_reason_as_the_message() -> None:
    assert str(ApplicationError("paused")) == "paused"


def test_a_subclass_is_still_an_application_error() -> None:
    error = WaitingForCamera()
    assert isinstance(error, ApplicationError)
    assert str(error) == "waiting for 4 webcam frames"


def test_outcome_defaults_to_nothing_happened() -> None:
    outcome = StepOutcome()
    assert outcome.result is None
    assert outcome.error is None
    assert outcome.elapsed == 0.0


def test_outcome_refuses_a_result_and_an_error_together() -> None:
    with pytest.raises(ValueError, match="never both"):
        StepOutcome(result=1, error=RuntimeError("boom"))


def test_outcome_is_frozen() -> None:
    outcome = StepOutcome(result=1, elapsed=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.result = 2  # type: ignore[ty:invalid-assignment]  # the rejection under test


def test_to_output_passes_an_output_through() -> None:
    frame = Frame(main_video=np.zeros((2, 2, 3), dtype=np.uint8))
    assert StepOutcome(result=frame).to_output() is frame


def test_to_output_reads_none_as_nothing_to_show() -> None:
    assert StepOutcome(result=None).to_output() is None


@pytest.mark.parametrize(
    "result",
    [np.zeros((2, 2, 3), dtype=np.uint8), (1, 2), {"frames": 1}, "frame"],
    ids=["array", "tuple", "dict", "str"],
)
def test_to_output_refuses_anything_else_by_name(result: object) -> None:
    with pytest.raises(NotImplementedError, match=type(result).__name__) as excinfo:
        StepOutcome(result=result).to_output()
    message = str(excinfo.value)
    assert "return an Output subclass from generate()" in message
    assert "override collect_step()" in message


def test_to_output_raises_the_error_the_outcome_holds() -> None:
    class RolloutExhausted(Exception):  # noqa: N818 (the model's own error, named for the state)
        pass

    with pytest.raises(RolloutExhausted):
        StepOutcome(error=RolloutExhausted()).to_output()
