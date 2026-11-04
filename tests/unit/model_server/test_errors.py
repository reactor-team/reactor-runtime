"""The model server's error classes: the families an application's ``except`` clauses name."""

from __future__ import annotations

import pytest

from reactor_runtime.distributed import WorkerCrashed, WorkerTimeout
from reactor_runtime.model_server import (
    ClientClosed,
    ModelServerError,
    ModelServerLost,
    ModelServerRejected,
    ModelServerStopping,
    ModelServerTimeout,
    ModelServerUnreachable,
    StepExpired,
    StepStillRunning,
    StepTimedOut,
    StepUnanswered,
)


@pytest.mark.parametrize(
    ("kind", "base"),
    [
        (ModelServerLost, RuntimeError),
        (ModelServerStopping, ModelServerLost),
        (ClientClosed, ModelServerLost),
        (StepUnanswered, TimeoutError),
        (StepExpired, StepUnanswered),
        (StepTimedOut, StepUnanswered),
        (ModelServerTimeout, TimeoutError),
        (ModelServerRejected, ValueError),
        (StepStillRunning, ModelServerRejected),
        (ModelServerUnreachable, RuntimeError),
        (ModelServerError, RuntimeError),
    ],
)
def test_each_error_sits_directly_on_its_base(kind: type, base: type) -> None:
    assert kind.__bases__ == (base,)


def test_a_connect_timeout_is_not_a_step_left_unanswered() -> None:
    # Sibling TimeoutErrors: an except around a step does not catch a connect
    # timeout, and an except around a connect does not catch a step's.
    assert not issubclass(ModelServerTimeout, StepUnanswered)
    assert not issubclass(StepUnanswered, ModelServerTimeout)


@pytest.mark.parametrize("kind", [ModelServerLost, StepUnanswered, ModelServerTimeout])
def test_the_distributed_groups_errors_do_not_catch_the_model_servers(kind: type) -> None:
    assert not issubclass(kind, (WorkerCrashed, WorkerTimeout))
