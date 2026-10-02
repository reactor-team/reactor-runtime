"""Step results: one folder per step, saved to local disk and served over HTTP.

A step result is everything one step produced, as files: the step's ``Output``
encoded as ``output.mp4`` with every track as its own stream, any extra files
the model adds, and a ``result.json`` that lists them. The model announces a
step as a :class:`~reactor_runtime.core.CompletedStep` and goes on; the
:class:`StepResultStore` saves it on a worker of its own and serves the
folders, and :class:`StepResult` is what the ``step_result_ready`` event
announces once a folder is complete.
"""

from reactor_runtime.step_results.result import (
    MEDIA_FILENAME,
    RESULT_FILENAME,
    StepResult,
    StepResultCancelledError,
    StepResultError,
    StepResultsDisabledError,
)
from reactor_runtime.step_results.store import StepReadyCallback, StepResultStore

__all__ = [
    "MEDIA_FILENAME",
    "RESULT_FILENAME",
    "StepReadyCallback",
    "StepResult",
    "StepResultCancelledError",
    "StepResultError",
    "StepResultStore",
    "StepResultsDisabledError",
]
