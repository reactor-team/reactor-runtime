"""What a session's application sees when its request does not come back as a result.

Every outcome other than a result is its own exception class, so an
application catches a class instead of matching text. A model's own exception
reaches the application as itself: the worker sends it whole, and the
application catches the model's type. The classes here cover what the worker
and the link report around the model.
"""

from __future__ import annotations

from reactor_runtime.distributed.errors import WorkerCrashed, WorkerTimeout


class WorkerError(RuntimeError):
    """The model failed a request with an exception this process cannot rebuild.

    The worker sends the model's exception whole; when its class cannot be
    imported here (it lives in a module only the worker loads), this stands in
    for it. Its message is the exception's qualified type and message, as
    ``"module.Type: message"``. Keep the exceptions a model raises for a
    request in a module both halves import, next to the request and result
    types, to receive them as themselves.
    """


class BatchFailed(RuntimeError):  # noqa: N818 (named for the event a caller catches)
    """The worker's ``step()`` raised for the whole batch this request was in.

    Every request of the batch gets it, with the same cause. The cause may be
    another session's input; a model that can tell which request is bad returns
    an exception for that request alone, which arrives as itself.

    Attributes:
        cause: The exception ``step()`` raised.
    """

    def __init__(self, cause: BaseException) -> None:
        """Record the exception ``step()`` raised."""
        super().__init__(cause)
        self.cause = cause

    def __str__(self) -> str:
        """Name the cause's type and message."""
        return f"step() raised for the whole batch: {type(self.cause).__name__}: {self.cause}"


class WorkerRejected(ValueError):  # noqa: N818 (named for the event a caller catches)
    """The worker refused a request or a session, and said why.

    Only the session it names is affected; the worker and every other session
    go on.
    """


class PayloadTooLarge(ValueError):  # noqa: N818 (named for the event a caller catches)
    """A request or a result is larger than the shared memory ``reactor.yaml`` gives it.

    The message names the payload's size, the block's, and the
    ``runtime.worker`` key that sets it. Nothing was sent: the request fails
    alone, and the link and every other request go on.
    """


class StepUnanswered(WorkerTimeout):
    """A step the model did not answer. The connection is still up."""


class StepExpired(StepUnanswered):
    """The worker dropped the request before stepping it: its timeout passed in the queue."""


class WorkerLost(WorkerCrashed):
    """The connection to the worker is gone, or the worker is going away: its message says why."""


class WorkerStopping(WorkerLost):
    """The worker is stopping, and answered a request still queued with this."""
