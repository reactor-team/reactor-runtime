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

    Every request of the batch gets it, with the same cause, and the cause
    may come from another session's input: its message, its arguments and its
    attributes may carry that session's data. The application, written
    together with the model, decides what of it reaches its own client, for
    example a generic message for the client and the cause in its own log. A
    model that can tell which request is bad returns an exception for that
    request alone, which arrives as itself.

    A ``BatchFailed`` always arrives as one. When its cause cannot travel (it
    does not pickle, it is too large, or its class is not in the receiving
    process), the cause arrives as a ``RuntimeError`` or a
    :class:`WorkerError` naming the original type and message.

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
    """A request or a result is larger than the link carries.

    The message names the payload's size and the limit it passed: the
    shared-memory block, with the ``runtime.worker`` key in ``reactor.yaml``
    that sizes it, or the link's frame limit. Nothing was sent: the request
    fails alone, and the link and every other request go on.
    """


class StepUnanswered(WorkerTimeout):
    """A step the model did not answer. The connection is still up."""


class StepExpired(StepUnanswered):
    """The worker dropped the request before stepping it: its timeout passed in the queue."""


class StepTimedOut(StepUnanswered):
    """No answer came within the step's timeout; the request was cancelled.

    A cancel withdraws a request still waiting in the worker's queue. A step the
    model is already running cannot be withdrawn: it keeps the session's one
    request until it ends, and a step sent meanwhile is refused with
    :class:`StepStillRunning`. The worker is slow or stuck; a worker that stays
    stuck reports itself unhealthy, and its deployment restarts it.
    """


class StepStillRunning(WorkerRejected):
    """The session's previous request, abandoned after a timeout, is still running on the worker.

    A session has one request at a time on the worker. Retry once the
    abandoned step ends; a worker stuck in it reports itself unhealthy.
    """


class ResetRequired(RuntimeError):  # noqa: N818 (named for the state a caller catches)
    """A step on a connection whose slot the application has not reset yet.

    The application resets its slot when its session starts, so a session never
    runs on state the worker kept for another one.
    """


class WorkerLost(WorkerCrashed):
    """The connection to the worker is gone, or the worker is going away: its message says why."""


class WorkerStopping(WorkerLost):
    """The worker is stopping, and answered a request still queued with this."""


class ClientClosed(WorkerLost):
    """The application closed its client: every call raises this from then on."""


class WorkerUnreachable(RuntimeError):  # noqa: N818 (named for the state a caller catches)
    """This deployment cannot reach a model server, and trying again will not help.

    The message says what is missing: the socket's directory (the volume the
    two halves share is not mounted), a socket this process may not open, or
    shared memory the two ends do not both see.
    """
