"""What a session's application sees when its request does not come back as a result.

Every outcome other than a result is its own exception class, so an
application catches a class instead of matching text. A model's own exception
reaches the application as itself: the model server sends it whole, and the
application catches the model's type. The classes here cover what the model
server and the link report around the model.

Three families group them for an ``except`` clause:

* :class:`ModelServerLost` (a ``RuntimeError``): the model server is gone or
  going away. :class:`ModelServerStopping` and :class:`ClientClosed` are its
  kinds.
* :class:`StepUnanswered` (a ``TimeoutError``): a step or a reset got no answer
  in time, and the link is still up. :class:`StepExpired` and
  :class:`StepTimedOut` are its kinds.
* :class:`ModelServerRejected` (a ``ValueError``): the model server refused
  this request or this session. :class:`StepStillRunning` is its kind.

``ModelClient.connect()`` also raises :class:`ModelServerTimeout` and
:class:`ModelServerUnreachable`. ``ModelServerTimeout`` is a ``TimeoutError``
beside ``StepUnanswered``, not above it, so ``except ModelServerTimeout``
around a connect and ``except StepUnanswered`` around a step each catch their
own case, and ``except TimeoutError`` catches both.
"""

from __future__ import annotations


class ModelServerError(RuntimeError):
    """The model failed a request with an exception this process cannot rebuild.

    The model server sends the model's exception whole; when its class cannot
    be imported here (it lives in a module only the model server loads), this
    stands in for it. Its message is the exception's qualified type and
    message, as ``"module.Type: message"``. Keep the exceptions a model raises
    for a request in a module both halves import, next to the request and
    result types, to receive them as themselves.

    It is not the base of the other classes here: catch each outcome by its
    own class.
    """


class BatchFailed(RuntimeError):  # noqa: N818 (named for the event a caller catches)
    """The model's ``step()`` raised for the whole batch this request was in.

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
    :class:`ModelServerError` naming the original type and message.

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


class ModelServerRejected(ValueError):  # noqa: N818 (named for the event a caller catches)
    """The model server refused a request or a session, and said why.

    Only the session it names is affected; the model server and every other
    session go on.
    """


class PayloadTooLarge(ValueError):  # noqa: N818 (named for the event a caller catches)
    """A request or a result is larger than the link carries.

    The message names the payload's size and the limit it passed: the
    shared-memory block, with the ``model.resources.workers.shared_memory`` key
    in ``reactor.yaml`` that sizes it, or the link's frame limit. Nothing was
    sent: the request fails alone, and the link and every other request go on.
    """


class StepUnanswered(TimeoutError):  # noqa: N818 (named for the event a caller catches)
    """A step the model did not answer. The connection is still up."""


class StepExpired(StepUnanswered):
    """The model server answered the request without a result.

    Its timeout passed while it waited in the queue, so it was never stepped,
    or while the model ran it, or its session reset. A request the model was
    running still keeps the session's one request until the model finishes
    it, so a step sent meanwhile is refused with :class:`StepStillRunning`.
    """


class StepTimedOut(StepUnanswered):
    """No answer came within the step's timeout; the request was cancelled.

    A cancel withdraws a request still waiting in the model server's queue,
    and tells the model to skip one whose work has not started. A step the
    model is already running keeps the session's one request until it ends,
    and a step sent meanwhile is refused with :class:`StepStillRunning`. The
    model server is slow or stuck; a model
    server that stays stuck reports itself unhealthy, and its deployment
    restarts it.
    """


class StepStillRunning(ModelServerRejected):
    """The session's previous request, abandoned after a timeout, still runs on the model server.

    A session has one request at a time on the model server. Retry once the
    abandoned step ends; a model server stuck in it reports itself unhealthy.
    """


class ResetRequired(RuntimeError):  # noqa: N818 (named for the state a caller catches)
    """A step on a connection whose slot the application has not reset yet.

    The application resets its slot when its session starts, so a session never
    runs on state the model server kept for another one.
    """


class ModelServerLost(RuntimeError):  # noqa: N818 (named for the event a caller catches)
    """The connection to the model server is gone, or the model server is going away.

    For a lost link and for :class:`ModelServerStopping`, the message starts
    with ``model_server_lost:`` and says why, for example
    ``model_server_lost: the model server closed the link``; a session runtime
    that re-raises it ends its session with that message as the error.
    :class:`ClientClosed` says the client is closed.
    """


class ModelServerStopping(ModelServerLost):
    """The model server is stopping, and answered a request still queued with this."""


class ClientClosed(ModelServerLost):
    """The application closed its client: every call raises this from then on."""


class ModelServerTimeout(TimeoutError):  # noqa: N818 (named for the event a caller catches)
    """No model server opened this session within the connect timeout.

    ``ModelClient.connect()`` raises it once the link's ``connect_timeout``
    has passed. The message names
    the timeout and the last thing the client saw: no listener yet, a link
    the model server accepted but did not open, or a link it closed first.
    A model server that comes up later may still open the session, so a
    caller may try again.
    """


class ModelServerUnreachable(RuntimeError):  # noqa: N818 (named for the state a caller catches)
    """This deployment cannot reach a model server, and trying again will not help.

    The message says what is missing: the socket's directory (the volume the
    two halves share is not mounted), a socket this process may not open, or
    shared memory the two ends do not both see.
    """
