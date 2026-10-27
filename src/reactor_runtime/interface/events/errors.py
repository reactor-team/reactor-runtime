"""The errors an author raises to answer a command, refuse a step, or drop one.

A :class:`CommandError` answers one client's command with a failure. An
:class:`ApplicationError` refuses one step of the loop. A
:class:`ModelUnavailable` drops a step the model could not run because
something it depends on is gone.
"""

from __future__ import annotations


class CommandError(Exception):
    """Raise from a command handler to answer its caller with a failure.

    The runtime sends the code and the message to the client that issued the
    command, correlated with that command, so an awaiting caller rejects with a
    reason instead of waiting for a reply that never arrives. Both fields cross
    the wire unchanged, so write the message for the client rather than for a log.

    Any other exception a handler raises is a fault the client cannot act on. The
    runtime logs it with its traceback and answers with a generic code, so the
    detail stays out of the reply.

    Raising is the only way to refuse. A handler that sends an error message and
    returns has succeeded as far as the runtime is concerned: the command is
    acknowledged, and its caller resolves as on success.

    Example::

        @event(name="generate")
        async def generate(self, prompt: str) -> Image:
            if self.credits <= 0:
                raise CommandError("quota_exceeded", "No credits remain for this session.")
            return Image(url=self.render(prompt))

    Attributes:
        code: Short, stable token the client branches on.
        message: Readable explanation for the client.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class ApplicationError(Exception):
    """Raise from ``process_input()`` to refuse the step.

    The application is not ready for a step: it is paused, a required input has
    not arrived, a prompt is not set. The model is not called, the reason is
    logged, and the loop asks again.

    Subclass it when a reason should be told apart in code or in a log filter,
    and give the subclass its own message::

        class WaitingForCamera(ApplicationError):
            def __init__(self) -> None:
                super().__init__("waiting for 4 webcam frames")

    The loop catches the base class, so every subclass refuses the same way.
    """


class ModelUnavailable(Exception):  # noqa: N818 (named for the state a caller catches)
    """Raise from ``process_input()`` or ``generate()`` when the model cannot run the step now.

    For a model whose work happens somewhere it does not own, such as a model
    worker in another process, that has gone away. The default step loop drops
    the step: ``process_output()`` is not called, nothing is emitted or
    reported, and the loop asks again a moment later. Raising it does not end
    the session; :meth:`~reactor_runtime.ReactorApp.report_unavailable` does, and
    a model that raises this calls that too (a
    :class:`~reactor_runtime.model_worker.WorkerClient` connected with the app
    does both). A hand-written ``run()`` catches it itself.
    """
