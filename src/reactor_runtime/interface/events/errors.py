"""Failure reporting for command handlers — :class:`CommandError`."""

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
