"""Greeter — a tiny command/message model for exercising request/response.

No media processing and no weights. It emits a plain animated frame so it is a
valid video model, but its point is the command surface, which covers the three
outcomes a client's awaited ``send_command`` can resolve to:

* ``greet`` returns a :class:`Greeting` message, so the awaited command resolves
  with a body the client can read.
* ``set_volume`` returns nothing, so the awaited command resolves with ``None``
  once the handler has run (a bodyless acknowledgement).
* ``boom`` raises :class:`CommandError`, so the awaited command rejects with the
  code and message the handler chose.

Run with:  reactor run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reactor_runtime import (
    CommandError,
    InputField,
    MessageField,
    ModelMessage,
    Output,
    ReactorModel,
    Video,
    event,
)

WIDTH = 320
HEIGHT = 240
FPS = 30


class GreeterOutput(Output):
    """The single video track this model sends back."""

    main_video: Video


class Greeting(ModelMessage):
    """The reply to a ``greet`` command."""

    text: str = MessageField(description="The greeting the model built for the caller.")
    count: int = MessageField(description="How many greetings this session has produced.")


class Greeter(ReactorModel):
    """Answer commands. The video track just proves media still flows alongside."""

    fps = FPS

    def load(self, config_path: Path | None) -> None:
        """Set the session counter. Reads no config and loads no weights."""
        self.greetings = 0
        self.volume = 1.0

    @event(name="greet", description="Greet someone by name and reply with the greeting.")
    async def greet(
        self,
        name: str = InputField(default="world", max_length=64, description="Who to greet."),
    ) -> Greeting:
        """Build a greeting and return it as this command's correlated reply.

        Returning the message is what makes the client's awaited ``send_command``
        resolve with a body instead of ``None``.
        """
        self.greetings += 1
        return Greeting(text=f"Hello, {name}!", count=self.greetings)

    @event(name="set_volume", description="Set the output volume. Acknowledges with no body.")
    async def set_volume(
        self,
        volume: float = InputField(default=1.0, ge=0.0, le=1.0, description="0 silent, 1 full."),
    ) -> None:
        """Store the volume and return nothing.

        A handler that returns nothing still acknowledges, so the client's
        awaited ``send_command`` resolves with ``None`` rather than waiting for a
        reply that never comes.
        """
        self.volume = volume

    @event(name="boom", description="Always fail, to show how an awaited command rejects.")
    async def boom(self) -> None:
        """Raise a command error so the awaited command rejects with a reason."""
        raise CommandError(code="BOOM", message="this command always fails on purpose")

    async def run(self) -> None:
        """Emit a plain animated frame while a client is connected.

        The colour drifts with the frame index so motion is visible; none of the
        command state feeds it, because the commands are the point here.
        """
        frame_index = 0
        while True:
            await self.connected.wait()
            while self.connected.is_set():
                shade = (frame_index * 2) % 256
                frame = np.full((HEIGHT, WIDTH, 3), shade, dtype=np.uint8)
                frame[:, :, 1] = (shade + 85) % 256
                frame_index += 1
                await self.emit(GreeterOutput(main_video=frame))
