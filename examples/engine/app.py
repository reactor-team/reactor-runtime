"""The application: binds the engine and owns the product surface.

Everything the client can do comes from the engine's declarations — ``move``,
``brush``, ``init``, and the ``camera`` track are generated from them, and the
default loop drives the rollout. What is written here is only what the engine
had no business knowing: which outbound track its frames land on, a house rule
about movement, and a convenience command the engine never declared.

Delete the two overrides and the application is three lines.
"""

from __future__ import annotations

from engine import Direction, Move, PaintPipeline
from reactor_runtime import EnginePipeline, Output, Video, event, override_input

BANNED = {"up"}
"""Directions this deployment does not allow, whatever the engine accepts."""


class Canvas(Output):
    """The outbound topology the engine's frames fill."""

    main_video: Video


class PaintApp(EnginePipeline):
    """A cursor you steer around a canvas, painting as it goes."""

    engine = PaintPipeline
    output: Canvas

    @override_input(Move)
    def move(self, direction: Direction, speed: float = 8.0) -> Move | None:
        """Move the brush, refusing a direction this deployment does not allow."""
        if direction in BANNED:
            return None
        return Move(direction=direction, speed=speed)

    @event(name="dash", description="Move four steps in one direction at once.")
    def dash(self, direction: Direction) -> None:
        """Queue a burst of moves, which the engine's fold integrates into a path."""
        for _ in range(4):
            self.inputs.push(Move(direction=direction, speed=16.0))
