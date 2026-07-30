"""The application: binds the engine and owns the product surface.

Everything the client can do comes from the engine's declarations — ``move``,
``brush``, ``init``, and the ``camera`` track are generated from them, the
outbound track is declared for us, and the default loop drives the rollout.
What is written here is only what the engine had no business knowing: a house
rule about movement, and a convenience command the engine never declared.

Delete the two overrides and the application is two lines.
"""

from __future__ import annotations

from engine import Direction, Move, PaintPipeline
from reactor_runtime import EnginePipeline, event, override_input

BANNED = {"up"}
"""Directions this deployment does not allow, whatever the engine accepts."""


class PaintApp(EnginePipeline):
    """A cursor you steer around a canvas, painting as it goes."""

    engine = PaintPipeline

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
