"""Replacing a generated event — :func:`override_input`.

The first of the three layers an application can reach for. The engine's
declaration produces an event whose wire payload is the input's own fields; an
override takes that event's place, so the handler's signature becomes the
payload and the handler decides what — if anything — reaches the window.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from reactor_runtime.engine_contract.inputs import UserInput

OVERRIDE_ATTR = "__reactor_input_override__"
"""Where :func:`override_input` records the input class a handler replaces."""

_Handler = TypeVar("_Handler", bound=Callable[..., Any])


def override_input(input_cls: type[UserInput]) -> Callable[[_Handler], _Handler]:
    """Replace the event generated for *input_cls* with this handler.

    The command keeps the input's wire name, so a client is unaffected, but its
    payload becomes the handler's parameters. Whatever the handler returns is
    queued into the window in the input's place; returning ``None`` drops it, so
    the model never sees it::

        @override_input(Move)
        def move(self, direction: str) -> Move | None:
            if direction not in ALLOWED:
                return None
            return Move(direction=mirror(direction))

    Args:
        input_cls: The engine-declared input whose event this handler replaces.

    Returns:
        A decorator that marks the handler as the override.
    """

    def decorator(func: _Handler) -> _Handler:
        setattr(func, OVERRIDE_ATTR, input_cls)
        return func

    return decorator
