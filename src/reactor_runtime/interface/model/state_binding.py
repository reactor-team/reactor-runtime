"""Class-build helpers that wire a model's ``state:`` annotation to its commands.

Internal machinery, run once when a model class is declared: resolve the
:class:`InputState` subclass the class annotates, then stamp a ``set_<field>``
command handler for each of its public fields. Stamping happens before the
:class:`~reactor_runtime.interface.model.contract.ModelContract` is assembled,
so a generated setter is discovered as an ordinary command and validated and
documented exactly like a hand-written ``@event``.

Module-level functions rather than methods, so nothing here appears on the
surface a model author reads.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_type_hints

from reactor_runtime.interface.events.decorators import EVENT_ATTR, EventHandler, make_command
from reactor_runtime.interface.model.input_state import InputState

STATE_TYPE_ATTR = "__state_type__"
"""Where the resolved :class:`InputState` subclass is cached on a model class."""


def resolve_state_class(cls: type) -> type[InputState] | None:
    """Return the :class:`InputState` subclass named by the ``state`` annotation.

    Args:
        cls: The model class being declared.

    Returns:
        The annotated state class, or ``None`` when the class annotates no
        usable one — an abstract intermediate, or annotations that cannot be
        resolved at declaration time.
    """
    try:
        hints = get_type_hints(cls)
    except Exception:
        return None
    hint = hints.get("state")
    if isinstance(hint, type) and issubclass(hint, InputState):
        return hint
    return None


def stamp_auto_setters(cls: type, state_cls: type[InputState]) -> None:
    """Stamp a ``set_<field>`` command handler for each public state field.

    Each handler carries the same :class:`EventHandler` metadata an ``@event``
    decorator produces, so the contract treats it identically. A field whose
    ``set_`` name is already claimed by a hand-written handler is skipped, so an
    author can override the generated setter.

    Args:
        cls: The model class to stamp the handlers onto.
        state_cls: The state class whose public fields become commands.
    """
    existing = _existing_command_names(cls)
    try:
        hints = get_type_hints(state_cls)
    except Exception:
        hints = dict(getattr(state_cls, "__annotations__", {}))

    for field_name, info in state_cls._public_fields.items():
        command_name = f"set_{field_name}"
        if command_name in existing:
            continue
        field_type = hints.get(field_name, Any)
        command = make_command(command_name, [(field_name, field_type, info)])
        handler = _make_setter(field_name)
        setattr(
            handler,
            EVENT_ATTR,
            EventHandler(
                name=command_name,
                description=info.description or f"Set {field_name}.",
                command=command,
                is_async=False,
                reserved=(),
            ),
        )
        setattr(cls, command_name, handler)


def _existing_command_names(cls: type) -> set[str]:
    """Collect the command names already claimed by ``@event`` handlers on *cls*."""
    names: set[str] = set()
    for klass in cls.__mro__:
        for attr in vars(klass).values():
            handler = getattr(attr, EVENT_ATTR, None)
            if isinstance(handler, EventHandler):
                names.add(handler.name)
    return names


def _make_setter(field_name: str) -> Callable[..., None]:
    """Build the handler that writes one field onto the live state."""

    def handler(self: Any, **kwargs: Any) -> None:
        if self.state is None:
            return
        setattr(self.state, field_name, kwargs[field_name])

    return handler
