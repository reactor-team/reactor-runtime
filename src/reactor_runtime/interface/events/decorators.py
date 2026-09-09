"""Handler decorators — ``@event`` and the lifecycle hooks.

A model author marks methods rather than registering classes. ``@event`` is the
workhorse: it reads the handler's signature and synthesises the :class:`Command`
subclass that mirrors it, so the wire payload a client sends is validated against
the exact parameters the handler declares.

The lifecycle decorators tag a method as a hook for a reactor-authoritative fact,
split by scope. ``@session_started`` / ``@session_ended`` fire once for the
session as a whole; ``@connected`` / ``@disconnected`` fire per client
connection, so a model can tell a fresh viewer joining an existing session apart
from the session itself beginning or ending. The scopes do not compose into one
another: a session ending is announced by ``@session_ended`` alone, never as one
``@disconnected`` per remaining client. ``@file_uploaded`` fires when a client
uploads a file. Each decorator only stamps metadata on the function; the
model class reads it back when it assembles its contract.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, get_type_hints

from reactor_runtime.core.model import Command
from reactor_runtime.core.naming import snake_to_pascal

RESERVED_PARAMS: tuple[str, ...] = ("client",)
"""Handler parameters the runtime injects rather than reading off the wire.

A handler opts in to one simply by declaring it; the parameter is stripped from
the synthesised command and never appears in the schema.
"""

EVENT_REGISTRY: dict[str, type[Command]] = {}
"""Every command an ``@event`` handler declares, by wire name.

Auto-populated when the decorator synthesises a handler's :class:`Command`, so
the model's command set is discoverable without inspecting the class. The
runtime runs one model per process, so this is that model's command surface;
the rendered schema reads it. A directly-defined ``Command`` subclass (without a
handler) is deliberately absent — only handler-backed events register.
"""

_RESERVED_EVENT_NAMES = frozenset({"connected", "disconnected"})

EVENT_ATTR = "__reactor_event__"
SESSION_STARTED_ATTR = "__reactor_session_started__"
SESSION_ENDED_ATTR = "__reactor_session_ended__"
CONNECTED_ATTR = "__reactor_connected__"
DISCONNECTED_ATTR = "__reactor_disconnected__"
FILE_UPLOADED_ATTR = "__reactor_file_uploaded__"


@dataclass(frozen=True)
class EventHandler:
    """Metadata an ``@event`` decorator stamps on a handler method.

    Attributes:
        name: The wire name a client sends to trigger the handler.
        description: Human-readable description, surfaced in the rendered schema.
        command: The :class:`Command` subclass synthesised from the signature.
        is_async: Whether the handler is a coroutine function.
        reserved: Reserved parameters the handler declares, in injection order.
    """

    name: str
    description: str
    command: type[Command]
    is_async: bool
    reserved: tuple[str, ...]


def _reserved_params(func: Callable[..., Any]) -> tuple[str, ...]:
    """Return the reserved parameters *func* declares, in registry order."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return ()
    return tuple(name for name in RESERVED_PARAMS if name in sig.parameters)


def event(
    *, name: str, description: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method as the handler for a named command.

    The client triggers the handler by sending a message with the matching
    ``type``. The method's annotated parameters define the payload; use
    :func:`InputField` as a default to attach validation constraints.

    A subclass inherits its bases' commands. Overriding one — its wire contract
    *or* the implementation that runs — requires re-applying ``@event``; the
    most-derived ``@event`` for a name wins. A plain, undecorated method of the
    same name is *not* an override: the inherited command stands, contract and
    handler both, and the base method is what runs. This keeps "declare a command
    with ``@event``" the single, explicit rule, with no implicit override.

    Args:
        name: The wire name a client sends as the ``type`` string.
        description: Human-readable description, surfaced in the rendered schema.

    Returns:
        A decorator that stamps the handler metadata onto the method.

    Raises:
        ValueError: If *name* collides with a reserved lifecycle name.
    """
    if name in _RESERVED_EVENT_NAMES:
        raise ValueError(
            f"@event(name={name!r}) is reserved for a lifecycle hook; use "
            "@connected / @disconnected instead."
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        command = _command_from_signature(func, name)
        EVENT_REGISTRY[name] = command
        setattr(
            func,
            EVENT_ATTR,
            EventHandler(
                name=name,
                description=description,
                command=command,
                is_async=inspect.iscoroutinefunction(func),
                reserved=_reserved_params(func),
            ),
        )
        return func

    return decorator


def session_started(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the handler run once when the session begins.

    Scoped to the whole session, not a single client: it fires when the session
    starts, before any client has connected. This is the hook for
    once-per-session initialization — it fires exactly once for every session
    the process serves, which no per-connection bookkeeping can guarantee (see
    ``@disconnected`` for why). Use ``@connected`` for per-client setup.
    """
    setattr(func, SESSION_STARTED_ATTR, True)
    return func


def session_ended(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the handler run once when the session ends.

    Scoped to the whole session, not a single client: it fires when the session
    itself ends, distinct from a client leaving. Use ``@disconnected`` for
    per-client teardown.
    """
    setattr(func, SESSION_ENDED_ATTR, True)
    return func


def connected(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the handler run once each time a client connects."""
    setattr(func, CONNECTED_ATTR, True)
    return func


def disconnected(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the handler run once when a client disconnects.

    Fires only when a client's own connection drops while the session stays
    live. When the session itself ends, its connections are torn down
    wholesale and this hook does not fire for them — the model hears
    ``@session_ended`` once instead. A counter incremented in ``@connected``
    and decremented here therefore does not return to zero across sessions;
    gate once-per-session work on ``@session_started``, and track
    first-vs-later-client logic within a session on session-scoped state.
    """
    setattr(func, DISCONNECTED_ATTR, True)
    return func


def file_uploaded(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a method as the handler run when a client uploads a file.

    The handler takes exactly one ``uploaded_file`` parameter (plus optional
    reserved parameters and ``self``).

    Args:
        func: The handler method.

    Returns:
        The same method, tagged as the upload hook.

    Raises:
        TypeError: If the handler's parameters are not exactly ``uploaded_file``.
    """
    sig = inspect.signature(func)
    params = [name for name in sig.parameters if name != "self"]
    required = [name for name in params if name not in RESERVED_PARAMS]
    if required != ["uploaded_file"]:
        raise TypeError(
            "@file_uploaded handler must take exactly one parameter named "
            f"'uploaded_file' (plus optional reserved parameters), got: {params}"
        )
    setattr(func, FILE_UPLOADED_ATTR, True)
    return func


def make_command(name: str, fields: list[tuple[str, Any] | tuple[str, Any, Any]]) -> type[Command]:
    """Build a :class:`Command` subclass for *name* from field tuples.

    The class is named so its PascalCase form round-trips to *name*, and its
    ``name`` is set explicitly so an irregular wire name is preserved exactly.

    Args:
        name: The command's wire name.
        fields: Field tuples of ``(name, type)`` or ``(name, type, default)``.

    Returns:
        The synthesised, fully resolved command class.
    """
    annotations: dict[str, Any] = {}
    namespace: dict[str, Any] = {}
    for field in fields:
        if len(field) == 2:
            field_name, field_type = field
        else:
            field_name, field_type, default = field
            namespace[field_name] = default
        annotations[field_name] = field_type
    namespace["__annotations__"] = annotations

    command = cast("type[Command]", type(snake_to_pascal(name), (Command,), namespace))
    command.__module__ = __name__
    command.name = name
    return command


def _command_from_signature(func: Callable[..., Any], name: str) -> type[Command]:
    """Synthesise the command class mirroring a handler's parameters.

    ``self`` and any reserved parameter are injected by the runtime, so they are
    left out of the command. Every other parameter becomes a command field
    carrying its annotation and, when present, its default.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    fields: list[tuple[str, Any] | tuple[str, Any, Any]] = []
    for param_name, param in sig.parameters.items():
        if param_name == "self" or param_name in RESERVED_PARAMS:
            continue
        param_type = hints.get(param_name, Any)
        if param.default is inspect.Parameter.empty:
            fields.append((param_name, param_type))
        else:
            fields.append((param_name, param_type, param.default))

    return make_command(name, fields)
