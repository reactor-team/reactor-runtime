"""The parameters a session starts with, read from the ``/start_session`` body.

Two optional keys shape a session beyond its id. ``starting_input`` holds the
commands the runtime applies before any client connects: ``state`` is a short
form for one ``set_<field>`` command per key, and ``commands`` is an ordered
list of ``{"command", "data"}`` pairs, the same pair a client sends. ``steps``
is how many completed steps the session runs before the runtime closes it.

This module checks the shape of those keys and nothing else. Whether a command
exists, and whether its data fits, is the model contract's decision, made later
on the normal command path. A body without any of the keys starts a session as
it always has.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class InvalidSessionStartError(ValueError):
    """The ``/start_session`` body has a key of the wrong shape.

    Raised before the session moves, so a rejected body leaves the session as
    it was. The message names the key and what it must be.
    """


@dataclass(frozen=True)
class StartingCommand:
    """One command of a starting input, as the caller wrote it.

    Attributes:
        command: The command name.
        data: The command arguments, not yet validated against the contract.
    """

    command: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StartingInput:
    """The commands a session applies before any client connects.

    Attributes:
        state: Values for the model's state fields, keyed by field name.
        commands: The commands to run after the state, in order.
    """

    state: Mapping[str, Any] = field(default_factory=dict)
    commands: tuple[StartingCommand, ...] = ()

    def as_commands(self) -> tuple[StartingCommand, ...]:
        """Return the commands this input runs, in the order they run.

        Each ``state`` key becomes the ``set_<key>`` command the model's state
        generates, with ``{key: value}`` as its data, ahead of ``commands``.
        The ``state`` keys keep the order the body lists them in. That order
        does not change the result, because each generated setter writes one
        field; a caller that needs an order sends those as ``commands``.
        """
        setters = tuple(
            StartingCommand(f"set_{key}", {key: value}) for key, value in self.state.items()
        )
        return setters + self.commands


@dataclass(frozen=True)
class SessionStart:
    """The shape a session starts with.

    Attributes:
        starting_input: The commands to apply before any client connects, or
            ``None`` when the body names none.
        steps: How many completed steps the session runs before the runtime
            closes it, or ``None`` for no limit.
    """

    starting_input: StartingInput | None = None
    steps: int | None = None


def parse_session_start(params: Mapping[str, Any]) -> SessionStart:
    """Read the session's shape from a ``/start_session`` body.

    Keys this module does not own, such as ``session_id``, are ignored. A key
    set to ``null`` reads as absent.

    Args:
        params: The decoded request body.

    Returns:
        The parsed shape, at its defaults for every key the body leaves out.

    Raises:
        InvalidSessionStartError: A key has the wrong shape.
    """
    return SessionStart(
        starting_input=_starting_input(params.get("starting_input")),
        steps=_steps(params.get("steps")),
    )


def _starting_input(raw: Any) -> StartingInput | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise InvalidSessionStartError("starting_input must be an object")
    unknown = sorted(set(raw) - {"state", "commands"})
    if unknown:
        raise InvalidSessionStartError(
            f"starting_input has unknown keys {unknown}; expected 'state' and 'commands'"
        )
    return StartingInput(state=_state(raw.get("state")), commands=_commands(raw.get("commands")))


def _state(raw: Any) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise InvalidSessionStartError("starting_input.state must be an object")
    return dict(raw)


def _commands(raw: Any) -> tuple[StartingCommand, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise InvalidSessionStartError("starting_input.commands must be a list")
    return tuple(_command(index, item) for index, item in enumerate(raw))


def _command(index: int, raw: Any) -> StartingCommand:
    where = f"starting_input.commands[{index}]"
    if not isinstance(raw, Mapping):
        raise InvalidSessionStartError(f"{where} must be an object")
    unknown = sorted(set(raw) - {"command", "data"})
    if unknown:
        raise InvalidSessionStartError(
            f"{where} has unknown keys {unknown}; expected 'command' and 'data'"
        )
    name = raw.get("command")
    if not isinstance(name, str) or not name:
        raise InvalidSessionStartError(f"{where}.command must be a non-empty string")
    data = raw.get("data")
    if data is None:
        return StartingCommand(name)
    if not isinstance(data, Mapping):
        raise InvalidSessionStartError(f"{where}.data must be an object")
    return StartingCommand(name, dict(data))


def _steps(raw: Any) -> int | None:
    if raw is None:
        return None
    # A bool is an int to Python, and `true` is never a step count.
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise InvalidSessionStartError("steps must be a positive integer")
    return raw
