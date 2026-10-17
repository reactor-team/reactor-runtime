"""The room's own signals, connected to the model by the manifest's ``runtime.livekit`` block.

Beside the Reactor protocol, a room carries signals any participant can send:
RPC calls, which wait for an answer, and data packets on a topic, which do
not. :class:`RoomSignals` reads which of them reach which command, and which
of the model's messages call an RPC method on the participants they reach.

A method name is matched exactly; LiveKit gives a dot in it no meaning. A data
topic matches a listed prefix, a dot, and a command name, and the longest
listed prefix wins. A command an RPC method runs is reachable only through that
method, so a packet that names it is dropped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from reactor_runtime.core import LiveKitConfig


class RoomSignals:
    """Which room signals run which commands, and which messages call which methods.

    Args:
        config: The manifest's ``runtime.livekit`` block, already checked
            against the model.
    """

    def __init__(self, config: LiveKitConfig) -> None:
        self.inbound: dict[str, str] = {
            entry.method: entry.command for entry in config.rpc if entry.command is not None
        }
        """Each RPC method the session's participant registers, and the command it runs."""
        outbound: dict[str, list[str]] = {}
        for entry in config.rpc:
            if entry.message is not None:
                outbound.setdefault(entry.message, []).append(entry.method)
        self.outbound: Mapping[str, tuple[str, ...]] = {
            message: tuple(methods) for message, methods in outbound.items()
        }
        """Each message, and the RPC methods sending it calls."""
        self._rpc_commands = frozenset(self.inbound.values())
        self._prefixes = sorted(config.topics, key=len, reverse=True)

    def topic_command(self, topic: str) -> str | None:
        """Return the command a data packet on *topic* runs, or ``None`` for none.

        ``None`` too for a command an RPC method runs, which no packet reaches.
        """
        for prefix in self._prefixes:
            if topic.startswith(prefix + "."):
                command = topic[len(prefix) + 1 :]
                if not command or command in self._rpc_commands:
                    return None
                return command
        return None


def payload_args(payload: str | bytes) -> dict[str, Any] | None:
    """Read a signal's payload as a command's arguments.

    An empty payload means no arguments. Anything other than a JSON object in
    valid UTF-8 is not a command's arguments.

    Returns:
        The arguments, or ``None`` when the payload is not a JSON object.
    """
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        if not text.strip():
            return {}
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
