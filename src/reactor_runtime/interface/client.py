"""Per-client handle — :class:`ClientInfo`.

The handle a model receives when an inbound handler declares a reserved
``client`` parameter. It identifies one connected client and carries an
addressed send, so a handler can reply to the client it is serving rather than
broadcasting to all of them. It is built by the runtime per connection, never
deserialised off the wire, so a client cannot set or spoof it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from reactor_runtime.core.values import ConnId
from reactor_runtime.interface.events.messages import ModelMessage


@dataclass(frozen=True)
class ClientInfo:
    """Identity and message handle for one connected client.

    Attributes:
        id: The connection id, stable for the lifetime of this connection.
        joined_at: Monotonic seconds (``time.monotonic()``) captured when the
            client connected.
    """

    id: ConnId
    joined_at: float
    _send: Callable[[ModelMessage], None] | None = field(default=None, compare=False, repr=False)

    async def send(self, message: ModelMessage) -> None:
        """Deliver a typed message to this client only.

        Routes to the addressed outbound sink as an unsolicited push — no
        request correlation. Falls back to a no-op when no sink is bound, so
        model code can address a client uniformly.

        Args:
            message: The typed message to deliver to this client.
        """
        if self._send is not None:
            self._send(message)
