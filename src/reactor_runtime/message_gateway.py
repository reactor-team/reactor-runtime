"""The inbound message router.

The one place where "what does this payload mean" lives. A payload that arrived
on any connection is decoded here through the wire :class:`~reactor_runtime.protocol.Codec`
— the single seam the JSON-to-protobuf swap touches — and routed by what it is:
a ping marks liveness, a command is handed onward as a decoded
:class:`InboundCommand` whose ``request_id`` is guaranteed present.

The gateway is transport-neutral: a WebSocket frame and a WebRTC data-channel
frame decode the same way. It does no validation — that belongs to the model
contract — and does not resolve uploads or submit to the model: it decodes and
emits, and the handler it is given owns what happens next.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from google.protobuf.message import DecodeError

from reactor_runtime.core import ConnectionSink, ConnId
from reactor_runtime.protocol import Channel, Codec
from reactor_runtime.protocol.common import struct_to_dict
from reactor_wire.v1 import control_pb2, data_pb2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InboundCommand:
    """A decoded client command, ready for upload resolution and submission.

    Attributes:
        name: The command name the client invoked.
        args: The command arguments as a plain mapping, exactly as decoded — no
            validation or coercion has been applied.
        uploads: Argument name to the ``upload_id`` it references, still
            unresolved. The handler fetches the bytes for these.
        conn_id: The connection the command arrived on.
        request_id: The client's correlation id, guaranteed present — carried
            from the client when supplied, minted here only when absent.
    """

    name: str
    args: Mapping[str, Any]
    uploads: Mapping[str, str]
    conn_id: ConnId
    request_id: str


CommandHandler = Callable[[InboundCommand], Awaitable[None]]
"""Receives each decoded command. Owns upload resolution and submission."""


class MessageGateway:
    """Decodes inbound frames and routes them by what they mean.

    Holds the wire codec and the upward sink. Pings mark liveness through the
    sink; commands are decoded into an :class:`InboundCommand` and passed to the
    handler. Other control messages decode cleanly but are not routed here — the
    components that serve them are wired by a later layer.
    """

    def __init__(
        self,
        *,
        sink: ConnectionSink,
        codec: Codec,
        on_command: CommandHandler,
    ) -> None:
        """Bind the gateway to its codec, upward sink, and command handler."""
        self._sink = sink
        self._codec = codec
        self._on_command = on_command

    async def handle(self, conn_id: ConnId, payload: bytes | str, channel: Channel) -> None:
        """Decode one inbound frame and route it.

        A ping notes liveness on the sink. A command is handed to the handler as
        an :class:`InboundCommand` with a guaranteed ``request_id``. Any other
        decoded message is left for a later layer to route. A frame that cannot
        be decoded is dropped with a warning rather than raised, so one malformed
        frame from any client cannot break the transport read loop.
        """
        try:
            message = self._codec.decode_inbound(payload, channel)
        except (ValueError, DecodeError):
            logger.warning(
                "MessageGateway dropped an undecodable frame on %s", channel, exc_info=True
            )
            return

        if (
            isinstance(message, control_pb2.ControlClientMessage)
            and message.WhichOneof("payload") == "ping"
        ):
            self._sink.keepalive(conn_id)
            return

        if (
            isinstance(message, data_pb2.DataClientMessage)
            and message.WhichOneof("payload") == "command"
        ):
            command = message.command
            await self._on_command(
                InboundCommand(
                    name=command.type,
                    args=struct_to_dict(command.data),
                    uploads={param: ref.upload_id for param, ref in command.uploads.items()},
                    conn_id=conn_id,
                    request_id=message.request_id or _new_request_id(),
                )
            )
            return

        logger.debug("MessageGateway received an unrouted message on %s", channel)


def _new_request_id() -> str:
    """Mint a fallback correlation id for a client that supplied none."""
    return uuid.uuid4().hex
