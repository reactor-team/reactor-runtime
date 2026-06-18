"""Binary protobuf codec (wire version v1).

Generic over the generated messages: encoding is ``SerializeToString`` and
decoding is ``ParseFromString``, with the physical channel mapping 1:1 to the
message's logical channel. Evolving the schema needs no change here.
"""

from __future__ import annotations

from reactor_runtime.protocol.base import (
    Channel,
    Codec,
    Direction,
    Message,
    ProtocolVersion,
    logical_channel,
)
from reactor_wire.v1 import control_pb2, data_pb2

_DECODE_TYPES: dict[tuple[Channel, Direction], type[Message]] = {
    (Channel.DATA, Direction.CLIENT): data_pb2.DataClientMessage,
    (Channel.DATA, Direction.SERVER): data_pb2.DataServerMessage,
    (Channel.CONTROL, Direction.CLIENT): control_pb2.ControlClientMessage,
    (Channel.CONTROL, Direction.SERVER): control_pb2.ControlServerMessage,
}


class V1Codec(Codec):
    """Encodes messages as binary protobuf frames."""

    version = ProtocolVersion.V1

    def encode(self, message: Message) -> tuple[Channel, bytes | str]:
        """Serialize a message to a binary frame on its logical channel."""
        return logical_channel(message), message.SerializeToString()

    def decode(self, frame: bytes | str, channel: Channel, direction: Direction) -> Message:
        """Parse a binary frame into the message type for its channel and direction."""
        raw = frame.encode("utf-8") if isinstance(frame, str) else frame
        message = _DECODE_TYPES[(channel, direction)]()
        message.ParseFromString(raw)
        return message
