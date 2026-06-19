"""Version-agnostic protocol surface: the codec contract and version helpers.

The canonical message vocabulary is the generated ``reactor_wire.v1``
types. A :class:`Codec` translates between those messages and the bytes on a
physical channel for one wire version; the rest of the system depends only on
this contract, never on a specific version.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from reactor_wire.v1 import control_pb2, data_pb2

# A client->runtime message, on either channel.
ClientMessage = data_pb2.DataClientMessage | control_pb2.ControlClientMessage
# A runtime->client message, on either channel.
ServerMessage = data_pb2.DataServerMessage | control_pb2.ControlServerMessage
Message = ClientMessage | ServerMessage


class ProtocolVersion(Enum):
    """A wire format. Both encode the same ``reactor_wire.v1`` messages."""

    V0 = "v0"
    """Legacy JSON spoken by shipped clients."""
    V1 = "v1"
    """Binary protobuf."""


class Channel(Enum):
    """A physical WebRTC data channel."""

    CONTROL = "control"
    DATA = "data"


class Direction(Enum):
    """Which way a frame travels."""

    CLIENT = "client"
    """Client to runtime."""
    SERVER = "server"
    """Runtime to client."""


class UnsupportedMessageError(ValueError):
    """A message has no representation in the target wire version."""


def logical_channel(message: Message) -> Channel:
    """Return the channel a message belongs to by its type.

    Data-channel messages carry only the model's own command traffic; every
    other message belongs to the control channel.
    """
    if isinstance(message, data_pb2.DataClientMessage | data_pb2.DataServerMessage):
        return Channel.DATA
    return Channel.CONTROL


class Codec(ABC):
    """Translates protocol messages to and from frames for one wire version."""

    version: ProtocolVersion

    @abstractmethod
    def encode(self, message: Message) -> tuple[Channel, bytes | str]:
        """Encode a message into its physical channel and frame.

        The channel is returned because it is version-dependent: v0 places most
        control-plane messages on the data channel, while v1 maps each message
        to its own channel.
        """

    @abstractmethod
    def decode(self, frame: bytes | str, channel: Channel, direction: Direction) -> Message:
        """Decode a frame received on a physical channel travelling a direction.

        Raises:
            UnsupportedMessageError: the frame is not representable in this
                version, or is malformed.
        """

    def encode_outbound(self, message: ServerMessage) -> tuple[Channel, bytes | str]:
        """Encode a runtime-to-client message (the runtime's outbound path)."""
        return self.encode(message)

    def decode_inbound(self, frame: bytes | str, channel: Channel) -> ClientMessage:
        """Decode a client-to-runtime frame (the runtime's inbound path)."""
        message = self.decode(frame, channel, Direction.CLIENT)
        if isinstance(message, data_pb2.DataServerMessage | control_pb2.ControlServerMessage):
            raise UnsupportedMessageError("expected a client message, decoded a server message")
        return message


def select(version: ProtocolVersion) -> Codec:
    """Return the codec for a negotiated wire version."""
    from reactor_runtime.protocol.v0.codec import V0Codec
    from reactor_runtime.protocol.v1.codec import V1Codec

    if version is ProtocolVersion.V0:
        return V0Codec()
    return V1Codec()


def sniff(frame: bytes | str) -> ProtocolVersion:
    """Detect a frame's wire version.

    A text frame, or a binary frame whose first non-space byte opens a JSON
    document, is legacy v0; anything else is binary v1. This is the robustness
    fallback to the authoritative transport-version negotiation.
    """
    if isinstance(frame, str):
        return ProtocolVersion.V0
    if frame.lstrip()[:1] in (b"{", b"["):
        return ProtocolVersion.V0
    return ProtocolVersion.V1
