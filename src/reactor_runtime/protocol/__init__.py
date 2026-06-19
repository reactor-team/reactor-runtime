"""Wire protocol: typed messages, version codecs, and version detection.

The runtime speaks the generated ``reactor_wire.v1`` messages natively. Two
wire versions encode them: ``v1`` (binary protobuf, the native surface) and
``v0`` (the frozen legacy JSON the shipped clients speak). Pick a codec with
:func:`select`, or detect a frame's version with :func:`sniff`.
"""

from reactor_runtime.protocol.base import (
    Channel,
    ClientMessage,
    Codec,
    Direction,
    Message,
    ProtocolVersion,
    ServerMessage,
    UnsupportedMessageError,
    logical_channel,
    select,
    sniff,
)

__all__ = [
    "Channel",
    "ClientMessage",
    "Codec",
    "Direction",
    "Message",
    "ProtocolVersion",
    "ServerMessage",
    "UnsupportedMessageError",
    "logical_channel",
    "select",
    "sniff",
]
