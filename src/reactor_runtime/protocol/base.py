"""Version-agnostic protocol surface: the codec contract and version helpers.

The canonical message vocabulary is the generated ``reactor_wire.v1``
types. A :class:`Codec` translates between those messages and the bytes on a
physical channel for one wire version; the rest of the system depends only on
this contract, never on a specific version.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Any

from reactor_runtime.protocol.common import dict_to_struct
from reactor_wire.v1 import common_pb2, control_pb2, data_pb2, model_pb2, platform_pb2, track_pb2

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

    def encode_model_message(
        self,
        type_name: str,
        data: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> tuple[Channel, bytes | str]:
        """Encode a model's outbound message into a server frame.

        Wraps a model message — its name and already-serialised payload — into a
        ``DataServerMessage`` and encodes it for this wire version, so every
        codec shares one model-message-to-frame path. A reply carries its
        ``request_id`` and is a response; a broadcast carries none and is a
        notification, the correlation applied here at send time rather than by
        the model.
        """
        message = data_pb2.DataServerMessage(
            request_id=request_id or "",
            kind=(
                common_pb2.MessageKind.MESSAGE_KIND_RESPONSE
                if request_id is not None
                else common_pb2.MessageKind.MESSAGE_KIND_NOTIFICATION
            ),
            message=model_pb2.ModelMessage(type=type_name, data=dict_to_struct(data)),
        )
        return self.encode(message)

    def encode_command_ack(self, request_id: str) -> tuple[Channel, bytes | str]:
        """Encode a bodyless acknowledgement that a command was processed.

        Correlated by *request_id* so a client's awaited command resolves once
        its handler has run and returned no message, rather than waiting for a
        reply that never comes.
        """
        message = data_pb2.DataServerMessage(
            request_id=request_id,
            kind=common_pb2.MessageKind.MESSAGE_KIND_RESPONSE,
        )
        return self.encode(message)

    def encode_command_error(
        self, request_id: str, code: str, detail: str
    ) -> tuple[Channel, bytes | str]:
        """Encode a command's failure, correlated by *request_id*.

        Carries a short *code* and human-readable *detail* so a client's awaited
        command rejects with a reason instead of timing out — whether the command
        failed contract validation or its handler raised.
        """
        message = data_pb2.DataServerMessage(
            request_id=request_id,
            kind=common_pb2.MessageKind.MESSAGE_KIND_RESPONSE,
            error=common_pb2.Error(code=code, message=detail),
        )
        return self.encode(message)

    def encode_publish_response(
        self, request_id: str, *, granted: bool, reason: str = ""
    ) -> tuple[Channel, bytes | str]:
        """Encode the runtime's reply to a client's publish-track request.

        A grant carries an empty :class:`PublishTrackResponse`; a refusal carries
        an error with a reason. Correlated by *request_id* so the client matches
        the reply to its request, and encoded for this wire version so every
        codec shares one publish-response-to-frame path.
        """
        if granted:
            message = control_pb2.ControlServerMessage(
                request_id=request_id,
                kind=common_pb2.MessageKind.MESSAGE_KIND_RESPONSE,
                publish_track=track_pb2.PublishTrackResponse(),
            )
        else:
            message = control_pb2.ControlServerMessage(
                request_id=request_id,
                kind=common_pb2.MessageKind.MESSAGE_KIND_RESPONSE,
                error=common_pb2.Error(
                    code="publish_refused", message=reason or "track already published"
                ),
            )
        return self.encode(message)

    def encode_schema_response(
        self, request_id: str, openapi: Mapping[str, Any]
    ) -> tuple[Channel, bytes | str]:
        """Encode the runtime's reply to a client's schema request.

        Wraps the rendered OpenAPI document in a ``ControlServerMessage`` and
        encodes it for this wire version, correlated by *request_id*. The
        physical channel is version-dependent — v0 carries the schema on the data
        channel, v1 on the control channel — and is returned alongside the frame.
        """
        message = control_pb2.ControlServerMessage(
            request_id=request_id,
            kind=common_pb2.MessageKind.MESSAGE_KIND_RESPONSE,
            model_schema=platform_pb2.ModelSchema(openapi=dict_to_struct(openapi)),
        )
        return self.encode(message)

    def encode_clip_ready(
        self, request_id: str, clip: Mapping[str, Any]
    ) -> tuple[Channel, bytes | str]:
        """Encode the runtime's reply to a client's clip or recording request.

        Wraps the clip descriptor — its recording id, kind, marker range, ready
        estimate, and playlist URL — in a ``ControlServerMessage`` and encodes it
        for this wire version, correlated by *request_id*. The physical channel is
        version-dependent (v0 carries the reply on the data channel, v1 on the
        control channel) and is returned alongside the frame.
        """
        message = control_pb2.ControlServerMessage(
            request_id=request_id,
            kind=common_pb2.MessageKind.MESSAGE_KIND_RESPONSE,
            clip_ready=platform_pb2.ClipReady(
                session_id=str(clip["session_id"]),
                kind=str(clip["kind"]),
                start_marker=float(clip["start_marker"]),
                end_marker=float(clip["end_marker"]),
                now_marker=float(clip["now_marker"]),
                predicted_ready_at_ms=int(clip["predicted_ready_at_ms"]),
                playlist_url=str(clip["playlist_url"]),
            ),
        )
        return self.encode(message)

    def encode_clip_failed(self, request_id: str, reason: str) -> tuple[Channel, bytes | str]:
        """Encode the runtime's refusal of a clip or recording request.

        Carries the human-readable *reason* in a ``ControlServerMessage``,
        correlated by *request_id* and encoded for this wire version.
        """
        message = control_pb2.ControlServerMessage(
            request_id=request_id,
            kind=common_pb2.MessageKind.MESSAGE_KIND_RESPONSE,
            clip_failed=platform_pb2.ClipFailed(reason=reason),
        )
        return self.encode(message)


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
