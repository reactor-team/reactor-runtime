from reactor_runtime import protocol
from reactor_runtime.protocol.common import struct_to_dict
from reactor_wire.v1 import common_pb2, data_pb2

K = common_pb2.MessageKind
SERVER = protocol.Direction.SERVER
DATA = protocol.Channel.DATA


def _decode_server(codec: protocol.Codec, frame: bytes | str) -> data_pb2.DataServerMessage:
    """Decode a frame the codec produced back into the server message it carries."""
    message = codec.decode(frame, DATA, SERVER)
    assert isinstance(message, data_pb2.DataServerMessage)
    return message


def test_v1_broadcast_round_trips_name_and_data() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V1)

    channel, frame = codec.encode_model_message("current_mode", {"mode": "turbo", "intensity": 3})

    assert channel is DATA
    decoded = _decode_server(codec, frame)
    assert decoded.message.type == "current_mode"
    assert struct_to_dict(decoded.message.data) == {"mode": "turbo", "intensity": 3}
    assert decoded.kind == K.MESSAGE_KIND_NOTIFICATION
    assert decoded.request_id == ""


def test_v0_broadcast_round_trips_name_and_data() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V0)

    channel, frame = codec.encode_model_message("current_mode", {"mode": "turbo"})

    assert channel is DATA
    assert isinstance(frame, str)  # a legacy text frame
    decoded = _decode_server(codec, frame)
    assert decoded.message.type == "current_mode"
    assert struct_to_dict(decoded.message.data) == {"mode": "turbo"}


def test_a_reply_carries_its_request_id_as_a_response() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V1)

    _channel, frame = codec.encode_model_message("ack", {}, request_id="req-1")

    decoded = _decode_server(codec, frame)
    assert decoded.request_id == "req-1"
    assert decoded.kind == K.MESSAGE_KIND_RESPONSE


def test_a_broadcast_is_a_notification_with_no_request_id() -> None:
    codec = protocol.select(protocol.ProtocolVersion.V1)

    _channel, frame = codec.encode_model_message("status", {"ok": True})

    decoded = _decode_server(codec, frame)
    assert decoded.request_id == ""
    assert decoded.kind == K.MESSAGE_KIND_NOTIFICATION
