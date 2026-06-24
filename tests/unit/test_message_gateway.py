from reactor_runtime.core import Connection, ConnId, InputFrame
from reactor_runtime.message_gateway import InboundCommand, MessageGateway
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.protocol.common import dict_to_struct
from reactor_runtime.protocol.v0.codec import V0Codec
from reactor_runtime.protocol.v1.codec import V1Codec
from reactor_wire.v1 import control_pb2, data_pb2, model_pb2, platform_pb2


class FakeSink:
    def __init__(self) -> None:
        self.keepalives: list[ConnId] = []

    def connection_opened(self, conn: Connection) -> None:
        pass

    def connection_closed(self, conn_id: ConnId) -> None:
        pass

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion
    ) -> None:
        pass

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        pass

    def keepalive(self, conn_id: ConnId) -> None:
        self.keepalives.append(conn_id)


def _gateway() -> tuple[MessageGateway, FakeSink, list[InboundCommand]]:
    sink = FakeSink()
    received: list[InboundCommand] = []

    async def on_command(command: InboundCommand) -> None:
        received.append(command)

    return MessageGateway(sink=sink, on_command=on_command), sink, received


async def test_ping_routes_to_keepalive() -> None:
    gateway, sink, received = _gateway()
    _, frame = V1Codec().encode(control_pb2.ControlClientMessage(ping=platform_pb2.Ping()))
    await gateway.handle(ConnId(7), frame, Channel.CONTROL, ProtocolVersion.V1)
    assert sink.keepalives == [ConnId(7)]
    assert received == []


async def test_command_is_decoded_and_emitted() -> None:
    gateway, sink, received = _gateway()
    message = data_pb2.DataClientMessage(
        request_id="req-1",
        command=model_pb2.Command(type="spawn", data=dict_to_struct({"prompt": "hi"})),
    )
    _, frame = V1Codec().encode(message)
    await gateway.handle(ConnId(3), frame, Channel.DATA, ProtocolVersion.V1)

    assert sink.keepalives == []
    assert received == [
        InboundCommand(
            name="spawn",
            args={"prompt": "hi"},
            uploads={},
            conn_id=ConnId(3),
            request_id="req-1",
        )
    ]


async def test_missing_request_id_is_minted() -> None:
    gateway, _, received = _gateway()
    _, frame = V1Codec().encode(data_pb2.DataClientMessage(command=model_pb2.Command(type="go")))
    await gateway.handle(ConnId(1), frame, Channel.DATA, ProtocolVersion.V1)
    assert received[0].request_id != ""
    assert len(received[0].request_id) >= 16


async def test_upload_references_are_carried_unresolved() -> None:
    gateway, _, received = _gateway()
    command = model_pb2.Command(type="edit")
    command.uploads["image"].upload_id = "u-42"
    _, frame = V1Codec().encode(data_pb2.DataClientMessage(command=command))
    await gateway.handle(ConnId(1), frame, Channel.DATA, ProtocolVersion.V1)
    assert received[0].uploads == {"image": "u-42"}


async def test_other_control_messages_are_not_routed() -> None:
    gateway, sink, received = _gateway()
    _, frame = V1Codec().encode(
        control_pb2.ControlClientMessage(request_schema=platform_pb2.RequestSchema())
    )
    await gateway.handle(ConnId(1), frame, Channel.CONTROL, ProtocolVersion.V1)
    assert sink.keepalives == []
    assert received == []


async def test_undecodable_v1_frame_is_dropped() -> None:
    gateway, sink, received = _gateway()
    await gateway.handle(ConnId(1), b"\xff\xff\xff not protobuf", Channel.DATA, ProtocolVersion.V1)
    assert sink.keepalives == []
    assert received == []


async def test_malformed_v0_json_is_dropped() -> None:
    gateway, sink, received = _gateway()
    await gateway.handle(ConnId(1), "{ not json", Channel.DATA, ProtocolVersion.V0)
    assert sink.keepalives == []
    assert received == []


async def test_decodes_legacy_v0_command_the_same_way() -> None:
    gateway, _, received = _gateway()
    message = data_pb2.DataClientMessage(
        command=model_pb2.Command(type="spawn", data=dict_to_struct({"prompt": "hi"}))
    )
    channel, frame = V0Codec().encode(message)
    await gateway.handle(ConnId(2), frame, channel, ProtocolVersion.V0)

    assert received[0].name == "spawn"
    assert received[0].args == {"prompt": "hi"}
    # Legacy app commands carry no correlation id, so the gateway mints one.
    assert received[0].request_id != ""


async def test_decodes_legacy_v0_ping_routed_by_message_type() -> None:
    gateway, sink, _ = _gateway()
    # v0 places the ping on the data channel; routing keys off the decoded
    # message type, not the physical channel it arrived on.
    channel, frame = V0Codec().encode(control_pb2.ControlClientMessage(ping=platform_pb2.Ping()))
    await gateway.handle(ConnId(5), frame, channel, ProtocolVersion.V0)
    assert sink.keepalives == [ConnId(5)]
