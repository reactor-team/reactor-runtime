import logging
from collections.abc import Mapping

import pytest

from reactor_runtime.core import Connection, ConnId, InputFrame
from reactor_runtime.message_gateway import InboundCommand, MessageGateway
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.protocol.common import dict_to_struct
from reactor_runtime.protocol.v0.codec import V0Codec
from reactor_runtime.protocol.v1.codec import V1Codec
from reactor_wire.v1 import common_pb2, control_pb2, data_pb2, model_pb2, platform_pb2, track_pb2


class FakeSink:
    def __init__(self) -> None:
        self.keepalives: list[ConnId] = []
        self.resumed: list[tuple[ConnId, str]] = []
        self.paused: list[tuple[ConnId, str]] = []
        self.published: list[tuple[ConnId, str, str]] = []
        self.unpublished: list[tuple[ConnId, str]] = []
        self.schema_requests: list[tuple[ConnId, str]] = []
        self.uploads: list[tuple[ConnId, str]] = []
        self.clips: list[tuple[ConnId, float, str]] = []
        self.recordings: list[tuple[ConnId, str]] = []

    def connection_opened(self, conn: Connection) -> None:
        pass

    def connection_closed(self, conn_id: ConnId) -> None:
        pass

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        pass

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        pass

    def keepalive(self, conn_id: ConnId) -> None:
        self.keepalives.append(conn_id)

    def resume_track(self, conn_id: ConnId, name: str) -> None:
        self.resumed.append((conn_id, name))

    def pause_track(self, conn_id: ConnId, name: str) -> None:
        self.paused.append((conn_id, name))

    def publish_requested(self, conn_id: ConnId, name: str, request_id: str) -> None:
        self.published.append((conn_id, name, request_id))

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        self.unpublished.append((conn_id, name))

    def file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        self.uploads.append((conn_id, upload_id))

    def schema_requested(self, conn_id: ConnId, request_id: str) -> None:
        self.schema_requests.append((conn_id, request_id))

    def clip_requested(self, conn_id: ConnId, duration_seconds: float, request_id: str) -> None:
        self.clips.append((conn_id, duration_seconds, request_id))

    def recording_requested(self, conn_id: ConnId, request_id: str) -> None:
        self.recordings.append((conn_id, request_id))

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        pass


# The arrival stamp rides the frame untouched, so one fixed reading is enough
# to assert that the gateway carries it onto the command it decodes.
_ARRIVED = 1000.0


def _gateway() -> tuple[MessageGateway, FakeSink, list[InboundCommand]]:
    sink = FakeSink()
    received: list[InboundCommand] = []

    async def on_command(command: InboundCommand) -> None:
        received.append(command)

    return MessageGateway(sink=sink, on_command=on_command), sink, received


async def test_ping_routes_to_keepalive() -> None:
    gateway, sink, received = _gateway()
    _, frame = V1Codec().encode(control_pb2.ControlClientMessage(ping=platform_pb2.Ping()))
    await gateway.handle(
        ConnId(7), frame, Channel.CONTROL, ProtocolVersion.V1, received_at=_ARRIVED
    )
    assert sink.keepalives == [ConnId(7)]
    assert received == []


async def test_command_is_decoded_and_emitted() -> None:
    gateway, sink, received = _gateway()
    message = data_pb2.DataClientMessage(
        request_id="req-1",
        command=model_pb2.Command(type="spawn", data=dict_to_struct({"prompt": "hi"})),
    )
    _, frame = V1Codec().encode(message)
    await gateway.handle(ConnId(3), frame, Channel.DATA, ProtocolVersion.V1, received_at=_ARRIVED)

    assert sink.keepalives == []
    assert received == [
        InboundCommand(
            name="spawn",
            args={"prompt": "hi"},
            uploads={},
            conn_id=ConnId(3),
            request_id="req-1",
            received_at=_ARRIVED,
        )
    ]


async def test_missing_request_id_is_minted() -> None:
    gateway, _, received = _gateway()
    _, frame = V1Codec().encode(data_pb2.DataClientMessage(command=model_pb2.Command(type="go")))
    await gateway.handle(ConnId(1), frame, Channel.DATA, ProtocolVersion.V1, received_at=_ARRIVED)
    assert received[0].request_id != ""
    assert len(received[0].request_id) >= 16


async def test_upload_references_are_carried_unresolved() -> None:
    gateway, _, received = _gateway()
    command = model_pb2.Command(type="edit")
    command.uploads["image"].upload_id = "u-42"
    _, frame = V1Codec().encode(data_pb2.DataClientMessage(command=command))
    await gateway.handle(ConnId(1), frame, Channel.DATA, ProtocolVersion.V1, received_at=_ARRIVED)
    assert received[0].uploads == {"image": "u-42"}


async def test_other_control_messages_are_not_routed() -> None:
    gateway, sink, received = _gateway()
    _, frame = V1Codec().encode(
        control_pb2.ControlClientMessage(error=common_pb2.Error(code="boom"))
    )
    await gateway.handle(
        ConnId(1), frame, Channel.CONTROL, ProtocolVersion.V1, received_at=_ARRIVED
    )
    assert sink.keepalives == []
    assert received == []


async def test_request_clip_routes_with_its_correlation_id() -> None:
    gateway, sink, _ = _gateway()
    _, frame = V1Codec().encode(
        control_pb2.ControlClientMessage(
            request_id="ctrl_clip",
            request_clip=platform_pb2.RequestClip(duration_seconds=30.0),
        )
    )
    await gateway.handle(
        ConnId(8), frame, Channel.CONTROL, ProtocolVersion.V1, received_at=_ARRIVED
    )
    assert sink.clips == [(ConnId(8), 30.0, "ctrl_clip")]


async def test_request_recording_routes_with_its_correlation_id() -> None:
    gateway, sink, _ = _gateway()
    _, frame = V1Codec().encode(
        control_pb2.ControlClientMessage(
            request_id="ctrl_rec", request_recording=platform_pb2.RequestRecording()
        )
    )
    await gateway.handle(
        ConnId(8), frame, Channel.CONTROL, ProtocolVersion.V1, received_at=_ARRIVED
    )
    assert sink.recordings == [(ConnId(8), "ctrl_rec")]


async def test_v0_clip_request_mints_a_correlation_id_off_the_data_channel() -> None:
    gateway, sink, _ = _gateway()
    # The shipped client sends clip requests on the data channel with no id and
    # correlates by receipt order; the gateway mints one for the runtime side.
    channel, frame = V0Codec().encode(
        control_pb2.ControlClientMessage(
            request_clip=platform_pb2.RequestClip(duration_seconds=5.0)
        )
    )
    await gateway.handle(ConnId(9), frame, channel, ProtocolVersion.V0, received_at=_ARRIVED)
    assert len(sink.clips) == 1
    conn_id, duration, request_id = sink.clips[0]
    assert (conn_id, duration) == (ConnId(9), 5.0)
    assert request_id != ""


async def test_request_schema_routes_with_its_correlation_id() -> None:
    gateway, sink, _ = _gateway()
    _, frame = V1Codec().encode(
        control_pb2.ControlClientMessage(
            request_id="ctrl_7", request_schema=platform_pb2.RequestSchema()
        )
    )
    await gateway.handle(
        ConnId(8), frame, Channel.CONTROL, ProtocolVersion.V1, received_at=_ARRIVED
    )
    assert sink.schema_requests == [(ConnId(8), "ctrl_7")]


async def test_v0_request_schema_routes_off_the_data_channel() -> None:
    gateway, sink, _ = _gateway()
    # v0 carries requestSchema on the data channel and with no correlation id.
    channel, frame = V0Codec().encode(
        control_pb2.ControlClientMessage(request_schema=platform_pb2.RequestSchema())
    )
    await gateway.handle(ConnId(9), frame, channel, ProtocolVersion.V0, received_at=_ARRIVED)
    assert sink.schema_requests == [(ConnId(9), "")]


async def test_v0_request_capabilities_is_dropped_without_a_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway, sink, _ = _gateway()
    frame = '{"scope": "runtime", "data": {"type": "requestCapabilities", "data": {}}}'
    with caplog.at_level(logging.WARNING, logger="reactor_runtime.message_gateway"):
        await gateway.handle(
            ConnId(9), frame, Channel.DATA, ProtocolVersion.V0, received_at=_ARRIVED
        )

    assert sink.schema_requests == []
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.exc_info is None
    message = record.getMessage()
    assert message.startswith("MessageGateway dropped an undecodable frame on Channel.DATA: ")
    assert "requestCapabilities" in message


async def test_undecodable_v1_frame_is_dropped() -> None:
    gateway, sink, received = _gateway()
    await gateway.handle(
        ConnId(1),
        b"\xff\xff\xff not protobuf",
        Channel.DATA,
        ProtocolVersion.V1,
        received_at=_ARRIVED,
    )
    assert sink.keepalives == []
    assert received == []


async def test_malformed_v0_json_is_dropped() -> None:
    gateway, sink, received = _gateway()
    await gateway.handle(
        ConnId(1), "{ not json", Channel.DATA, ProtocolVersion.V0, received_at=_ARRIVED
    )
    assert sink.keepalives == []
    assert received == []


async def test_decodes_legacy_v0_command_the_same_way() -> None:
    gateway, _, received = _gateway()
    message = data_pb2.DataClientMessage(
        command=model_pb2.Command(type="spawn", data=dict_to_struct({"prompt": "hi"}))
    )
    channel, frame = V0Codec().encode(message)
    await gateway.handle(ConnId(2), frame, channel, ProtocolVersion.V0, received_at=_ARRIVED)

    assert received[0].name == "spawn"
    assert received[0].args == {"prompt": "hi"}
    # Legacy app commands carry no correlation id, so the gateway mints one.
    assert received[0].request_id != ""


async def test_decodes_legacy_v0_ping_routed_by_message_type() -> None:
    gateway, sink, _ = _gateway()
    # v0 places the ping on the data channel; routing keys off the decoded
    # message type, not the physical channel it arrived on.
    channel, frame = V0Codec().encode(control_pb2.ControlClientMessage(ping=platform_pb2.Ping()))
    await gateway.handle(ConnId(5), frame, channel, ProtocolVersion.V0, received_at=_ARRIVED)
    assert sink.keepalives == [ConnId(5)]


async def test_publish_track_request_routes_with_its_correlation_id() -> None:
    gateway, sink, _ = _gateway()
    channel, frame = V0Codec().encode(
        control_pb2.ControlClientMessage(
            request_id="ctrl_9", publish_track=track_pb2.PublishTrack(name="webcam")
        )
    )
    await gateway.handle(ConnId(3), frame, channel, ProtocolVersion.V0, received_at=_ARRIVED)
    assert sink.published == [(ConnId(3), "webcam", "ctrl_9")]


async def test_resume_and_pause_notifications_route_to_the_sink() -> None:
    gateway, sink, _ = _gateway()
    _, resume = V0Codec().encode(
        control_pb2.ControlClientMessage(resume_track=track_pb2.ResumeTrack(name="main_video"))
    )
    _, pause = V0Codec().encode(
        control_pb2.ControlClientMessage(pause_track=track_pb2.PauseTrack(name="main_audio"))
    )
    await gateway.handle(
        ConnId(4), resume, Channel.CONTROL, ProtocolVersion.V0, received_at=_ARRIVED
    )
    await gateway.handle(
        ConnId(4), pause, Channel.CONTROL, ProtocolVersion.V0, received_at=_ARRIVED
    )
    assert sink.resumed == [(ConnId(4), "main_video")]
    assert sink.paused == [(ConnId(4), "main_audio")]


async def test_file_uploaded_notification_routes_to_the_sink() -> None:
    gateway, sink, _ = _gateway()
    channel, frame = V0Codec().encode(
        control_pb2.ControlClientMessage(
            file_uploaded=platform_pb2.FileUploaded(
                upload_id="u-7", name="cat.png", mime_type="image/png", size=4
            )
        )
    )
    await gateway.handle(ConnId(2), frame, channel, ProtocolVersion.V0, received_at=_ARRIVED)
    assert sink.uploads == [(ConnId(2), "u-7")]


async def test_unpublish_notification_routes_to_the_sink() -> None:
    gateway, sink, _ = _gateway()
    channel, frame = V0Codec().encode(
        control_pb2.ControlClientMessage(unpublish_track=track_pb2.UnpublishTrack(name="webcam"))
    )
    await gateway.handle(ConnId(6), frame, channel, ProtocolVersion.V0, received_at=_ARRIVED)
    assert sink.unpublished == [(ConnId(6), "webcam")]
