from collections.abc import Mapping

import numpy as np

from reactor_runtime.core import (
    Connection,
    ConnectionCapabilities,
    ConnectionSink,
    ConnId,
    InputFrame,
    MediaBundle,
)
from reactor_runtime.protocol import Channel, ProtocolVersion


class FakeConnection:
    def __init__(self, conn_id: int) -> None:
        self.id = ConnId(conn_id)
        self.capabilities = ConnectionCapabilities(carries_video=True)
        self.protocol_version = ProtocolVersion.V0
        self.sent: list[bytes | str] = []
        self.control: list[bytes | str] = []
        self.paused: list[str] = []
        self.closed = False

    def send_message(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def send_media(self, bundle: MediaBundle) -> None:
        pass

    def resume_track(self, name: str) -> None:
        pass

    def pause_track(self, name: str) -> None:
        self.paused.append(name)

    def send_control(self, payload: bytes | str) -> None:
        self.control.append(payload)

    async def close(self) -> None:
        self.closed = True


class FakeSink:
    def __init__(self) -> None:
        self.opened: list[ConnId] = []
        self.messages: list[tuple[ConnId, bytes | str, Channel]] = []
        self.answers: list[tuple[ConnId, dict[str, str]]] = []

    def connection_opened(self, conn: Connection) -> None:
        self.opened.append(conn.id)

    def connection_closed(self, conn_id: ConnId) -> None:
        pass

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        self.messages.append((conn_id, payload, channel))

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        pass

    def keepalive(self, conn_id: ConnId) -> None:
        pass

    def resume_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def pause_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def publish_requested(self, conn_id: ConnId, name: str, request_id: str) -> None:
        pass

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        pass

    def file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        pass

    def schema_requested(self, conn_id: ConnId, request_id: str) -> None:
        pass

    def clip_requested(self, conn_id: ConnId, duration_seconds: float, request_id: str) -> None:
        pass

    def recording_requested(self, conn_id: ConnId, request_id: str) -> None:
        pass

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        self.answers.append((conn_id, dict(answer)))


def test_fake_connection_conforms_by_shape() -> None:
    assert isinstance(FakeConnection(1), Connection)


def test_fake_sink_conforms_by_shape() -> None:
    assert isinstance(FakeSink(), ConnectionSink)


def test_a_plain_object_does_not_conform() -> None:
    assert not isinstance(object(), Connection)
    assert not isinstance(object(), ConnectionSink)


def test_facts_flow_up_through_the_sink() -> None:
    sink: ConnectionSink = FakeSink()
    conn: Connection = FakeConnection(7)

    sink.connection_answered(conn.id, {"type": "answer", "sdp": "v=0..."})
    sink.connection_opened(conn)
    sink.message_received(conn.id, b"hello", ProtocolVersion.V1, Channel.DATA)
    sink.message_received(conn.id, '{"type":"notification"}', ProtocolVersion.V0, Channel.CONTROL)
    sink.media_received(conn.id, "camera", InputFrame(np.zeros((1, 1, 3), dtype=np.uint8)))

    assert isinstance(sink, FakeSink)
    assert sink.opened == [ConnId(7)]
    assert sink.answers == [(ConnId(7), {"type": "answer", "sdp": "v=0..."})]
    assert sink.messages == [
        (ConnId(7), b"hello", Channel.DATA),
        (ConnId(7), '{"type":"notification"}', Channel.CONTROL),
    ]


def test_commands_flow_down_through_the_connection() -> None:
    conn = FakeConnection(3)
    conn.send_message(b"frame")
    conn.send_message('{"type":"current_mode"}')
    conn.pause_track("main_video")
    conn.send_control('{"type":"response"}')
    assert conn.sent == [b"frame", '{"type":"current_mode"}']
    assert conn.paused == ["main_video"]
    assert conn.control == ['{"type":"response"}']
