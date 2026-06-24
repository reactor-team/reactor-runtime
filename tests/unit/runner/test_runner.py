import asyncio
from typing import Any, ClassVar

import pytest

from reactor_runtime import protocol
from reactor_runtime.core import (
    ConnectionCapabilities,
    ConnId,
    HealthStatus,
    MediaBundle,
    RuntimeConfig,
    SessionState,
)
from reactor_runtime.model import InputField, ModelMessage, event
from reactor_runtime.model.reactor_core import AddressedSink, BroadcastSink
from reactor_runtime.model.reactor_model import ReactorModel
from reactor_runtime.model.tracks import Output, Video
from reactor_runtime.protocol.common import struct_to_dict
from reactor_runtime.runner.runner import Runner
from reactor_wire.v1 import control_pb2, data_pb2

DATA = protocol.Channel.DATA
CONTROL = protocol.Channel.CONTROL
SERVER = protocol.Direction.SERVER
V0 = protocol.ProtocolVersion.V0


class Greeting(ModelMessage):
    text: str


class FakeOut(Output):
    main: Video


class FakeModel(ReactorModel):
    """A minimal model that records its bring-up order and then idles."""

    output: FakeOut

    created: ClassVar[list["FakeModel"]] = []

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.loaded: dict[str, Any] | None = None
        FakeModel.created.append(self)

    @event(name="set_mode")
    async def set_mode(self, mode: str = InputField(min_length=1)) -> None: ...

    def load(self, config: dict[str, Any]) -> None:
        self.events.append("load")
        self.loaded = config

    def bind_output(self, *, broadcast: BroadcastSink, addressed: AddressedSink) -> None:
        self.events.append("bind")
        super().bind_output(broadcast=broadcast, addressed=addressed)

    def start_thread(self) -> None:
        self.events.append("start")
        super().start_thread()

    async def run(self) -> None:
        await asyncio.sleep(60)


class FakeConnection:
    """A connection that records the frames sent down to it."""

    def __init__(self, cid: int) -> None:
        self.id = ConnId(cid)
        self.capabilities = ConnectionCapabilities(carries_video=True)
        self.protocol_version = V0
        self.sent: list[bytes | str] = []
        self.control: list[bytes | str] = []

    def send_message(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def send_control(self, payload: bytes | str) -> None:
        self.control.append(payload)

    def send_media(self, bundle: MediaBundle) -> None: ...

    def resume_track(self, name: str) -> None: ...

    def pause_track(self, name: str) -> None: ...

    async def close(self) -> None: ...


def _runner() -> Runner:
    return Runner(RuntimeConfig(model_ref="fake:Model"))


async def test_start_resolves_loads_and_readies(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeModel.created.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(RuntimeConfig(model_ref="fake:Model", model_config={"weights": "small"}))

    await runner.start()
    try:
        assert runner._sm.current_state is SessionState.READY
        model = FakeModel.created[-1]
        assert model.loaded == {"weights": "small"}
    finally:
        await runner.stop()


async def test_start_binds_outbound_before_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeModel.created.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = _runner()

    await runner.start()
    try:
        model = FakeModel.created[-1]
        assert model.events == ["load", "bind", "start"]
    finally:
        await runner.stop()


async def test_start_failure_terminates_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ref: str) -> type:
        raise RuntimeError("no such model")

    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", boom)
    runner = _runner()

    await runner.start()

    assert runner._sm.current_state is SessionState.TERMINATED
    assert runner.health().status is HealthStatus.UNHEALTHY


def test_broadcast_encodes_a_model_message_and_fans_it_out() -> None:
    runner = _runner()
    conn = FakeConnection(1)
    runner.connection_opened(conn)

    runner._broadcast_message(Greeting(text="hello"))

    assert len(conn.sent) == 1
    decoded = protocol.select(V0).decode(conn.sent[0], DATA, SERVER)
    assert isinstance(decoded, data_pb2.DataServerMessage)
    assert decoded.message.type == "greeting"
    assert struct_to_dict(decoded.message.data) == {"text": "hello"}


def test_addressed_send_reaches_only_the_target() -> None:
    runner = _runner()
    a, b = FakeConnection(1), FakeConnection(2)
    runner.connection_opened(a)
    runner.connection_opened(b)

    runner._send_addressed(ConnId(2), Greeting(text="for-b"), request_id=None)

    assert a.sent == []
    assert len(b.sent) == 1


def _decode_control_reply(frame: bytes | str) -> control_pb2.ControlServerMessage:
    decoded = protocol.select(V0).decode(frame, CONTROL, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    return decoded


def test_publish_request_grants_and_replies_on_the_control_channel() -> None:
    runner = _runner()
    conn = FakeConnection(1)
    runner.connection_opened(conn)

    runner.publish_requested(ConnId(1), "webcam", "ctrl_5")

    assert len(conn.control) == 1
    reply = _decode_control_reply(conn.control[0])
    assert reply.request_id == "ctrl_5"
    assert reply.WhichOneof("payload") == "publish_track"


def test_publish_request_for_a_held_track_is_refused() -> None:
    runner = _runner()
    a, b = FakeConnection(1), FakeConnection(2)
    runner.connection_opened(a)
    runner.connection_opened(b)

    runner.publish_requested(ConnId(1), "webcam", "ctrl_1")
    runner.publish_requested(ConnId(2), "webcam", "ctrl_2")

    granted = _decode_control_reply(a.control[0])
    refused = _decode_control_reply(b.control[0])
    assert granted.WhichOneof("payload") == "publish_track"
    assert refused.request_id == "ctrl_2"
    assert refused.WhichOneof("payload") == "error"
