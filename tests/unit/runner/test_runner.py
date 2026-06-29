import asyncio
from pathlib import Path
from typing import Any

import pytest

from reactor_runtime import InputField, ModelMessage, Output, ReactorModel, Video, event, protocol
from reactor_runtime.core import (
    ConnectionCapabilities,
    ConnId,
    HealthStatus,
    MediaBundle,
    RuntimeConfig,
    SessionEvent,
    SessionState,
    TransitionEvent,
)
from reactor_runtime.interface.internal.reactor_core import AddressedSink, BroadcastSink
from reactor_runtime.protocol.common import struct_to_dict
from reactor_runtime.runner.runner import SESSION_ID, Runner
from reactor_runtime.transport.router import (
    SessionControl,
    SessionNotRunningError,
    UnknownSessionError,
)
from reactor_wire.v1 import control_pb2, data_pb2

DATA = protocol.Channel.DATA
CONTROL = protocol.Channel.CONTROL
SERVER = protocol.Direction.SERVER
V0 = protocol.ProtocolVersion.V0
V1 = protocol.ProtocolVersion.V1


class Greeting(ModelMessage):
    text: str


class FakeOut(Output):
    main: Video


class FakeModel(ReactorModel):
    """A minimal model that records its bring-up order and then idles."""

    output: FakeOut

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.loaded: Path | None = None
        created_models.append(self)

    @event(name="set_mode")
    async def set_mode(self, mode: str = InputField(min_length=1)) -> None: ...

    def load(self, config_path: Path | None) -> None:
        self.events.append("load")
        self.loaded = config_path

    def bind_output(self, *, broadcast: BroadcastSink, addressed: AddressedSink) -> None:
        self.events.append("bind")
        super().bind_output(broadcast=broadcast, addressed=addressed)

    def start_thread(self) -> None:
        self.events.append("start")
        super().start_thread()

    async def run(self) -> None:
        await asyncio.sleep(60)


# Instances FakeModel records as it is constructed, so a test can inspect the
# model the runner built. Kept off the model class: an annotation on the model
# would be read as part of its contract.
created_models: list[FakeModel] = []


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


@pytest.fixture
async def started_runner(monkeypatch: pytest.MonkeyPatch) -> Any:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = _runner()
    await runner.start()
    try:
        yield runner
    finally:
        await runner.stop()


async def test_start_resolves_loads_and_readies(monkeypatch: pytest.MonkeyPatch) -> None:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(RuntimeConfig(model_ref="fake:Model", config_path=Path("/cfg/config.yml")))

    await runner.start()
    try:
        assert runner._sm.current_state is SessionState.READY
        model = created_models[-1]
        assert model.loaded == Path("/cfg/config.yml")
    finally:
        await runner.stop()


async def test_start_binds_outbound_before_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = _runner()

    await runner.start()
    try:
        model = created_models[-1]
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


async def _started_runner(monkeypatch: pytest.MonkeyPatch) -> Runner:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = _runner()
    await runner.start()
    return runner


async def test_schema_request_v0_replies_on_the_data_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = await _started_runner(monkeypatch)
    try:
        conn = FakeConnection(1)
        runner.connection_opened(conn)
        runner.schema_requested(ConnId(1), "ctrl_3")
        decoded = protocol.select(V0).decode(conn.sent[0], DATA, SERVER)
        assert isinstance(decoded, control_pb2.ControlServerMessage)
        assert decoded.WhichOneof("payload") == "model_schema"
    finally:
        await runner.stop()


async def test_schema_request_v1_replies_on_control_correlated_by_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = await _started_runner(monkeypatch)
    try:
        conn = FakeConnection(2)
        conn.protocol_version = V1
        runner.connection_opened(conn)
        runner.schema_requested(ConnId(2), "ctrl_9")
        decoded = protocol.select(V1).decode(conn.control[0], CONTROL, SERVER)
        assert isinstance(decoded, control_pb2.ControlServerMessage)
        assert decoded.WhichOneof("payload") == "model_schema"
        assert decoded.request_id == "ctrl_9"
    finally:
        await runner.stop()


# --- the session-control face --------------------------------------------


def test_runner_satisfies_the_session_control_surface() -> None:
    runner = _runner()
    control: SessionControl = runner
    assert isinstance(control, SessionControl)


def test_new_conn_id_is_unique() -> None:
    runner = _runner()
    ids = {runner.new_conn_id() for _ in range(5)}
    assert len(ids) == 5


def test_require_session_running_raises_when_idle() -> None:
    runner = _runner()
    with pytest.raises(SessionNotRunningError):
        runner.require_session_running(SESSION_ID)


def test_connection_opened_registers_the_connection() -> None:
    runner = _runner()
    runner.connection_opened(FakeConnection(1))
    assert runner._connections.count == 1
    runner.connection_closed(ConnId(1))
    assert runner._connections.count == 0


async def test_connection_answered_rides_a_self_loop_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = await _started_runner(monkeypatch)
    try:
        runner.start_session({})
        stream = runner._events.subscribe()
        runner.connection_answered(ConnId(1), {"type": "answer", "sdp": "v=0..."})
        event = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert isinstance(event, TransitionEvent)
        assert event.transition.event is SessionEvent.CONNECTION_ANSWERED
        assert event.transition.from_state is SessionState.WAITING
        assert event.transition.to_state is SessionState.WAITING
        assert event.transition.detail == {
            "conn_id": ConnId(1),
            "answer": {"type": "answer", "sdp": "v=0..."},
        }
    finally:
        await runner.stop()


async def test_start_session_opens_the_session(started_runner: Runner) -> None:
    started_runner.start_session({})
    assert started_runner._sm.current_state is SessionState.WAITING
    started_runner.require_session_running(SESSION_ID)


async def test_require_session_running_rejects_an_unknown_sid(started_runner: Runner) -> None:
    started_runner.start_session({})
    with pytest.raises(UnknownSessionError):
        started_runner.require_session_running("not-the-session")


async def test_stop_session_closes_the_session(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner.stop_session()
    assert started_runner._sm.current_state is SessionState.CLOSING


async def test_enforce_blocks_a_running_session(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner.enforce(block=True)
    assert started_runner._sm.current_state is SessionState.CLOSING


async def test_enforce_without_block_is_a_noop(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner.enforce(block=False)
    assert started_runner._sm.current_state is SessionState.WAITING


async def test_track_map_reports_declared_tracks(started_runner: Runner) -> None:
    tracks = started_runner.track_map()
    assert tracks["main"]["kind"] == "video"
    assert tracks["main"]["direction"] == "out"
