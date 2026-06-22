import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from reactor_runtime import InputField, ModelMessage, Output, ReactorModel, Video, event, protocol
from reactor_runtime.core import (
    ClientConnected,
    ClientDisconnected,
    ConnectionCapabilities,
    ConnId,
    EndReason,
    ErrorEvent,
    HealthStatus,
    InboundCommandEvent,
    MediaBundle,
    RuntimeConfig,
    SessionEnded,
    SessionEvent,
    SessionStarted,
    SessionState,
    TransitionEvent,
)
from reactor_runtime.interface.internal.reactor_core import AddressedSink, BroadcastSink
from reactor_runtime.message_gateway import InboundCommand
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
        self.closed = False

    def send_message(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def send_control(self, payload: bytes | str) -> None:
        self.control.append(payload)

    def send_media(self, bundle: MediaBundle) -> None: ...

    def resume_track(self, name: str) -> None: ...

    def pause_track(self, name: str) -> None: ...

    async def close(self) -> None:
        self.closed = True


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


# --- the dispatch brain ---------------------------------------------------


def _record_reactor_events(runner: Runner, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    assert runner._bridge is not None
    events: list[Any] = []
    monkeypatch.setattr(runner._bridge, "dispatch_reactor_event", events.append)
    return events


async def test_session_start_crosses_into_the_model(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session({})
    assert any(isinstance(e, SessionStarted) for e in events)


async def test_connection_open_and_close_cross_into_the_model(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session({})
    started_runner.connection_opened(FakeConnection(1))
    started_runner.connection_closed(ConnId(1))
    kinds = [type(e) for e in events]
    assert ClientConnected in kinds
    assert ClientDisconnected in kinds


async def test_session_end_crosses_into_the_model_after_cleanup(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session({})
    started_runner.stop_session()
    await asyncio.sleep(0.01)
    assert started_runner._sm.current_state is SessionState.READY
    ended = [e for e in events if isinstance(e, SessionEnded)]
    assert len(ended) == 1
    assert ended[0].reason is EndReason.STOPPED


async def test_enforce_ends_the_session_as_moderated(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session({})
    started_runner.enforce(block=True)
    await asyncio.sleep(0.01)
    ended = [e for e in events if isinstance(e, SessionEnded)]
    assert len(ended) == 1
    assert ended[0].reason is EndReason.MODERATED


async def test_drain_ends_an_active_session_within_grace(started_runner: Runner) -> None:
    started_runner.start_session({})
    await started_runner.drain()
    assert started_runner._sm.current_state is SessionState.READY


# --- orphan timeout, teardown, egress journal, fatal exit -----------------


def _egress(runner: Runner) -> list[Any]:
    return [event for _seq, event in runner._events._history]


def _expect_state(runner: Runner, state: SessionState) -> None:
    # The call boundary keeps the type checker from narrowing the property to a
    # single literal across the consecutive asserts of a walk.
    assert runner._sm.current_state is state


async def test_orphan_timeout_closes_a_clientless_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(RuntimeConfig(model_ref="fake:Model", orphan_timeout=0.05))
    await runner.start()
    try:
        events = _record_reactor_events(runner, monkeypatch)
        runner.start_session({})
        _expect_state(runner, SessionState.WAITING)
        await asyncio.sleep(0.2)
        _expect_state(runner, SessionState.READY)
        ended = [e for e in events if isinstance(e, SessionEnded)]
        assert len(ended) == 1
        assert ended[0].reason is EndReason.TIMED_OUT
    finally:
        await runner.stop()


async def test_a_connecting_client_cancels_the_orphan_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(RuntimeConfig(model_ref="fake:Model", orphan_timeout=0.05))
    await runner.start()
    try:
        runner.start_session({})
        runner.connection_opened(FakeConnection(1))
        await asyncio.sleep(0.2)
        _expect_state(runner, SessionState.STREAMING)
    finally:
        await runner.stop()


async def test_session_end_tears_down_connections(started_runner: Runner) -> None:
    started_runner.start_session({})
    conn = FakeConnection(1)
    started_runner.connection_opened(conn)
    _expect_state(started_runner, SessionState.STREAMING)
    started_runner.stop_session()
    await asyncio.sleep(0.01)
    _expect_state(started_runner, SessionState.READY)
    assert started_runner._connections.count == 0
    assert conn.closed is True


class SlowCloseConnection(FakeConnection):
    """A connection whose close blocks until a test releases it."""

    def __init__(self, cid: int) -> None:
        super().__init__(cid)
        self.release = asyncio.Event()

    async def close(self) -> None:
        await self.release.wait()
        self.closed = True


async def test_cleanup_completes_only_after_connections_close(started_runner: Runner) -> None:
    started_runner.start_session({})
    conn = SlowCloseConnection(1)
    started_runner.connection_opened(conn)
    _expect_state(started_runner, SessionState.STREAMING)
    started_runner.stop_session()
    await asyncio.sleep(0.01)
    # The wire is still closing, so the session has not yet unwound to ready.
    _expect_state(started_runner, SessionState.CLOSING)
    conn.release.set()
    await asyncio.sleep(0.01)
    _expect_state(started_runner, SessionState.READY)
    assert conn.closed is True


async def test_stop_awaits_outstanding_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = _runner()
    await runner.start()
    runner.start_session({})
    conn = SlowCloseConnection(1)
    runner.connection_opened(conn)
    runner.stop_session()
    await asyncio.sleep(0.01)
    stopping = asyncio.create_task(runner.stop())
    await asyncio.sleep(0.01)
    # stop() cannot finish while a connection-close coroutine is still in flight.
    assert not stopping.done()
    conn.release.set()
    await stopping
    assert conn.closed is True
    assert runner._teardown == set()


async def test_drain_with_zero_grace_still_unwinds_to_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(RuntimeConfig(model_ref="fake:Model", grace_period=0))
    await runner.start()
    try:
        runner.start_session({})
        _expect_state(runner, SessionState.WAITING)
        await runner.drain()
        _expect_state(runner, SessionState.READY)
    finally:
        await runner.stop()


async def test_connection_open_and_close_are_journalled(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner.connection_opened(FakeConnection(1))
    started_runner.connection_closed(ConnId(1))
    moves = [
        e.transition
        for e in _egress(started_runner)
        if isinstance(e, TransitionEvent)
        and e.transition.event in (SessionEvent.CONNECTION_OPENED, SessionEvent.CONNECTION_CLOSED)
    ]
    assert [(m.event, m.detail["conn_id"]) for m in moves] == [
        (SessionEvent.CONNECTION_OPENED, ConnId(1)),
        (SessionEvent.CONNECTION_CLOSED, ConnId(1)),
    ]


async def test_accepted_command_is_journalled(started_runner: Runner) -> None:
    started_runner.start_session({})
    command = InboundCommand(
        name="set_mode", args={"mode": "fast"}, uploads={}, conn_id=ConnId(1), request_id="r1"
    )
    await started_runner._submit_command(command)
    journalled = [e for e in _egress(started_runner) if isinstance(e, InboundCommandEvent)]
    assert len(journalled) == 1
    assert journalled[0].name == "set_mode"
    assert dict(journalled[0].args) == {"mode": "fast"}


async def test_rejected_command_is_journalled_as_an_error(started_runner: Runner) -> None:
    started_runner.start_session({})
    command = InboundCommand(
        name="set_mode", args={"mode": ""}, uploads={}, conn_id=ConnId(1), request_id="r1"
    )
    await started_runner._submit_command(command)
    errors = [e for e in _egress(started_runner) if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert "set_mode" in errors[0].message
    assert not [e for e in _egress(started_runner) if isinstance(e, InboundCommandEvent)]


async def test_init_failure_requests_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ref: str) -> type:
        raise RuntimeError("no such model")

    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", boom)
    runner = _runner()
    called: list[bool] = []
    runner.request_shutdown = lambda: called.append(True)

    await runner.start()

    assert runner._sm.current_state is SessionState.TERMINATED
    assert called == [True]


async def test_transitions_are_logged(
    started_runner: Runner, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="reactor_runtime.runner.runner"):
        started_runner.start_session({})
    record = next(r for r in caplog.records if r.getMessage() == "session transition")
    fields = getattr(record, "reactor_fields", {})
    assert fields["event"] == "start_session"
    assert fields["from_state"] == "ready"
    assert fields["to_state"] == "waiting"
