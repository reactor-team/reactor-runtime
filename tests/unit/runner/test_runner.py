import asyncio
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reactor_runtime import (
    InputField,
    ModelMessage,
    Output,
    ReactorModel,
    UploadedFile,
    Video,
    event,
    file_uploaded,
    protocol,
)
from reactor_runtime.core import (
    ClientConnected,
    ClientDisconnected,
    ConnectionCapabilities,
    ConnId,
    EndReason,
    FileUploaded,
    HealthStatus,
    MediaBundle,
    RecordingConfig,
    RuntimeConfig,
    SessionEnded,
    SessionEvent,
    SessionStarted,
    SessionState,
    Transition,
    TransitionEvent,
)
from reactor_runtime.interface.internal.bridge import CommandOutcome
from reactor_runtime.interface.internal.reactor_core import AddressedSink, BroadcastSink
from reactor_runtime.message_gateway import InboundCommand
from reactor_runtime.protocol.common import struct_to_dict
from reactor_runtime.recording import ClipResult
from reactor_runtime.runner.runner import SESSION_ID, Runner
from reactor_runtime.transport.router import (
    SessionControl,
    SessionNotRunningError,
    SessionTransitionError,
    UnknownSessionError,
)
from reactor_runtime.upload_store import UnknownUploadError
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

    @event(name="set_image")
    async def set_image(self, image: UploadedFile) -> None: ...

    @file_uploaded
    def on_file(self, uploaded_file: UploadedFile) -> None: ...

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


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None,
    register_model: Callable[[type], None],
    register: Callable[..., None],
) -> None:
    register_model(FakeModel)
    register(Greeting)


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
        _seq, event = await asyncio.wait_for(anext(stream), timeout=1.0)
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


async def test_start_session_adopts_a_supplied_recording_id(started_runner: Runner) -> None:
    supplied = "11111111-2222-3333-4444-555555555555"
    started_runner.start_session({"session_id": supplied})
    assert started_runner._recording_id == supplied


async def test_start_session_without_a_session_id_mints_a_recording_id(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    # A director aligns clips by supplying an id; without one the recording gets a
    # freshly minted id rather than the fixed transport id, so sequential recordings
    # in a reused process never write to the same directory.
    assert started_runner._recording_id != SESSION_ID
    assert uuid.UUID(started_runner._recording_id)


async def test_sessions_without_a_session_id_get_distinct_recording_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    minted: list[str] = []
    for _ in range(2):
        runner = _runner()
        await runner.start()
        try:
            runner.start_session({})
            minted.append(runner._recording_id)
        finally:
            await runner.stop()
    assert minted[0] != minted[1]


async def test_require_session_running_rejects_an_unknown_sid(started_runner: Runner) -> None:
    started_runner.start_session({})
    with pytest.raises(UnknownSessionError):
        started_runner.require_session_running("not-the-session")


async def test_stop_session_closes_the_session(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner.stop_session()
    assert started_runner._sm.current_state is SessionState.CLOSING


async def test_start_session_rejects_a_double_start(started_runner: Runner) -> None:
    started_runner.start_session({})
    with pytest.raises(SessionTransitionError) as rejected:
        started_runner.start_session({})
    assert rejected.value.action == "start"
    assert rejected.value.state is SessionState.WAITING


def test_start_session_rejects_before_the_model_is_loaded() -> None:
    runner = _runner()  # constructed but not started: the session is CREATED
    with pytest.raises(SessionTransitionError) as rejected:
        runner.start_session({})
    assert rejected.value.state is SessionState.CREATED


async def test_stop_session_rejects_when_no_session_is_running(started_runner: Runner) -> None:
    with pytest.raises(SessionTransitionError) as rejected:
        started_runner.stop_session()
    assert rejected.value.action == "stop"
    assert rejected.value.state is SessionState.READY


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


async def test_descriptor_renders_the_v0_shape(started_runner: Runner) -> None:
    started_runner.start_session({})
    descriptor = started_runner.descriptor()

    assert descriptor["cluster"] == "local"
    assert descriptor["model"]["name"] == "fake_model"
    assert descriptor["server_info"]["server_version"]
    assert descriptor["selected_transport"] == {"protocol": "webrtc", "version": "1.0"}
    caps = descriptor["capabilities"]
    assert caps["protocol_version"] == "v0"
    # The model's outbound track is reported from the client's perspective.
    assert {"name": "main", "kind": "video", "direction": "recvonly"} in caps["tracks"]
    # Commands are not carried on the descriptor; a client reads them from /schema.
    assert caps["commands"] == []
    assert "emission_fps" not in caps
    # Recording metadata rides the descriptor so a consumer can mirror it at start.
    assert descriptor["recording"] == {"enabled": False, "chunk_seconds": 4}


async def test_schema_renders_the_model_contract(started_runner: Runner) -> None:
    schema = started_runner.schema()

    assert isinstance(schema, dict)
    assert schema
    assert "set_mode" in str(schema)


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


def _moves(runner: Runner, event: SessionEvent) -> list[Transition]:
    return [
        e.transition
        for e in _egress(runner)
        if isinstance(e, TransitionEvent) and e.transition.event is event
    ]


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


async def test_accepted_command_is_journalled_as_a_self_loop(started_runner: Runner) -> None:
    started_runner.start_session({})
    command = InboundCommand(
        name="set_mode", args={"mode": "fast"}, uploads={}, conn_id=ConnId(1), request_id="r1"
    )
    await started_runner._submit_command(command)
    journalled = _moves(started_runner, SessionEvent.COMMAND)
    assert len(journalled) == 1
    assert journalled[0].from_state is journalled[0].to_state
    assert journalled[0].detail == {
        "name": "set_mode",
        "args": {"mode": "fast"},
        "conn_id": ConnId(1),
    }


async def test_rejected_command_is_journalled_as_an_error(started_runner: Runner) -> None:
    started_runner.start_session({})
    command = InboundCommand(
        name="set_mode", args={"mode": ""}, uploads={}, conn_id=ConnId(1), request_id="r1"
    )
    await started_runner._submit_command(command)
    errors = _moves(started_runner, SessionEvent.ERROR)
    assert len(errors) == 1
    assert "set_mode" in errors[0].detail["message"]
    assert not _moves(started_runner, SessionEvent.COMMAND)


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


# --- model run-loop crash -------------------------------------------------


class CrashingModel(FakeModel):
    """A model whose run loop dies as soon as it starts."""

    async def run(self) -> None:
        raise RuntimeError("gpu fell off")


async def test_model_crash_terminates_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    created_models.clear()
    monkeypatch.setattr(
        "reactor_runtime.runner.runner.import_model_class", lambda ref: CrashingModel
    )
    runner = _runner()
    called: list[bool] = []
    runner.request_shutdown = lambda: called.append(True)

    await runner.start()
    try:
        # The crash reports from the model thread; joining it and letting the
        # loop run the scheduled eviction callback settles the terminal move.
        model = created_models[-1]
        assert model._thread is not None
        model._thread.join(timeout=2)
        await asyncio.sleep(0.05)

        assert runner._sm.current_state is SessionState.TERMINATED
        assert runner.health().status is HealthStatus.UNHEALTHY
        assert called == [True]
        moves = [
            e.transition
            for e in _egress(runner)
            if isinstance(e, TransitionEvent) and e.transition.event is SessionEvent.EVICTION
        ]
        assert len(moves) == 1
        assert moves[0].to_state is SessionState.TERMINATED
        assert moves[0].detail["reason"] is EndReason.ERROR
        assert moves[0].detail["error"] == "gpu fell off"
    finally:
        await runner.stop()


async def test_model_crash_mid_session_tears_the_connections_down(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    conn = FakeConnection(1)
    started_runner.connection_opened(conn)
    _expect_state(started_runner, SessionState.STREAMING)

    started_runner._on_model_failure(RuntimeError("gpu fell off"))
    await asyncio.sleep(0.05)  # let the loop run the scheduled eviction callback
    _expect_state(started_runner, SessionState.TERMINATED)
    await started_runner._drain_teardown()

    assert conn.closed
    assert started_runner.health().status is HealthStatus.UNHEALTHY


# --- file uploads --------------------------------------------------------


class PlainModel(ReactorModel):
    """A model with no upload hook, to prove file_uploaded is gated on one."""

    output: FakeOut

    def load(self, config_path: Path | None) -> None: ...

    async def run(self) -> None:
        await asyncio.sleep(60)


async def test_command_uploads_are_resolved_before_submit(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    upload_id = started_runner.uploads.create_slot("cat.png", "image/png", 4)
    started_runner.uploads.put(upload_id, b"\x89PNG")
    submitted: list[tuple[str, dict[str, Any]]] = []

    async def fake_submit(name: str, args: dict[str, Any], **kwargs: Any) -> CommandOutcome:
        submitted.append((name, args))
        return CommandOutcome.accept()

    assert started_runner._bridge is not None
    monkeypatch.setattr(started_runner._bridge, "submit_command", fake_submit)

    command = InboundCommand(
        name="set_image",
        args={},
        uploads={"image": upload_id},
        conn_id=ConnId(1),
        request_id="r1",
    )
    await started_runner._submit_command(command)

    assert submitted[0][0] == "set_image"
    assert submitted[0][1]["image"] == UploadedFile(
        name="cat.png", mime_type="image/png", data=b"\x89PNG"
    )


async def test_command_with_an_unresolved_upload_is_dropped(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    submitted: list[str] = []

    async def fake_submit(name: str, args: dict[str, Any], **kwargs: Any) -> CommandOutcome:
        submitted.append(name)
        return CommandOutcome.accept()

    assert started_runner._bridge is not None
    monkeypatch.setattr(started_runner._bridge, "submit_command", fake_submit)

    command = InboundCommand(
        name="set_image",
        args={},
        uploads={"image": "missing"},
        conn_id=ConnId(1),
        request_id="r1",
    )
    await started_runner._submit_command(command)

    assert submitted == []
    errors = _moves(started_runner, SessionEvent.ERROR)
    assert len(errors) == 1
    assert "set_image" in errors[0].detail["message"]


async def test_file_uploaded_dispatches_when_a_hook_exists(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    upload_id = started_runner.uploads.create_slot("cat.png", "image/png", 4)
    started_runner.uploads.put(upload_id, b"\x89PNG")

    started_runner.file_uploaded(ConnId(1), upload_id)
    await asyncio.sleep(0.01)

    uploaded = [e for e in events if isinstance(e, FileUploaded)]
    assert len(uploaded) == 1
    assert uploaded[0].file == UploadedFile(name="cat.png", mime_type="image/png", data=b"\x89PNG")
    assert uploaded[0].conn_id == ConnId(1)


async def test_file_uploaded_is_ignored_without_a_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: PlainModel)
    runner = _runner()
    await runner.start()
    try:
        events = _record_reactor_events(runner, monkeypatch)
        upload_id = runner.uploads.create_slot("cat.png", "image/png", 4)
        runner.uploads.put(upload_id, b"\x89PNG")
        runner.file_uploaded(ConnId(1), upload_id)
        await asyncio.sleep(0.01)
        assert not [e for e in events if isinstance(e, FileUploaded)]
    finally:
        await runner.stop()


async def test_upload_store_is_cleared_on_session_stop(started_runner: Runner) -> None:
    started_runner.start_session({})
    upload_id = started_runner.uploads.create_slot("a.bin", "application/octet-stream", 2)
    started_runner.uploads.put(upload_id, b"hi")

    started_runner.stop_session()
    await asyncio.sleep(0.01)

    with pytest.raises(UnknownUploadError):
        await started_runner.uploads.fetch(upload_id)


# --- recording -----------------------------------------------------------


def _clip() -> ClipResult:
    return ClipResult(
        session_id="rec-1",
        kind="snap",
        start_marker=1.0,
        end_marker=2.0,
        now_marker=2.0,
        predicted_ready_at_ms=123,
        playlist_url="/clips?session_id=rec-1&start=1.000&end=2.000",
    )


def test_clip_request_replies_clip_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    conn = FakeConnection(1)
    runner.connection_opened(conn)
    monkeypatch.setattr(runner._recorder, "request_clip", lambda _d: _clip())

    runner.clip_requested(ConnId(1), 30.0, "ctrl_c")

    # v0 carries the clip reply on the data channel.
    decoded = protocol.select(V0).decode(conn.sent[0], DATA, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    assert decoded.WhichOneof("payload") == "clip_ready"
    assert decoded.clip_ready.session_id == "rec-1"


def test_clip_request_on_a_disabled_recorder_replies_clip_failed() -> None:
    runner = _runner()  # recording is off by default
    conn = FakeConnection(1)
    runner.connection_opened(conn)

    runner.clip_requested(ConnId(1), 30.0, "ctrl_c")

    decoded = protocol.select(V0).decode(conn.sent[0], DATA, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    assert decoded.WhichOneof("payload") == "clip_failed"


def test_recording_request_reply_is_correlated_on_v1() -> None:
    runner = _runner()
    conn = FakeConnection(2)
    conn.protocol_version = V1
    runner.connection_opened(conn)

    runner.recording_requested(ConnId(2), "ctrl_r")

    decoded = protocol.select(V1).decode(conn.control[0], CONTROL, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    assert decoded.request_id == "ctrl_r"
    assert decoded.WhichOneof("payload") == "clip_failed"


async def _recording_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Runner:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(
        RuntimeConfig(
            model_ref="fake:Model",
            recording=RecordingConfig(enabled=True, recording_dir=str(tmp_path)),
        )
    )
    await runner.start()
    return runner


async def test_recorder_starts_on_session_start_and_stops_on_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = await _recording_runner(monkeypatch, tmp_path)
    try:
        runner.start_session({})
        assert runner.recorder._started is True
        runner.stop_session()
        await asyncio.sleep(0.1)
        assert runner.recorder._started is False
    finally:
        await runner.stop()


async def test_clip_ready_is_journalled_on_the_egress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = await _recording_runner(monkeypatch, tmp_path)
    try:
        runner._on_clip_ready(_clip())
        await asyncio.sleep(0.01)
        ready = _moves(runner, SessionEvent.CLIP_READY)
        assert len(ready) == 1
        assert ready[0].from_state is ready[0].to_state
        assert ready[0].detail == {
            "session_id": "rec-1",
            "kind": "snap",
            "start_marker": 1.0,
            "end_marker": 2.0,
            "now_marker": 2.0,
            "predicted_ready_at_ms": 123,
            "playlist_url": "/clips?session_id=rec-1&start=1.000&end=2.000",
        }
    finally:
        await runner.stop()


async def test_chunk_ready_is_journalled_on_the_egress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = await _recording_runner(monkeypatch, tmp_path)
    try:
        runner._on_chunk_ready("rec-1", 4)
        await asyncio.sleep(0.01)
        chunks = _moves(runner, SessionEvent.CHUNK_READY)
        assert [c.detail for c in chunks] == [{"recording_id": "rec-1", "idx": 4}]
        assert chunks[0].from_state is chunks[0].to_state
    finally:
        await runner.stop()


async def test_closing_self_loop_does_not_rerun_teardown(started_runner: Runner) -> None:
    started_runner.start_session({})
    conn = SlowCloseConnection(1)
    started_runner.connection_opened(conn)
    started_runner.stop_session()
    await asyncio.sleep(0.01)
    _expect_state(started_runner, SessionState.CLOSING)
    pending = set(started_runner._teardown)

    # The recording's final segment lands while the session is tearing down; its
    # self-loop is journalled but must not re-clear uploads or spawn a second
    # teardown, which would race the one already unwinding the session.
    started_runner._on_chunk_ready("rec-1", 7)
    await asyncio.sleep(0.01)

    _expect_state(started_runner, SessionState.CLOSING)
    assert started_runner._teardown.issubset(pending)
    assert [c.detail for c in _moves(started_runner, SessionEvent.CHUNK_READY)] == [
        {"recording_id": "rec-1", "idx": 7}
    ]
    conn.release.set()
    await asyncio.sleep(0.01)
    _expect_state(started_runner, SessionState.READY)


async def test_terminated_self_loop_does_not_rerequest_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(ref: str) -> type:
        raise RuntimeError("no such model")

    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", boom)
    runner = _runner()
    called: list[bool] = []
    runner.request_shutdown = lambda: called.append(True)
    await runner.start()
    assert called == [True]

    # An error journalled after the terminal move self-loops in TERMINATED
    # without asking the service to bring the process down a second time.
    assert runner._sm.send(SessionEvent.ERROR, message="late") is True
    assert runner._sm.current_state is SessionState.TERMINATED
    assert called == [True]


async def test_transitions_are_logged(
    started_runner: Runner, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="reactor_runtime.runner.runner"):
        started_runner.start_session({})
    record = next(r for r in caplog.records if r.getMessage() == "session transition")
    assert record.levelno == logging.INFO
    fields = getattr(record, "reactor_fields", {})
    assert fields["event"] == "start_session"
    assert fields["from_state"] == "ready"
    assert fields["to_state"] == "waiting"


async def test_journal_self_loops_are_logged_at_debug(
    started_runner: Runner, caplog: pytest.LogCaptureFixture
) -> None:
    started_runner.start_session({})
    with caplog.at_level(logging.DEBUG, logger="reactor_runtime.runner.runner"):
        started_runner._sm.send(SessionEvent.CHUNK_READY, recording_id="rec-1", idx=0)
    record = next(
        r
        for r in caplog.records
        if r.getMessage() == "session transition"
        and getattr(r, "reactor_fields", {}).get("event") == "chunk_ready"
    )
    assert record.levelno == logging.DEBUG
