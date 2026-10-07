import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime import (
    InputField,
    ModelMessage,
    Output,
    ReactorApp,
    UploadedFile,
    Video,
    event,
    file_uploaded,
    log,
    protocol,
    session_ended,
    session_started,
)
from reactor_runtime.codes import UNRESOLVED_UPLOAD
from reactor_runtime.core import (
    ClientConnected,
    ClientConnectionStat,
    ClientDisconnected,
    ClientStatsBatch,
    ClientTrackDirection,
    ClientTrackStat,
    CommandFailure,
    CompletedStep,
    Connection,
    ConnectionCapabilities,
    ConnId,
    EndReason,
    FileUploaded,
    HealthStatus,
    MediaBundle,
    MediaChunk,
    RecordingConfig,
    RuntimeConfig,
    SessionEnded,
    SessionEvent,
    SessionStarted,
    SessionState,
    StartingInputApplied,
    StepResultsConfig,
    TrackData,
    TrackInfo,
    TrackKind,
    Transition,
    TransitionEvent,
    TransportReading,
    TransportStatsSource,
)
from reactor_runtime.interface.internal.bridge import CommandOutcome
from reactor_runtime.interface.internal.reactor_core import (
    AddressedSink,
    BroadcastSink,
    MediaOps,
    MediaSink,
    StepSink,
)
from reactor_runtime.message_gateway import InboundCommand
from reactor_runtime.metrics import RuntimeMetrics
from reactor_runtime.protocol.common import dict_to_struct, struct_to_dict
from reactor_runtime.recording import ClipResult
from reactor_runtime.runner.client_stats import ClientStatsGate
from reactor_runtime.runner.runner import (
    _DRAIN_CLOSE_REASON,
    _RUNTIME_STATES,
    SESSION_ID,
    Runner,
)
from reactor_runtime.runner.session_start import (
    InvalidSessionStartError,
    SessionStart,
)
from reactor_runtime.runner.system_client import SYSTEM_CONN_ID, SystemConnection
from reactor_runtime.transport.router import (
    SessionControl,
    SessionNotRunningError,
    SessionTransitionError,
    UnknownSessionError,
)
from reactor_runtime.upload_store import UnknownUploadError
from reactor_wire.v1 import common_pb2, control_pb2, data_pb2, model_pb2

DATA = protocol.Channel.DATA
CONTROL = protocol.Channel.CONTROL
SERVER = protocol.Direction.SERVER
V0 = protocol.ProtocolVersion.V0
V1 = protocol.ProtocolVersion.V1


class Greeting(ModelMessage):
    text: str


class FakeOut(Output):
    main: Video


@dataclass
class Page:
    image: UploadedFile
    caption: str = ""


class FakeModel(ReactorApp):
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

    @event(name="set_images")
    async def set_images(self, images: list[UploadedFile]) -> None: ...

    @event(name="set_gallery")
    async def set_gallery(
        self,
        cover: UploadedFile,
        images: list[UploadedFile] | None = None,
        labels: dict[str, str] | None = None,
    ) -> None: ...

    @event(name="set_book")
    async def set_book(self, pages: dict[str, UploadedFile], cover: Page) -> None: ...

    @file_uploaded
    def on_file(self, uploaded_file: UploadedFile) -> None: ...

    def load(self, config_path: Path | None) -> None:
        self.events.append("load")
        self.loaded = config_path

    def bind_output(
        self,
        *,
        broadcast: BroadcastSink,
        addressed: AddressedSink,
        media: MediaSink,
        media_ops: MediaOps | None = None,
        step: StepSink | None = None,
    ) -> None:
        self.events.append("bind")
        super().bind_output(
            broadcast=broadcast, addressed=addressed, media=media, media_ops=media_ops, step=step
        )

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
        self.media_rate: float | None = None
        self.media_depth: int | None = None
        self.flushed = False

    def send_message(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def send_control(self, payload: bytes | str) -> None:
        self.control.append(payload)

    def send_media(self, chunk: MediaChunk) -> None: ...

    def flush_media(self) -> None:
        self.flushed = True

    def set_media_rate(self, fps: float) -> None:
        self.media_rate = fps

    def set_media_depth(self, depth: int) -> None:
        self.media_depth = depth

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


async def test_start_journals_initializing_before_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    # The loading phase is otherwise silent: a consumer subscribed from the
    # start of the journal must see an INITIALIZING self-loop on CREATED before
    # the INITIALIZATION_SUCCESS that leaves CREATED, so it can report a booting
    # pod during the load window rather than nothing until READY.
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = _runner()
    stream = runner._events.subscribe(since=0)

    await runner.start()
    try:

        async def first(n: int) -> list[Transition]:
            out: list[Transition] = []
            async for _seq, event in stream:
                out.append(event.transition)
                if len(out) >= n:
                    break
            return out

        boot = await asyncio.wait_for(first(2), timeout=2)
        assert boot[0].event is SessionEvent.INITIALIZING
        assert boot[0].from_state is SessionState.CREATED
        assert boot[0].to_state is SessionState.CREATED
        assert boot[1].event is SessionEvent.INITIALIZATION_SUCCESS
        assert boot[1].to_state is SessionState.READY
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


def test_every_session_state_has_a_lifecycle_word() -> None:
    # A session state missing from the table makes state() raise on /health,
    # the endpoint a probe reads to decide the process is alive.
    assert set(_RUNTIME_STATES) == set(SessionState)


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


def test_playout_settings_fan_out_and_reach_a_late_joiner() -> None:
    # The model's output-handle operations land on every live connection and
    # are remembered, so a connection opening later starts with them.
    runner = _runner()
    early = FakeConnection(1)
    runner.connection_opened(early)

    runner._set_media_rate(24.0)
    runner._set_media_depth(8)
    late = FakeConnection(2)
    runner.connection_opened(late)

    assert (early.media_rate, early.media_depth) == (24.0, 8)
    assert (late.media_rate, late.media_depth) == (24.0, 8)


def test_flush_media_cuts_every_connection() -> None:
    runner = _runner()
    a, b = FakeConnection(1), FakeConnection(2)
    runner.connection_opened(a)
    runner.connection_opened(b)

    runner._flush_media()

    assert a.flushed
    assert b.flushed


def test_a_flush_during_fan_out_abandons_the_remaining_connections() -> None:
    # A flush landing mid-broadcast must cut every connection: the chunk being
    # fanned out belongs to the flushed run and may reach no further wire.
    runner = _runner()
    delivered: list[ConnId] = []

    class Flushing(FakeConnection):
        def send_media(self, chunk: MediaChunk) -> None:
            delivered.append(self.id)
            runner._flush_media()

    class Recording(FakeConnection):
        def send_media(self, chunk: MediaChunk) -> None:
            delivered.append(self.id)

    runner.connection_opened(Flushing(1))
    runner.connection_opened(Recording(2))

    runner._emit_media(MediaChunk(bundle=MediaBundle(), fps=30.0, n_frames=1))

    assert delivered == [ConnId(1)]


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
        runner.start_session({})
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
        runner.start_session({})
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


def test_bodyless_reply_acks_a_v1_connection() -> None:
    runner = _runner()
    conn = FakeConnection(1)
    conn.protocol_version = V1
    runner.connection_opened(conn)

    runner._send_addressed(ConnId(1), None, "req-1")

    assert len(conn.sent) == 1
    decoded = protocol.select(V1).decode(conn.sent[0], DATA, SERVER)
    assert isinstance(decoded, data_pb2.DataServerMessage)
    assert decoded.request_id == "req-1"
    assert decoded.kind == common_pb2.MessageKind.MESSAGE_KIND_RESPONSE
    assert decoded.WhichOneof("payload") is None


def test_command_rejection_carries_its_reason_on_v1() -> None:
    runner = _runner()
    conn = FakeConnection(1)
    conn.protocol_version = V1
    runner.connection_opened(conn)

    runner._reject_command(ConnId(1), "req-2", "invalid_command", "value out of range")

    decoded = protocol.select(V1).decode(conn.sent[0], DATA, SERVER)
    assert isinstance(decoded, data_pb2.DataServerMessage)
    assert decoded.request_id == "req-2"
    assert decoded.WhichOneof("payload") == "error"
    assert decoded.error.code == "invalid_command"
    assert decoded.error.message == "value out of range"


def test_bodyless_reply_is_withheld_from_a_legacy_connection() -> None:
    runner = _runner()
    conn = FakeConnection(1)  # v0 by default: fire-and-forget commands, no acks
    runner.connection_opened(conn)

    runner._send_addressed(ConnId(1), None, "req-1")

    assert conn.sent == []


def test_bodyless_reply_without_a_request_id_sends_nothing() -> None:
    runner = _runner()
    conn = FakeConnection(1)
    conn.protocol_version = V1
    runner.connection_opened(conn)

    runner._send_addressed(ConnId(1), None, None)

    assert conn.sent == []


def test_a_handler_failure_travels_as_an_error_on_v1() -> None:
    runner = _runner()
    conn = FakeConnection(1)
    conn.protocol_version = V1
    runner.connection_opened(conn)

    runner._send_addressed(
        ConnId(1), CommandFailure("quota_exceeded", "No credits remain."), "req-3"
    )

    decoded = protocol.select(V1).decode(conn.sent[0], DATA, SERVER)
    assert isinstance(decoded, data_pb2.DataServerMessage)
    assert decoded.request_id == "req-3"
    assert decoded.WhichOneof("payload") == "error"
    assert decoded.error.code == "quota_exceeded"
    assert decoded.error.message == "No credits remain."


def test_a_handler_failure_is_withheld_from_a_legacy_connection() -> None:
    runner = _runner()
    conn = FakeConnection(1)  # v0 by default: fire-and-forget commands, no acks
    runner.connection_opened(conn)

    runner._send_addressed(
        ConnId(1), CommandFailure("quota_exceeded", "No credits remain."), "req-3"
    )

    assert conn.sent == []


async def test_a_handler_failure_is_journalled_with_its_correlation(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    # The model reports the failure from its own thread; this is the half the hop
    # schedules on the runtime loop, where the journal is single-writer.
    started_runner._emit_handler_failure(
        CommandFailure("quota_exceeded", "No credits remain."), ConnId(1), "req-3"
    )

    errors = _moves(started_runner, SessionEvent.ERROR)
    assert len(errors) == 1
    assert "quota_exceeded" in errors[0].detail["message"]
    # The entry carries what the COMMAND move carries, so an operator lines the
    # two up without matching timestamps.
    assert errors[0].detail["conn_id"] == ConnId(1)
    assert errors[0].detail["request_id"] == "req-3"


def test_an_uncorrelated_handler_failure_is_journalled_but_not_sent() -> None:
    runner = _runner()
    conn = FakeConnection(1)
    conn.protocol_version = V1
    runner.connection_opened(conn)

    # Every command a client sends carries a request id, minted by the gateway when
    # absent, so this is only reachable internally. A RESPONSE with no request id
    # is the one shape a client keyed on request ids cannot place.
    runner._send_addressed(ConnId(1), CommandFailure("quota_exceeded", "No credits."), None)

    assert conn.sent == []


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


async def test_a_connection_opened_during_closing_is_refused_and_closed(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    started_runner.stop_session()
    assert started_runner._sm.current_state is SessionState.CLOSING

    late = FakeConnection(7)
    started_runner.connection_opened(late)

    await asyncio.sleep(0.01)
    assert started_runner._sm.current_state is SessionState.READY
    assert started_runner._connections.count == 0
    assert late.closed


async def test_a_connection_opened_with_no_session_open_is_refused_and_closed(
    started_runner: Runner,
) -> None:
    late = FakeConnection(7)
    started_runner.connection_opened(late)

    await asyncio.sleep(0.01)
    assert started_runner._connections.count == 0
    assert late.closed


async def test_a_connection_opened_after_termination_is_refused_and_closed(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    started_runner._on_model_failure(RuntimeError("gpu fell off"))
    await asyncio.sleep(0.05)  # let the loop run the scheduled eviction callback
    _expect_state(started_runner, SessionState.TERMINATED)

    late = FakeConnection(7)
    started_runner.connection_opened(late)

    await asyncio.sleep(0.01)
    assert started_runner._connections.count == 0
    assert late.closed


async def test_a_wire_admitted_in_an_earlier_session_cannot_join_the_next(
    started_runner: Runner,
) -> None:
    # The offer is admitted in the first session, but its wire only connects
    # once the next session is running — the state looks valid, so only the
    # offer's epoch stamp can tell the sessions apart.
    started_runner.start_session({})
    stale_id = started_runner.new_conn_id()
    started_runner.offer_admitted(stale_id)
    started_runner.stop_session()
    await asyncio.sleep(0.01)
    _expect_state(started_runner, SessionState.READY)
    started_runner.start_session({})

    stale = FakeConnection(stale_id)
    started_runner.connection_opened(stale)

    await asyncio.sleep(0.01)
    assert started_runner._connections.count == 0
    assert stale.closed

    # An offer admitted in the live session still registers.
    fresh_id = started_runner.new_conn_id()
    started_runner.offer_admitted(fresh_id)
    fresh = FakeConnection(fresh_id)
    started_runner.connection_opened(fresh)
    assert started_runner._connections.count == 1
    assert not fresh.closed


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


async def _await_log_release() -> None:
    """Wait for the model thread to retire the log binding, which it does off-loop."""
    deadline = time.monotonic() + 2.0
    while log.get_session_id() is not None:
        assert time.monotonic() < deadline, f"binding never released: {log.get_session_id()}"
        await asyncio.sleep(0.01)


async def test_a_live_session_stamps_its_id_on_the_logs(started_runner: Runner) -> None:
    supplied = "11111111-2222-3333-4444-555555555555"
    assert log.get_session_id() is None
    started_runner.start_session({"session_id": supplied})
    assert log.get_session_id() == supplied


async def test_the_stamped_state_tracks_the_lifecycle(started_runner: Runner) -> None:
    # The runner stamps its starting state at construction — the model-load
    # window is what makes initialization logs retrievable — and re-stamps on
    # every move, so a record always reads the phase it was written in, in both
    # the machine's words and the health route's coarse projection.
    assert (log.get_state(), log.get_runtime_state()) == ("ready", "available")
    started_runner.start_session({})
    assert (log.get_state(), log.get_runtime_state()) == ("waiting", "serving")
    conn = FakeConnection(1)
    started_runner.connection_opened(conn)
    _expect_state(started_runner, SessionState.STREAMING)
    assert (log.get_state(), log.get_runtime_state()) == ("streaming", "serving")
    started_runner.stop_session()
    await asyncio.sleep(0.01)
    _expect_state(started_runner, SessionState.READY)
    assert (log.get_state(), log.get_runtime_state()) == ("ready", "available")


async def test_construction_stamps_the_created_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    assert log.get_state() is None
    runner = _runner()
    assert (log.get_state(), log.get_runtime_state()) == ("created", "loading")
    await runner.start()
    try:
        assert (log.get_state(), log.get_runtime_state()) == ("ready", "available")
    finally:
        await runner.stop()


async def test_an_eviction_stamps_the_terminated_state(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner._on_model_failure(RuntimeError("gpu fell off"))
    await asyncio.sleep(0.05)  # let the loop run the scheduled eviction callback
    _expect_state(started_runner, SessionState.TERMINATED)
    assert (log.get_state(), log.get_runtime_state()) == ("terminated", "terminated")


async def test_a_record_written_in_session_carries_state_and_id(
    started_runner: Runner,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # caplog's handler bypasses the stamping filter, so run it explicitly over
    # the captured record — this asserts the ambient context a real handler
    # would stamp, on a record no call site enriched.
    started_runner.start_session({"session_id": "11111111-2222-3333-4444-555555555555"})
    with caplog.at_level(logging.INFO, logger="some.model.module"):
        logging.getLogger("some.model.module").info("mid-session record")
    record = caplog.records[-1]
    assert log.SessionContextFilter().filter(record)
    fields = getattr(record, "reactor_fields", {})
    assert fields["session_id"] == "11111111-2222-3333-4444-555555555555"
    assert fields["state"] == "waiting"
    assert fields["runtime_state"] == "serving"


async def test_closing_a_session_releases_the_stamped_id(started_runner: Runner) -> None:
    started_runner.start_session({"session_id": "11111111-2222-3333-4444-555555555555"})
    started_runner.stop_session()
    await asyncio.sleep(0.01)
    _expect_state(started_runner, SessionState.READY)
    # The process may host another session, so a stale id would misattribute it.
    await started_runner._drain_teardown()
    await _await_log_release()


async def _two_sessions_with_a_started_barrier(
    monkeypatch: pytest.MonkeyPatch, first_sid: str, second_sid: str
) -> None:
    """Run two sessions back to back and prove the first's release spared the second.

    The reactor loop dispatches in order, so once the second session's
    ``@session_started`` hook has run, the first session's ``SessionEnded`` — and
    with it the release of its binding — has already been dispatched. Waiting on
    the hook is the deterministic barrier that makes the assertion meaningful: it
    asserts only after the late release has provably happened.
    """
    started = threading.Event()

    class BarrierModel(FakeModel):
        @session_started
        def on_session_started(self) -> None:
            started.set()

    monkeypatch.setattr(
        "reactor_runtime.runner.runner.import_model_class", lambda ref: BarrierModel
    )
    runner = _runner()
    await runner.start()
    try:
        # The model thread creates its queue on its own loop, and an event
        # enqueued before then is deliberately dropped. Production start_session
        # calls arrive long after boot; this test's arrives instantly, so wait
        # for the loop before opening the first session.
        model = created_models[-1]
        deadline = time.monotonic() + 2.0
        while model._inbound_q is None:
            assert time.monotonic() < deadline, "model loop never became ready"
            await asyncio.sleep(0.01)
        runner.start_session({"session_id": first_sid})
        assert await asyncio.to_thread(started.wait, 2.0), "first session_started never ran"
        started.clear()
        runner.stop_session()
        await asyncio.sleep(0.01)
        _expect_state(runner, SessionState.READY)
        runner.start_session({"session_id": second_sid})
        assert await asyncio.to_thread(started.wait, 2.0), "second session_started never ran"
        assert log.get_session_id() == second_sid
    finally:
        await runner.stop()


async def test_a_late_release_cannot_strip_the_session_that_followed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The release rides the first session's SessionEnded, so the next session is
    # live before it runs; the binding token is what keeps it from unbinding her.
    await _two_sessions_with_a_started_barrier(
        monkeypatch,
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    )


async def test_a_session_reusing_the_previous_id_keeps_its_own_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A caller may start two sessions under one id; the first session's release
    # must not unbind the second, which would leave its whole run unstamped.
    reused = "11111111-1111-1111-1111-111111111111"
    await _two_sessions_with_a_started_barrier(monkeypatch, reused, reused)


async def test_session_ended_hook_records_are_stamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The release rides the reactor queue behind SessionEnded, so the hook has
    # returned — its records stamped — before the binding is retired. Recording
    # is off here, so teardown is immediate and only the queue order protects
    # the hook; this is the case a teardown-only wait loses.
    seen: list[str | None] = []

    class HookedModel(FakeModel):
        @session_ended
        def on_session_ended(self) -> None:
            seen.append(log.get_session_id())

    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: HookedModel)
    runner = _runner()
    await runner.start()
    try:
        sid = "11111111-2222-3333-4444-555555555555"
        runner.start_session({"session_id": sid})
        runner.stop_session()
        await asyncio.sleep(0.01)
        _expect_state(runner, SessionState.READY)
        await runner._drain_teardown()
        # Released only after the hook, so the release doubles as its barrier.
        await _await_log_release()
        assert seen == [sid]
    finally:
        await runner.stop()


async def test_a_second_session_stamps_its_own_id(started_runner: Runner) -> None:
    started_runner.start_session({"session_id": "11111111-1111-1111-1111-111111111111"})
    started_runner.stop_session()
    await asyncio.sleep(0.01)
    started_runner.start_session({"session_id": "22222222-2222-2222-2222-222222222222"})
    assert log.get_session_id() == "22222222-2222-2222-2222-222222222222"


async def test_an_eviction_leaves_the_binding_for_the_process_exit(
    started_runner: Runner,
) -> None:
    # An eviction is the model loop's own death: no SessionEnded is dispatched,
    # so nothing retires the binding — deliberately. The process is exiting, and
    # its last records belong to the session that brought it down.
    sid = "11111111-2222-3333-4444-555555555555"
    started_runner.start_session({"session_id": sid})
    started_runner._on_model_failure(RuntimeError("gpu fell off"))
    await asyncio.sleep(0.05)  # let the loop run the scheduled eviction callback
    _expect_state(started_runner, SessionState.TERMINATED)
    await started_runner._drain_teardown()
    assert log.get_session_id() == sid


async def test_the_transition_log_carries_no_id_of_its_own(
    started_runner: Runner,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The fixed transport id is one constant per process and carries nothing,
    # and a call-site session_id would beat the stamped one — so the transition
    # log names neither, and the session it belongs to arrives via the stamp
    # alone on exactly the records an operator reaches for first.
    with caplog.at_level(logging.INFO, logger="reactor_runtime.runner.runner"):
        started_runner.start_session({"session_id": "11111111-2222-3333-4444-555555555555"})
    moves = [r for r in caplog.records if r.message == "session transition"]
    assert moves
    fields = getattr(moves[-1], "reactor_fields", {})
    assert "session_id" not in fields
    assert "transport_session_id" not in fields


async def test_require_session_running_rejects_an_unknown_sid(started_runner: Runner) -> None:
    started_runner.start_session({})
    with pytest.raises(UnknownSessionError):
        started_runner.require_session_running("not-the-session")


async def test_stop_session_closes_the_session(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner.stop_session()
    assert started_runner._sm.current_state is SessionState.CLOSING


async def test_the_runner_records_its_session_on_the_registry_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    metrics = RuntimeMetrics(version="0.0.0", model="fake:Model")
    runner = Runner(RuntimeConfig(model_ref="fake:Model"), metrics)

    await runner.start()
    try:
        runner.start_session({})
        runner.stop_session()
        await asyncio.sleep(0.01)
    finally:
        await runner.stop()

    # Pins the seam the rest of the metrics tests cannot see: they subscribe a
    # recorder to a state machine of their own, so the runner could drop the
    # listener, or be handed a different holder, and they would all stay green.
    assert metrics.registry.get_sample_value("runtime_sessions_total", {"reason": "stopped"}) == 1.0


async def test_start_session_rejects_a_double_start(started_runner: Runner) -> None:
    started_runner.start_session({})
    with pytest.raises(SessionTransitionError) as rejected:
        started_runner.start_session({})
    assert rejected.value.action == "start"
    assert rejected.value.state is SessionState.WAITING


async def test_a_rejected_start_leaves_the_recording_id_untouched(
    started_runner: Runner,
) -> None:
    supplied = "11111111-2222-3333-4444-555555555555"
    started_runner.start_session({"session_id": supplied})
    # A start is legal only from READY. A second one arriving while the session is
    # live is rejected, and that rejection must not rebind the live session's
    # recording id: neither to the rejected request's own id, nor to a freshly
    # minted one when it carries none.
    with pytest.raises(SessionTransitionError):
        started_runner.start_session({"session_id": "99999999-9999-9999-9999-999999999999"})
    assert started_runner._recording_id == supplied
    with pytest.raises(SessionTransitionError):
        started_runner.start_session({})
    assert started_runner._recording_id == supplied


async def test_start_session_adopts_the_session_shape(started_runner: Runner) -> None:
    started_runner.start_session(
        {
            "starting_input": {"commands": [{"command": "set_mode", "data": {"mode": "x"}}]},
            "steps": 2,
        }
    )
    shape = started_runner._session_start
    assert shape.steps == 2
    assert shape.starting_input is not None
    assert [command.command for command in shape.starting_input.commands] == ["set_mode"]


async def test_a_malformed_start_is_rejected_before_the_session_moves(
    started_runner: Runner,
) -> None:
    with pytest.raises(InvalidSessionStartError):
        started_runner.start_session({"steps": 0})
    assert started_runner._sm.current_state is SessionState.READY
    assert _moves(started_runner, SessionEvent.START_SESSION) == []


@pytest.mark.parametrize(
    ("starting_input", "message"),
    [
        ({"state": {"mode": ""}}, "starting_input.state.mode is refused: mode: "),
        (
            {
                "commands": [
                    {"command": "set_mode", "data": {"mode": "ok"}},
                    {"command": "set_mode", "data": {"mode": ""}},
                ]
            },
            "starting_input.commands[1] is refused: mode: ",
        ),
        (
            {"commands": [{"command": "set_mode", "data": {"mode": "ok", "extra": 1}}]},
            "starting_input.commands[0] is refused: extra: unexpected argument",
        ),
        (
            {"commands": [{"command": "nope"}]},
            "starting_input.commands[0] is refused: nope: unknown command",
        ),
        (
            {"state": {"speed": 2}},
            "starting_input.state.speed is refused: set_speed: unknown command",
        ),
        (
            {"commands": [{"command": "set_image", "data": {"image": "fox.png"}}]},
            "starting_input.commands[0] is refused: image: ",
        ),
    ],
)
async def test_a_starting_command_the_contract_refuses_refuses_the_start(
    started_runner: Runner, starting_input: dict[str, Any], message: str
) -> None:
    with pytest.raises(InvalidSessionStartError, match=re.escape(message)):
        started_runner.start_session({"starting_input": starting_input, "steps": 1})

    assert started_runner._sm.current_state is SessionState.READY
    assert _moves(started_runner, SessionEvent.START_SESSION) == []
    assert started_runner._session_start == SessionStart()


async def test_a_refused_starting_command_names_its_field_once(started_runner: Runner) -> None:
    with pytest.raises(InvalidSessionStartError) as refused:
        started_runner.start_session({"starting_input": {"state": {"mode": ""}}})

    assert str(refused.value).count("mode:") == 1


async def test_a_starting_upload_is_checked_as_a_reference_only(started_runner: Runner) -> None:
    # The bytes arrive after the start, so only the reference's shape is checked.
    started_runner.start_session(
        {
            "starting_input": {
                "commands": [{"command": "set_image", "data": {"image": {"upload_id": "u-1"}}}]
            }
        }
    )

    assert started_runner._sm.current_state is not SessionState.READY


async def test_a_rejected_start_leaves_the_session_shape_untouched(
    started_runner: Runner,
) -> None:
    started_runner.start_session({"steps": 3})
    with pytest.raises(SessionTransitionError):
        started_runner.start_session({"steps": 5})
    with pytest.raises(InvalidSessionStartError):
        started_runner.start_session({"steps": 0})
    assert started_runner._session_start.steps == 3


def _journalled_commands(runner: Runner) -> list[tuple[str, dict[str, Any], Any]]:
    return [
        (move.detail["name"], move.detail["args"], move.detail["conn_id"])
        for move in _moves(runner, SessionEvent.COMMAND)
    ]


def _client_command(name: str, **args: Any) -> InboundCommand:
    return InboundCommand(
        name=name,
        args=args,
        uploads={},
        conn_id=ConnId(1),
        request_id="r1",
        received_at=time.monotonic(),
    )


async def test_the_starting_input_runs_its_state_then_its_commands(
    started_runner: Runner,
) -> None:
    started_runner.start_session(
        {
            "starting_input": {
                "state": {"mode": "from-state"},
                "commands": [{"command": "set_mode", "data": {"mode": "from-commands"}}],
            }
        }
    )
    await started_runner._starting_input_done.wait()

    assert _journalled_commands(started_runner) == [
        ("set_mode", {"mode": "from-state"}, SYSTEM_CONN_ID),
        ("set_mode", {"mode": "from-commands"}, SYSTEM_CONN_ID),
    ]


async def test_a_starting_command_that_fails_when_it_runs_is_journalled_and_the_rest_still_run(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Its arguments pass at the start; only running it shows the upload is missing.
    async def never_arrives(upload_id: str, **kwargs: Any) -> UploadedFile:
        raise UnknownUploadError(upload_id)

    monkeypatch.setattr(started_runner._uploads, "fetch", never_arrives)
    started_runner.start_session(
        {
            "starting_input": {
                "commands": [
                    {"command": "set_image", "data": {"image": {"upload_id": "never"}}},
                    {"command": "set_mode", "data": {"mode": "ok"}},
                ],
            }
        }
    )
    await started_runner._starting_input_done.wait()

    errors = [move.detail["message"] for move in _moves(started_runner, SessionEvent.ERROR)]
    assert errors == ["command 'set_image' references an unresolved upload"]
    assert _journalled_commands(started_runner) == [("set_mode", {"mode": "ok"}, SYSTEM_CONN_ID)]


async def test_a_client_command_waits_behind_the_starting_input(started_runner: Runner) -> None:
    started_runner.start_session(
        {
            "starting_input": {
                "commands": [{"command": "set_image", "data": {"image": {"upload_id": "u-1"}}}]
            }
        }
    )
    client = asyncio.create_task(
        started_runner._submit_client_command(_client_command("set_mode", mode="client"))
    )
    await asyncio.sleep(0.05)
    # The starting command waits for its upload, and the client's waits behind it.
    assert _journalled_commands(started_runner) == []

    started_runner.uploads.create_slot("fox.png", "image/png", 3, "u-1")
    started_runner.uploads.put("u-1", b"png")
    await client

    assert [name for name, _, _ in _journalled_commands(started_runner)] == [
        "set_image",
        "set_mode",
    ]


async def test_a_starting_command_waits_longer_than_a_client_for_its_upload(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The starting wait is the orphan timeout, never less than the client's.
    # Shrinking the client's shows the starting command does not use it.
    monkeypatch.setattr("reactor_runtime.runner.runner._UPLOAD_RESOLVE_TIMEOUT_SECONDS", 0.01)
    started_runner.start_session(
        {
            "starting_input": {
                "commands": [{"command": "set_image", "data": {"image": {"upload_id": "u-1"}}}]
            }
        }
    )
    await asyncio.sleep(0.1)
    started_runner.uploads.create_slot("fox.png", "image/png", 3, "u-1")
    started_runner.uploads.put("u-1", b"png")
    await started_runner._starting_input_done.wait()

    assert [name for name, _, _ in _journalled_commands(started_runner)] == ["set_image"]
    assert _moves(started_runner, SessionEvent.ERROR) == []


async def test_the_starting_input_runs_once_per_session(started_runner: Runner) -> None:
    started_runner.start_session(
        {"starting_input": {"commands": [{"command": "set_mode", "data": {"mode": "a"}}]}}
    )
    await started_runner._starting_input_done.wait()
    started_runner.connection_opened(FakeConnection(1))
    started_runner.connection_closed(ConnId(1))
    started_runner.connection_opened(FakeConnection(2))
    await asyncio.sleep(0.01)
    assert len(_journalled_commands(started_runner)) == 1

    started_runner.stop_session()
    await started_runner._drain_teardown()
    started_runner.start_session({})
    await started_runner._starting_input_done.wait()
    assert len(_journalled_commands(started_runner)) == 1


async def test_a_session_end_stops_the_starting_input_and_frees_client_commands(
    started_runner: Runner,
) -> None:
    started_runner.start_session(
        {
            "starting_input": {
                "commands": [
                    {"command": "set_image", "data": {"image": {"upload_id": "never"}}},
                    {"command": "set_mode", "data": {"mode": "after"}},
                ]
            }
        }
    )
    await asyncio.sleep(0.01)
    started_runner.stop_session()
    await asyncio.wait_for(started_runner._starting_input_done.wait(), 1.0)
    await asyncio.sleep(0.01)

    assert _journalled_commands(started_runner) == []
    errors = [move.detail["message"] for move in _moves(started_runner, SessionEvent.ERROR)]
    assert errors == ["command 'set_image' references an unresolved upload"]


def _connection_moves(runner: Runner) -> list[tuple[SessionEvent, dict[str, Any]]]:
    connection_events = {SessionEvent.CONNECTION_OPENED, SessionEvent.CONNECTION_CLOSED}
    return [
        (e.transition.event, dict(e.transition.detail))
        for e in _egress(runner)
        if isinstance(e, TransitionEvent) and e.transition.event in connection_events
    ]


_SYSTEM_DETAIL = {"conn_id": SYSTEM_CONN_ID, "system": True}


async def test_the_system_client_connects_before_the_starting_commands(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session(
        {"starting_input": {"commands": [{"command": "set_mode", "data": {"mode": "a"}}]}}
    )
    await started_runner._starting_input_done.wait()

    # The model learns the session started, then that the system client joined,
    # and only then receives the starting command it sends.
    assert isinstance(events[0], SessionStarted)
    assert events[1] == ClientConnected(SYSTEM_CONN_ID, 1, system=True)
    moves = [
        e.transition.event
        for e in _egress(started_runner)
        if isinstance(e, TransitionEvent)
        and e.transition.event
        in {
            SessionEvent.START_SESSION,
            SessionEvent.CONNECTION_OPENED,
            SessionEvent.COMMAND,
            SessionEvent.CONNECTION_CLOSED,
        }
    ]
    assert moves == [
        SessionEvent.START_SESSION,
        SessionEvent.CONNECTION_OPENED,
        SessionEvent.COMMAND,
        SessionEvent.CONNECTION_CLOSED,
    ]


async def test_without_steps_the_system_client_leaves_after_the_starting_input(
    started_runner: Runner,
) -> None:
    started_runner.start_session(
        {"starting_input": {"commands": [{"command": "set_mode", "data": {"mode": "a"}}]}}
    )
    await started_runner._starting_input_done.wait()

    assert _connection_moves(started_runner) == [
        (SessionEvent.CONNECTION_OPENED, _SYSTEM_DETAIL),
        (SessionEvent.CONNECTION_CLOSED, _SYSTEM_DETAIL),
    ]
    # The system client does not occupy a session without steps: its open and
    # close are self-loops, and the session waits for a client of its own.
    for connection_event in (SessionEvent.CONNECTION_OPENED, SessionEvent.CONNECTION_CLOSED):
        (move,) = _moves(started_runner, connection_event)
        assert (move.from_state, move.to_state) == (SessionState.WAITING, SessionState.WAITING)
    _expect_state(started_runner, SessionState.WAITING)
    assert started_runner._orphan_task is not None


async def test_without_steps_the_orphan_timer_runs_while_the_starting_input_applies(
    started_runner: Runner,
) -> None:
    started_runner.start_session(
        {"starting_input": {"commands": [{"command": "set_mode", "data": {"mode": "a"}}]}}
    )
    # The timer armed by the start is the one that still runs: the system
    # client's open and close leave the session in waiting, so neither re-arms it.
    armed = started_runner._orphan_task
    assert armed is not None
    await started_runner._starting_input_done.wait()

    assert started_runner._orphan_task is armed


async def test_without_steps_a_client_that_joins_during_the_starting_input_occupies_the_session(
    started_runner: Runner,
) -> None:
    started_runner.start_session(
        {"starting_input": {"commands": [{"command": "set_mode", "data": {"mode": "a"}}]}}
    )
    conn = FakeConnection(1002)
    started_runner.connection_opened(conn)
    _expect_state(started_runner, SessionState.STREAMING)
    await started_runner._starting_input_done.wait()

    # The system client's close is a self-loop, so the client keeps the session
    # streaming; the client's own close then leaves it orphaned.
    (closed,) = _moves(started_runner, SessionEvent.CONNECTION_CLOSED)
    assert closed.detail == _SYSTEM_DETAIL
    assert (closed.from_state, closed.to_state) == (
        SessionState.STREAMING,
        SessionState.STREAMING,
    )
    started_runner.connection_closed(conn.id)
    _expect_state(started_runner, SessionState.ORPHANED)


async def test_with_steps_the_system_client_stays_after_the_starting_input(
    started_runner: Runner,
) -> None:
    started_runner.start_session(
        {
            "starting_input": {"commands": [{"command": "set_mode", "data": {"mode": "a"}}]},
            "steps": 1,
        }
    )
    await started_runner._starting_input_done.wait()

    assert _connection_moves(started_runner) == [(SessionEvent.CONNECTION_OPENED, _SYSTEM_DETAIL)]
    _expect_state(started_runner, SessionState.STREAMING)
    assert started_runner._orphan_task is None


async def test_steps_alone_connect_the_system_client(started_runner: Runner) -> None:
    started_runner.start_session({"steps": 3})

    assert _connection_moves(started_runner) == [(SessionEvent.CONNECTION_OPENED, _SYSTEM_DETAIL)]
    _expect_state(started_runner, SessionState.STREAMING)


async def test_a_session_without_a_starting_input_or_steps_has_no_system_client(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})

    assert _connection_moves(started_runner) == []
    _expect_state(started_runner, SessionState.WAITING)


async def test_the_session_start_tells_the_model_whether_a_starting_input_follows(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session({"starting_input": {"state": {"mode": "a"}}})
    await started_runner._starting_input_done.wait()
    started_runner.stop_session()
    await started_runner._drain_teardown()
    started_runner.start_session({"steps": 1})

    starts = [event for event in events if isinstance(event, SessionStarted)]
    assert [start.starting_input for start in starts] == [True, False]
    assert [type(event) for event in events].count(StartingInputApplied) == 1


async def test_the_step_loop_opens_after_the_last_starting_command(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = started_runner._bridge
    assert bridge is not None
    order: list[Any] = []
    submit = bridge.submit_command

    async def record_submit(name: str, args: dict[str, Any], **kwargs: Any) -> Any:
        order.append(name)
        return await submit(name, args, **kwargs)

    monkeypatch.setattr(bridge, "submit_command", record_submit)
    monkeypatch.setattr(bridge, "dispatch_reactor_event", order.append)
    started_runner.start_session(
        {
            "starting_input": {
                "state": {"mode": "a"},
                "commands": [{"command": "set_image", "data": {"image": {"upload_id": "u-1"}}}],
            },
            "steps": 1,
        }
    )
    await asyncio.sleep(0.05)
    # The upload has not arrived, so the step loop is still shut.
    assert StartingInputApplied() not in order

    started_runner.uploads.create_slot("fox.png", "image/png", 3, "u-1")
    started_runner.uploads.put("u-1", b"png")
    await started_runner._starting_input_done.wait()

    assert order[-3:] == ["set_mode", "set_image", StartingInputApplied()]


async def test_without_steps_the_system_client_leaves_before_the_step_loop_opens(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session(
        {"starting_input": {"commands": [{"command": "set_mode", "data": {"mode": "a"}}]}}
    )
    await started_runner._starting_input_done.wait()

    # A live session takes no step until a client of its own joins.
    assert events[-2:] == [ClientDisconnected(SYSTEM_CONN_ID, 0), StartingInputApplied()]


async def test_a_session_that_ends_before_its_starting_input_lands_opens_no_step_loop(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session(
        {
            "starting_input": {
                "commands": [{"command": "set_image", "data": {"image": {"upload_id": "never"}}}]
            }
        }
    )
    await asyncio.sleep(0.01)
    started_runner.stop_session()
    await asyncio.wait_for(started_runner._starting_input_done.wait(), 1.0)

    assert StartingInputApplied() not in events


async def test_a_real_client_keeps_the_session_streaming_when_the_system_client_leaves(
    started_runner: Runner,
) -> None:
    started_runner.start_session(
        {
            "starting_input": {
                "commands": [{"command": "set_image", "data": {"image": {"upload_id": "u-1"}}}]
            }
        }
    )
    started_runner.connection_opened(FakeConnection(1002))
    started_runner.uploads.create_slot("fox.png", "image/png", 3, "u-1")
    started_runner.uploads.put("u-1", b"png")
    await started_runner._starting_input_done.wait()

    _expect_state(started_runner, SessionState.STREAMING)
    assert started_runner._connections.count == 1


async def test_the_session_end_closes_the_system_client(started_runner: Runner) -> None:
    started_runner.start_session({"steps": 1})
    started_runner.stop_session()
    await started_runner._drain_teardown()

    _expect_state(started_runner, SessionState.READY)
    assert started_runner._connections.count == 0


def test_the_system_client_carries_no_media_and_conforms_to_the_protocol() -> None:
    conn = SystemConnection()
    assert isinstance(conn, Connection)
    assert conn.id == SYSTEM_CONN_ID
    assert not conn.capabilities.carries_video
    assert not conn.capabilities.carries_audio
    # No wire, so nothing to sample.
    assert not isinstance(conn, TransportStatsSource)


def _step(runner: Runner, *, error: str | None = None) -> CompletedStep:
    """A step report from the session the runner is serving now."""
    return CompletedStep(bundle=None, error=error, session=runner._sessions_posted)


def _step_facts(runner: Runner) -> list[dict[str, Any]]:
    return [dict(move.detail) for move in _moves(runner, SessionEvent.STEP_COMPLETED)]


async def test_each_reported_step_is_journalled_with_its_number(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner._record_step(_step(started_runner))
    started_runner._record_step(_step(started_runner, error="RuntimeError: out of memory"))

    assert _step_facts(started_runner) == [
        {"step": 1, "saved": False, "error": None},
        {"step": 2, "saved": False, "error": "RuntimeError: out of memory"},
    ]
    move = _moves(started_runner, SessionEvent.STEP_COMPLETED)[0]
    assert move.from_state is move.to_state


async def test_step_numbers_start_over_with_each_session(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner._record_step(_step(started_runner))
    started_runner.stop_session()
    await started_runner._drain_teardown()
    started_runner.start_session({})
    started_runner._record_step(_step(started_runner))

    assert [fact["step"] for fact in _step_facts(started_runner)] == [1, 1]


async def test_a_session_stops_when_it_reaches_its_steps(started_runner: Runner) -> None:
    started_runner.start_session({"steps": 2})
    started_runner._record_step(_step(started_runner))
    _expect_state(started_runner, SessionState.STREAMING)
    started_runner._record_step(_step(started_runner))

    stops = _moves(started_runner, SessionEvent.STOP_SESSION)
    assert len(stops) == 1
    assert stops[0].detail == {
        "reason": EndReason.STOPPED,
        "close_reason": "Session ended: the requested steps are complete.",
    }
    _expect_state(started_runner, SessionState.CLOSING)
    await started_runner._drain_teardown()
    _expect_state(started_runner, SessionState.READY)


async def test_a_step_after_the_stop_is_journalled_and_stops_nothing(
    started_runner: Runner,
) -> None:
    started_runner.start_session({"steps": 1})
    started_runner._record_step(_step(started_runner))
    started_runner._record_step(_step(started_runner))

    assert [fact["step"] for fact in _step_facts(started_runner)] == [1, 2]
    assert len(_moves(started_runner, SessionEvent.STOP_SESSION)) == 1


async def test_a_late_step_from_the_previous_session_cannot_close_the_next(
    started_runner: Runner,
) -> None:
    started_runner.start_session({"steps": 1})
    previous = _step(started_runner)
    started_runner.stop_session()
    await started_runner._drain_teardown()
    started_runner.start_session({"steps": 1})

    # The model finishes a unit it began before the restart and reports it now.
    started_runner._record_step(previous)

    assert _step_facts(started_runner) == []
    _expect_state(started_runner, SessionState.STREAMING)
    started_runner._record_step(_step(started_runner))
    assert _step_facts(started_runner) == [{"step": 1, "saved": False, "error": None}]
    _expect_state(started_runner, SessionState.CLOSING)


async def test_each_session_start_posted_to_the_model_is_counted(started_runner: Runner) -> None:
    assert started_runner._sessions_posted == 0
    started_runner.start_session({})
    started_runner.stop_session()
    await started_runner._drain_teardown()
    started_runner.start_session({})
    assert started_runner._sessions_posted == 2


async def test_a_session_without_steps_never_stops_on_its_own(started_runner: Runner) -> None:
    started_runner.start_session({})
    for _ in range(10):
        started_runner._record_step(_step(started_runner))

    assert _moves(started_runner, SessionEvent.STOP_SESSION) == []
    _expect_state(started_runner, SessionState.WAITING)


async def test_a_session_that_reaches_its_steps_tells_its_clients(started_runner: Runner) -> None:
    started_runner.start_session({"steps": 1})
    conn = FakeConnection(1002)
    started_runner.connection_opened(conn)
    started_runner._record_step(_step(started_runner))
    await started_runner._drain_teardown()

    frame = next(f for f in conn.sent if isinstance(f, str) and "sessionEnded" in f)
    reason = json.loads(frame)["data"]["data"]["reason"]
    assert reason == "Session ended: the requested steps are complete."
    assert len(reason) <= 64
    assert conn.closed


async def test_a_step_reported_from_the_model_thread_is_journalled_on_the_loop(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    await asyncio.to_thread(started_runner._on_step_completed, _step(started_runner))
    await asyncio.sleep(0)

    assert _step_facts(started_runner) == [{"step": 1, "saved": False, "error": None}]


async def test_a_step_reported_before_a_crash_is_journalled_before_the_eviction(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})

    def report_then_crash() -> None:
        started_runner._on_step_completed(_step(started_runner))
        started_runner._on_model_failure(RuntimeError("boom"))

    await asyncio.to_thread(report_then_crash)
    await asyncio.sleep(0)

    events = [
        e.transition.event
        for e in _egress(started_runner)
        if isinstance(e, TransitionEvent)
        and e.transition.event in {SessionEvent.STEP_COMPLETED, SessionEvent.EVICTION}
    ]
    assert events == [SessionEvent.STEP_COMPLETED, SessionEvent.EVICTION]


async def test_the_descriptor_echoes_how_many_starting_commands_apply(
    started_runner: Runner,
) -> None:
    assert "starting_input" not in started_runner.descriptor()
    started_runner.start_session(
        {
            "starting_input": {
                "state": {"mode": "a"},
                "commands": [{"command": "set_mode", "data": {"mode": "b"}}],
            }
        }
    )
    assert started_runner.descriptor()["starting_input"] == {"applied": 2}
    started_runner.stop_session()
    await started_runner._drain_teardown()
    assert "starting_input" not in started_runner.descriptor()


def test_the_descriptor_says_whether_step_results_are_kept() -> None:
    assert _runner().descriptor()["step_results"] == {"enabled": False}
    keeping = Runner(
        RuntimeConfig(model_ref="fake:Model", step_results=StepResultsConfig(enabled=True))
    )
    assert keeping.descriptor()["step_results"] == {"enabled": True}


_SAVING_SESSION_ID = "3a1b2c3d-0000-4000-8000-0000000000aa"


@pytest.fixture
async def saving_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    created_models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(
        RuntimeConfig(model_ref="fake:Model", step_results=StepResultsConfig(enabled=True))
    )
    assert runner.step_store is not None
    runner.step_store._root = tmp_path / "steps"
    await runner.start()
    try:
        yield runner
    finally:
        await runner.stop()


async def _saved_facts(runner: Runner, count: int) -> list[dict[str, Any]]:
    """Wait for *count* step_result_ready facts and return their details."""
    for _ in range(500):
        facts = [dict(move.detail) for move in _moves(runner, SessionEvent.STEP_RESULT_READY)]
        if len(facts) >= count:
            return facts
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected {count} step_result_ready facts")


def _saved_result(runner: Runner, step: int) -> dict[str, Any]:
    store = runner.step_store
    assert store is not None
    assert store.root is not None
    path = store.root / _SAVING_SESSION_ID / str(step) / "result.json"
    return json.loads(path.read_text())


async def test_a_saved_step_is_journalled_saved_then_ready(saving_runner: Runner) -> None:
    saving_runner.start_session({"session_id": _SAVING_SESSION_ID})
    saving_runner._record_step(_step(saving_runner))

    assert _step_facts(saving_runner) == [{"step": 1, "saved": True, "error": None}]
    assert await _saved_facts(saving_runner, 1) == [
        {"session_id": _SAVING_SESSION_ID, "step": 1, "files": []}
    ]
    assert _saved_result(saving_runner, 1)["step"] == 1


async def test_a_saved_step_keeps_the_messages_broadcast_since_the_step_before(
    saving_runner: Runner,
) -> None:
    saving_runner.start_session({"session_id": _SAVING_SESSION_ID})

    def model_thread() -> None:
        saving_runner._broadcast_message(Greeting(text="before the first step"))
        saving_runner._on_step_completed(_step(saving_runner))
        saving_runner._on_step_completed(_step(saving_runner))

    await asyncio.to_thread(model_thread)
    await _saved_facts(saving_runner, 2)

    assert _saved_result(saving_runner, 1)["messages"] == [
        {"type": "greeting", "data": {"text": "before the first step"}}
    ]
    assert _saved_result(saving_runner, 2)["messages"] == []


async def test_messages_from_before_the_session_are_not_kept(saving_runner: Runner) -> None:
    saving_runner._broadcast_message(Greeting(text="between sessions"))
    saving_runner.start_session({"session_id": _SAVING_SESSION_ID})
    await asyncio.to_thread(saving_runner._on_step_completed, _step(saving_runner))
    await _saved_facts(saving_runner, 1)

    assert _saved_result(saving_runner, 1)["messages"] == []


async def test_a_late_report_leaves_the_current_sessions_messages(
    saving_runner: Runner,
) -> None:
    saving_runner.start_session({"session_id": _SAVING_SESSION_ID})
    previous = _step(saving_runner)
    saving_runner.stop_session()
    await saving_runner._drain_teardown()
    saving_runner.start_session({"session_id": _SAVING_SESSION_ID})

    def model_thread() -> None:
        saving_runner._broadcast_message(Greeting(text="sent in the new session"))
        saving_runner._on_step_completed(previous)
        saving_runner._on_step_completed(_step(saving_runner))

    await asyncio.to_thread(model_thread)
    await _saved_facts(saving_runner, 1)

    assert _saved_result(saving_runner, 1)["messages"] == [
        {"type": "greeting", "data": {"text": "sent in the new session"}}
    ]


async def test_a_step_after_the_steps_stop_is_saved_too(saving_runner: Runner) -> None:
    saving_runner.start_session({"session_id": _SAVING_SESSION_ID, "steps": 1})
    saving_runner._record_step(_step(saving_runner))
    saving_runner._record_step(_step(saving_runner))

    assert [fact["saved"] for fact in _step_facts(saving_runner)] == [True, True]
    assert [fact["step"] for fact in await _saved_facts(saving_runner, 2)] == [1, 2]


async def test_a_session_whose_id_is_not_a_uuid_saves_nothing(saving_runner: Runner) -> None:
    saving_runner.start_session({"session_id": "my-session"})
    saving_runner._record_step(_step(saving_runner))

    assert _step_facts(saving_runner) == [{"step": 1, "saved": False, "error": None}]


async def test_without_step_results_no_message_is_kept(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner._broadcast_message(Greeting(text="hello"))

    assert started_runner.step_store is None
    assert list(started_runner._step_messages) == []


async def test_a_session_without_a_starting_input_echoes_none(started_runner: Runner) -> None:
    started_runner.start_session({})
    assert "starting_input" not in started_runner.descriptor()


async def test_the_next_session_starts_from_its_own_shape(started_runner: Runner) -> None:
    started_runner.start_session({"steps": 3})
    started_runner.stop_session()
    await started_runner._drain_teardown()
    started_runner.start_session({})
    assert started_runner._session_start == SessionStart()


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


async def test_moderated_stop_closes_a_running_session(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner.stop_session(moderated=True)
    assert started_runner._sm.current_state is SessionState.CLOSING


async def test_moderated_stop_rejects_when_no_session_is_running(started_runner: Runner) -> None:
    with pytest.raises(SessionTransitionError) as rejected:
        started_runner.stop_session(moderated=True)
    assert rejected.value.action == "stop"
    assert rejected.value.state is SessionState.READY


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


async def test_the_published_name_titles_the_schema_and_the_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The served document has to agree with the one the schema command renders
    # from the same manifest, and the descriptor has to name the same model, so
    # all three read the published name.
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: FakeModel)
    runner = Runner(RuntimeConfig(model_ref="fake:Model", model_name="mage-vl"))
    await runner.start()
    try:
        runner.start_session({})
        assert runner.schema()["info"]["title"] == "mage-vl"
        assert runner.descriptor()["model"]["name"] == "mage-vl"
    finally:
        await runner.stop()


async def test_schema_falls_back_to_the_class_when_no_name_is_published(
    started_runner: Runner,
) -> None:
    assert started_runner.schema()["info"]["title"] == "fake_model"


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


async def test_moderated_stop_ends_the_session_as_moderated(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session({})
    started_runner.stop_session(moderated=True)
    await asyncio.sleep(0.01)
    ended = [e for e in events if isinstance(e, SessionEnded)]
    assert len(ended) == 1
    assert ended[0].reason is EndReason.MODERATED


async def test_moderated_stop_notifies_every_client_before_teardown(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    v0_conn = FakeConnection(1)
    v1_conn = FakeConnection(2)
    v1_conn.protocol_version = V1
    started_runner.connection_opened(v0_conn)
    started_runner.connection_opened(v1_conn)

    started_runner.stop_session(moderated=True)

    # The v0 client sees the legacy runtime-scope JSON on the data channel.
    v0_frame = next(f for f in v0_conn.sent if isinstance(f, str) and "moderation" in f)
    body = json.loads(v0_frame)
    assert body["scope"] == "runtime"
    assert body["data"]["type"] == "moderation"
    assert body["data"]["data"]["action"] == "terminate"
    # The v1 client sees the binary notification on the control channel.
    raw = v1_conn.control[-1]
    assert isinstance(raw, bytes)
    decoded = protocol.select(V1).decode(raw, CONTROL, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    assert decoded.WhichOneof("payload") == "moderation"
    assert decoded.moderation.action == "terminate"


async def test_plain_stop_sends_no_moderation_notice(started_runner: Runner) -> None:
    started_runner.start_session({})
    conn = FakeConnection(1)
    started_runner.connection_opened(conn)

    started_runner.stop_session()

    assert not any(isinstance(f, str) and "moderation" in f for f in conn.sent)


async def test_reasoned_stop_notifies_every_client_before_teardown(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    v0_conn = FakeConnection(1)
    v1_conn = FakeConnection(2)
    v1_conn.protocol_version = V1
    started_runner.connection_opened(v0_conn)
    started_runner.connection_opened(v1_conn)

    started_runner.stop_session(reason="Session ended: the model was updated.")

    # The notice is queued synchronously on entering CLOSING; the connections
    # only close in the teardown task that has not run yet.
    assert not v0_conn.closed
    assert not v1_conn.closed
    # The v0 client sees the legacy runtime-scope JSON on the data channel.
    v0_frame = next(f for f in v0_conn.sent if isinstance(f, str) and "sessionEnded" in f)
    body = json.loads(v0_frame)
    assert body["scope"] == "runtime"
    assert body["data"]["type"] == "sessionEnded"
    assert body["data"]["data"]["reason"] == "Session ended: the model was updated."
    # The v1 client sees the binary notification on the control channel.
    raw = v1_conn.control[-1]
    assert isinstance(raw, bytes)
    decoded = protocol.select(V1).decode(raw, CONTROL, SERVER)
    assert isinstance(decoded, control_pb2.ControlServerMessage)
    assert decoded.WhichOneof("payload") == "session_ended"
    assert decoded.session_ended.reason == "Session ended: the model was updated."

    await asyncio.sleep(0.01)
    assert v0_conn.closed
    assert v1_conn.closed


async def test_reasoned_stop_ends_the_session_as_stopped(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _record_reactor_events(started_runner, monkeypatch)
    started_runner.start_session({})
    started_runner.stop_session(reason="deployment")
    await asyncio.sleep(0.01)
    ended = [e for e in events if isinstance(e, SessionEnded)]
    assert len(ended) == 1
    assert ended[0].reason is EndReason.STOPPED


async def test_the_close_reason_reaches_the_client_verbatim(
    started_runner: Runner,
) -> None:
    # The platform authors the wording; the runtime must not rewrite it.
    started_runner.start_session({})
    conn = FakeConnection(1)
    started_runner.connection_opened(conn)

    started_runner.stop_session(reason="cosmic rays flipped a bit")

    frame = next(f for f in conn.sent if isinstance(f, str) and "sessionEnded" in f)
    body = json.loads(frame)
    assert body["data"]["data"]["reason"] == "cosmic rays flipped a bit"


async def test_plain_stop_sends_no_session_ended_notice(started_runner: Runner) -> None:
    started_runner.start_session({})
    conn = FakeConnection(1)
    started_runner.connection_opened(conn)

    started_runner.stop_session()

    assert not any(isinstance(f, str) and "sessionEnded" in f for f in conn.sent)


async def test_a_moderated_and_reasoned_stop_sends_only_the_moderation_notice(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    conn = FakeConnection(1)
    started_runner.connection_opened(conn)

    started_runner.stop_session(moderated=True, reason="deployment")

    assert any(isinstance(f, str) and "moderation" in f for f in conn.sent)
    assert not any(isinstance(f, str) and "sessionEnded" in f for f in conn.sent)


async def test_reasoned_stop_without_a_client_still_stops(started_runner: Runner) -> None:
    started_runner.start_session({})
    started_runner.stop_session(reason="deployment")
    assert started_runner._sm.current_state is SessionState.CLOSING


async def test_reasoned_stop_survives_a_failing_broadcast(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    started_runner.connection_opened(FakeConnection(1))

    def explode(encode: Any) -> None:
        raise RuntimeError("wire fell over")

    monkeypatch.setattr(started_runner._connections, "broadcast_response", explode)

    started_runner.stop_session(reason="deployment")

    # The teardown still runs to completion: the session unwinds to READY
    # rather than stranding in CLOSING behind the failed notice.
    await asyncio.sleep(0.01)
    assert started_runner._sm.current_state is SessionState.READY


async def test_drain_ends_an_active_session_within_grace(started_runner: Runner) -> None:
    started_runner.start_session({})
    await started_runner.drain()
    assert started_runner._sm.current_state is SessionState.READY


async def test_drain_tells_clients_the_server_is_stopping(started_runner: Runner) -> None:
    # The runtime initiates a drain stop, so it authors the close reason; the
    # notice must reach the client before the drain closes its connection.
    started_runner.start_session({})
    conn = FakeConnection(1)
    started_runner.connection_opened(conn)

    await started_runner.drain()

    frame = next(f for f in conn.sent if isinstance(f, str) and "sessionEnded" in f)
    body = json.loads(frame)
    assert body["data"]["data"]["reason"] == _DRAIN_CLOSE_REASON
    # The bound the stop route enforces on platform reasons; ours obeys it too.
    assert len(_DRAIN_CLOSE_REASON) <= 64
    assert conn.closed


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
        name="set_mode",
        args={"mode": "fast"},
        uploads={},
        conn_id=ConnId(1),
        request_id="r1",
        received_at=time.monotonic(),
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
        name="set_mode",
        args={"mode": ""},
        uploads={},
        conn_id=ConnId(1),
        request_id="r1",
        received_at=time.monotonic(),
    )
    await started_runner._submit_command(command)
    errors = _moves(started_runner, SessionEvent.ERROR)
    assert len(errors) == 1
    assert "set_mode" in errors[0].detail["message"]
    assert not _moves(started_runner, SessionEvent.COMMAND)


async def test_rejected_command_error_acks_the_v1_sender(started_runner: Runner) -> None:
    started_runner.start_session({})
    conn = FakeConnection(1)
    conn.protocol_version = V1
    started_runner.connection_opened(conn)
    command = InboundCommand(
        name="set_mode",
        args={"mode": ""},
        uploads={},
        conn_id=ConnId(1),
        request_id="r1",
        received_at=time.monotonic(),
    )
    await started_runner._submit_command(command)
    frames = [protocol.select(V1).decode(frame, DATA, SERVER) for frame in conn.sent]
    errors = [
        frame
        for frame in frames
        if isinstance(frame, data_pb2.DataServerMessage) and frame.WhichOneof("payload") == "error"
    ]
    assert len(errors) == 1
    assert errors[0].request_id == "r1"
    assert errors[0].error.code == "invalid_command"


async def test_init_failure_requests_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ref: str) -> type:
        raise RuntimeError("no such model")

    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", boom)
    runner = _runner()
    called: list[bool] = []
    runner.request_shutdown = lambda *, failure=False: called.append(failure)

    await runner.start()

    assert runner._sm.current_state is SessionState.TERMINATED
    # A model that refused to load is a clean exit, not a failure: the process
    # had nothing to serve and says so with a zero status.
    assert called == [False]


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
    runner.request_shutdown = lambda *, failure=False: called.append(failure)

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
        # A crashed loop asks for the process to go down as a failure, so the
        # exit status tells the orchestrator this was not a clean stop.
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


class PlainModel(ReactorApp):
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
        received_at=time.monotonic(),
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

    async def never_arrives(upload_id: str, **kwargs: Any) -> UploadedFile:
        raise UnknownUploadError(upload_id)

    assert started_runner._bridge is not None
    monkeypatch.setattr(started_runner._bridge, "submit_command", fake_submit)
    # The store gives up only after the upload timeout, which this outcome does
    # not depend on, so it gives up at once here.
    monkeypatch.setattr(started_runner._uploads, "fetch", never_arrives)

    command = InboundCommand(
        name="set_image",
        args={},
        uploads={"image": "missing"},
        conn_id=ConnId(1),
        request_id="r1",
        received_at=time.monotonic(),
    )
    await started_runner._submit_command(command)

    assert submitted == []
    errors = _moves(started_runner, SessionEvent.ERROR)
    assert len(errors) == 1
    assert "set_image" in errors[0].detail["message"]


def _metric(runner: Runner, name: str, **labels: str) -> float | None:
    """Read one sample off the registry the runner observes on."""
    return runner._metrics.registry.get_sample_value(name, labels or None)


async def _submit(runner: Runner, name: str, args: dict[str, Any], **extra: Any) -> None:
    """Submit one command through the runner's own choke point."""
    await runner._submit_command(
        InboundCommand(
            name=name,
            args=args,
            uploads=extra.pop("uploads", {}),
            conn_id=ConnId(1),
            request_id="r1",
            received_at=extra.pop("received_at", time.monotonic()),
        )
    )


async def test_an_accepted_command_is_counted_and_timed(started_runner: Runner) -> None:
    started_runner.start_session({})

    await _submit(started_runner, "set_mode", {"mode": "fast"})

    assert _metric(started_runner, "runtime_commands_total", command="set_mode", outcome="accepted")
    assert _metric(started_runner, "runtime_command_ingress_seconds_count", command="set_mode") == 1


async def test_a_rejected_command_is_counted_and_timed(started_runner: Runner) -> None:
    started_runner.start_session({})

    await _submit(started_runner, "set_mode", {"mode": ""})

    # A rejection costs the same ingress work an acceptance does, so it is timed
    # too and only the outcome tells them apart.
    assert _metric(started_runner, "runtime_commands_total", command="set_mode", outcome="rejected")
    assert _metric(started_runner, "runtime_command_ingress_seconds_count", command="set_mode") == 1


async def test_the_ingress_starts_when_the_frame_arrived(started_runner: Runner) -> None:
    started_runner.start_session({})

    await _submit(started_runner, "set_mode", {"mode": "fast"}, received_at=time.monotonic() - 5.0)

    # The measurement runs from the stamp the transport edge put on the frame, so
    # a frame that waited five seconds for the loop reports five seconds.
    total = _metric(started_runner, "runtime_command_ingress_seconds_sum", command="set_mode")
    assert total is not None
    assert total >= 5.0


async def test_the_wait_for_uploaded_bytes_is_left_out_of_the_ingress(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})

    async def arrives_late(upload_id: str, **kwargs: Any) -> UploadedFile:
        await asyncio.sleep(0.3)
        return UploadedFile(name="cat.png", mime_type="image/png", data=b"\x89PNG")

    monkeypatch.setattr(started_runner._uploads, "fetch", arrives_late)
    await _submit(started_runner, "set_image", {}, uploads={"image": "late"})

    # A client that takes its time sending a file is not a runtime that is slow to
    # carry a command. Counting the wait would put a client's upload speed in the
    # tail of the histogram and hide the starved loop it exists to expose.
    total = _metric(started_runner, "runtime_command_ingress_seconds_sum", command="set_image")
    assert total is not None
    assert total < 0.3


async def test_every_command_the_model_declares_starts_at_zero(started_runner: Runner) -> None:
    # A command nobody sent has no series, which reads the same as a command the
    # model does not have. The seed makes "nobody uses this one" answerable.
    assert (
        _metric(started_runner, "runtime_commands_total", command="set_mode", outcome="accepted")
        == 0.0
    )
    assert (
        _metric(started_runner, "runtime_commands_total", command="set_mode", outcome="rejected")
        == 0.0
    )


async def test_an_unresolved_upload_is_counted_with_no_ingress(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})

    async def never_arrives(upload_id: str, **kwargs: Any) -> UploadedFile:
        raise UnknownUploadError(upload_id)

    # The store gives up only after the upload timeout, which the outcome under
    # test does not depend on, so it gives up at once here.
    monkeypatch.setattr(started_runner._uploads, "fetch", never_arrives)
    await _submit(started_runner, "set_image", {}, uploads={"image": "missing"})

    assert _metric(
        started_runner, "runtime_commands_total", command="set_image", outcome="unresolved_upload"
    )
    # Ingress measures a command the runtime carried to the model, and this one
    # never got there, so it contributes no measurement.
    assert (
        _metric(started_runner, "runtime_command_ingress_seconds_count", command="set_image")
        is None
    )


# --- inline upload references ---------------------------------------------


def _seed_upload(runner: Runner, name: str, data: bytes) -> str:
    upload_id = runner.uploads.create_slot(name, "image/png", len(data))
    runner.uploads.put(upload_id, data)
    return upload_id


def _capture_submissions(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, dict[str, Any], CommandOutcome]]:
    """Record what reaches the bridge while still validating it against the contract."""
    assert runner._bridge is not None
    original = runner._bridge.submit_command
    submitted: list[tuple[str, dict[str, Any], CommandOutcome]] = []

    async def capture(name: str, args: dict[str, Any], **kwargs: Any) -> CommandOutcome:
        outcome = await original(name, args, **kwargs)
        submitted.append((name, args, outcome))
        return outcome

    monkeypatch.setattr(runner._bridge, "submit_command", capture)
    return submitted


async def test_a_list_of_references_reaches_the_model_as_files_in_the_client_order(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    first = _seed_upload(started_runner, "first.png", b"1")
    second = _seed_upload(started_runner, "second.png", b"2")
    third = _seed_upload(started_runner, "third.png", b"3")
    submitted = _capture_submissions(started_runner, monkeypatch)

    await _submit(
        started_runner,
        "set_images",
        {"images": [{"upload_id": third}, {"upload_id": first}, {"upload_id": second}]},
    )

    name, args, outcome = submitted[0]
    assert (name, outcome.accepted) == ("set_images", True)
    assert args["images"] == [
        UploadedFile(name="third.png", mime_type="image/png", data=b"3"),
        UploadedFile(name="first.png", mime_type="image/png", data=b"1"),
        UploadedFile(name="second.png", mime_type="image/png", data=b"2"),
    ]
    # The typed command the handler is called with carries the same files.
    assert started_runner._bridge is not None
    command = started_runner._bridge.contract.validate("set_images", args)
    assert all(isinstance(image, UploadedFile) for image in vars(command)["images"])


async def test_a_single_file_resolves_through_either_channel(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    upload_id = _seed_upload(started_runner, "cat.png", b"cat")
    submitted = _capture_submissions(started_runner, monkeypatch)
    expected = UploadedFile(name="cat.png", mime_type="image/png", data=b"cat")

    await _submit(started_runner, "set_image", {}, uploads={"image": upload_id})
    await _submit(started_runner, "set_image", {"image": {"upload_id": upload_id}})

    assert [(args["image"], outcome.accepted) for _name, args, outcome in submitted] == [
        (expected, True),
        (expected, True),
    ]


async def test_a_sidecar_file_and_an_inline_list_resolve_in_one_command(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    cover = _seed_upload(started_runner, "cover.png", b"c")
    page = _seed_upload(started_runner, "page.png", b"p")
    submitted = _capture_submissions(started_runner, monkeypatch)

    await _submit(
        started_runner,
        "set_gallery",
        {"images": [{"upload_id": page}], "labels": {"upload_id": "not a reference"}},
        uploads={"cover": cover},
    )

    _name, args, outcome = submitted[0]
    assert outcome.accepted
    assert args["cover"] == UploadedFile(name="cover.png", mime_type="image/png", data=b"c")
    assert args["images"] == [UploadedFile(name="page.png", mime_type="image/png", data=b"p")]
    # A mapping the model declared as its own is not an upload, whatever its keys.
    assert args["labels"] == {"upload_id": "not a reference"}


async def test_references_nested_in_a_dict_and_a_dataclass_reach_the_model_as_files(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    left = _seed_upload(started_runner, "left.png", b"l")
    right = _seed_upload(started_runner, "right.png", b"r")
    cover = _seed_upload(started_runner, "cover.png", b"c")
    submitted = _capture_submissions(started_runner, monkeypatch)

    await _submit(
        started_runner,
        "set_book",
        {
            "pages": {"left": {"upload_id": left}, "right": {"upload_id": right}},
            "cover": {"image": {"upload_id": cover}, "caption": "front"},
        },
    )

    _name, args, outcome = submitted[0]
    assert outcome.accepted
    assert args["pages"] == {
        "left": UploadedFile(name="left.png", mime_type="image/png", data=b"l"),
        "right": UploadedFile(name="right.png", mime_type="image/png", data=b"r"),
    }
    assert args["cover"] == {
        "image": UploadedFile(name="cover.png", mime_type="image/png", data=b"c"),
        "caption": "front",
    }


async def test_an_absent_optional_list_needs_no_resolution(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    cover = _seed_upload(started_runner, "cover.png", b"c")
    submitted = _capture_submissions(started_runner, monkeypatch)

    await _submit(started_runner, "set_gallery", {"images": None}, uploads={"cover": cover})

    _name, args, outcome = submitted[0]
    assert outcome.accepted
    assert args["images"] is None


async def test_a_list_with_one_unresolved_entry_drops_the_whole_command(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})
    present = _seed_upload(started_runner, "cat.png", b"cat")
    submitted = _capture_submissions(started_runner, monkeypatch)
    rejected: list[tuple[ConnId, str, str, str]] = []
    monkeypatch.setattr(
        started_runner,
        "_reject_command",
        lambda conn_id, request_id, code, reason: rejected.append(
            (conn_id, request_id, code, reason)
        ),
    )
    original_fetch = started_runner._uploads.fetch

    async def no_grace(upload_id: str, **kwargs: Any) -> UploadedFile:
        # The store waits the upload grace for a missing id, which this outcome
        # does not depend on, so a missing id fails at once here.
        return await original_fetch(upload_id)

    monkeypatch.setattr(started_runner._uploads, "fetch", no_grace)

    await _submit(
        started_runner,
        "set_images",
        {"images": [{"upload_id": present}, {"upload_id": "missing"}]},
    )

    assert submitted == []
    assert rejected == [
        (ConnId(1), "r1", UNRESOLVED_UPLOAD, "command 'set_images' references an unresolved upload")
    ]
    errors = _moves(started_runner, SessionEvent.ERROR)
    assert len(errors) == 1
    assert "set_images" in errors[0].detail["message"]
    assert _metric(
        started_runner, "runtime_commands_total", command="set_images", outcome="unresolved_upload"
    )


async def test_the_journal_carries_the_references_of_a_list_and_never_its_bytes(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    upload_id = _seed_upload(started_runner, "cat.png", b"cat")

    await _submit(started_runner, "set_images", {"images": [{"upload_id": upload_id}]})

    commands = _moves(started_runner, SessionEvent.COMMAND)
    assert len(commands) == 1
    assert commands[0].detail["args"] == {"images": [{"upload_id": upload_id}]}


async def test_the_wait_for_a_list_of_uploads_is_left_out_of_the_ingress(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    started_runner.start_session({})

    async def arrives_late(upload_id: str, **kwargs: Any) -> UploadedFile:
        await asyncio.sleep(0.3)
        return UploadedFile(name=f"{upload_id}.png", mime_type="image/png", data=b"\x89PNG")

    monkeypatch.setattr(started_runner._uploads, "fetch", arrives_late)
    await _submit(
        started_runner, "set_images", {"images": [{"upload_id": "a"}, {"upload_id": "b"}]}
    )

    # A list waits for its entries together, and the wait is the client's, so a
    # command carrying a list is timed the same way one carrying a single file is.
    total = _metric(started_runner, "runtime_command_ingress_seconds_sum", command="set_images")
    assert total is not None
    assert total < 0.3
    assert _metric(
        started_runner, "runtime_commands_total", command="set_images", outcome="accepted"
    )


async def test_a_frame_off_the_wire_is_timed_from_the_moment_it_arrived(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    _, frame = protocol.select(V1).encode(
        data_pb2.DataClientMessage(
            request_id="req-1",
            command=model_pb2.Command(type="set_mode", data=dict_to_struct({"mode": "fast"})),
        )
    )

    # The whole path, not just the choke point: the transport edge stamps the
    # arrival, the gateway carries the stamp onto the command it decodes, and the
    # submit path measures against it.
    started_runner.message_received(ConnId(1), frame, V1, DATA)
    await asyncio.sleep(0.05)

    assert _metric(started_runner, "runtime_commands_total", command="set_mode", outcome="accepted")
    assert _metric(started_runner, "runtime_command_ingress_seconds_count", command="set_mode") == 1


async def test_a_command_the_model_does_not_declare_shares_one_series(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})

    await _submit(started_runner, "definitely_not_a_command", {})
    await _submit(started_runner, "also_not_a_command", {})

    # A client names the command, so the name is not a bounded label value. Only
    # the schema is bounded, and everything outside it shares one series.
    assert (
        _metric(started_runner, "runtime_commands_total", command="unknown", outcome="rejected")
        == 2
    )
    rendered = started_runner._metrics.render().decode()
    assert "definitely_not_a_command" not in rendered


async def test_a_model_that_came_up_is_measured(started_runner: Runner) -> None:
    assert _metric(started_runner, "runtime_model_load_seconds_count", outcome="ok") == 1.0


async def test_a_model_that_failed_to_load_is_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unloadable(FakeModel):
        def load(self, config_path: Path | None) -> None:
            raise RuntimeError("weights missing")

    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: Unloadable)
    runner = _runner()

    await runner.start()
    try:
        # A failed load is terminal, so this is the one observation the process
        # ever makes, and it says how long it spent before it gave up.
        assert _metric(runner, "runtime_model_load_seconds_count", outcome="failed") == 1.0
        assert _metric(runner, "runtime_model_load_seconds_count", outcome="ok") is None
    finally:
        await runner.stop()


async def test_emitted_media_is_counted_in_frames_per_track(started_runner: Runner) -> None:
    started_runner.start_session({})
    bundle = MediaBundle(
        tracks={
            "main_video": TrackData(
                info=TrackInfo(name="main_video", kind=TrackKind.VIDEO),
                data=np.zeros((4, 2, 2, 3), dtype=np.uint8),
            ),
            "main_audio": TrackData(
                info=TrackInfo(name="main_audio", kind=TrackKind.AUDIO),
                data=np.zeros((4, 2), dtype=np.float32),
            ),
        }
    )

    started_runner._emit_media(MediaChunk(bundle=bundle, fps=30.0, n_frames=4))

    # Counted in frames, not in emissions: the model batches four frames into one
    # chunk here, and a counter of chunks would report a quarter of the real rate.
    assert _metric(started_runner, "runtime_media_frames_total", track="main_video") == 4.0
    assert _metric(started_runner, "runtime_media_frames_total", track="main_audio") == 4.0


async def test_the_sessions_own_output_counts_start_over_with_each_session(
    started_runner: Runner,
) -> None:
    # The process-wide counter keeps growing; the per-session count that feeds
    # the runtime's stats starts from zero for every session.
    started_runner.start_session({})
    started_runner._emit_media(MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=3))
    assert started_runner._model_output.take()["main"]["frames_emitted"] == 3.0

    started_runner.stop_session()
    await started_runner._drain_teardown()
    started_runner.start_session({})

    assert started_runner._model_output.take()["main"]["frames_emitted"] == 0.0


async def test_frames_the_model_emits_as_the_session_starts_are_counted(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model hears of the session while the start transition is dispatched,
    # and its thread can emit right away. Those frames belong to the session.
    dispatch = started_runner._dispatch_reactor_events

    def dispatch_then_emit(transition: Transition, bridge: Any) -> None:
        dispatch(transition, bridge)
        if transition.is_session_start:
            started_runner._model_output.emitted("main", 2)

    monkeypatch.setattr(started_runner, "_dispatch_reactor_events", dispatch_then_emit)

    started_runner.start_session({})

    assert started_runner._model_output.take()["main"]["frames_emitted"] == 2.0


async def test_a_late_chunk_of_an_earlier_session_is_not_counted_for_the_next(
    started_runner: Runner,
) -> None:
    # The model can still emit for the session that ended (from its session-end
    # hook, say) after the next one starts. That output plays out, but it is
    # not the new session's.
    started_runner.start_session({})
    current = started_runner._sessions_posted

    for session, frames in ((current - 1, 5), (current, 2), (None, 1)):
        started_runner._emit_media(
            MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=frames, session=session)
        )

    assert started_runner._model_output.take()["main"]["frames_emitted"] == 3.0


async def test_a_late_chunk_landing_as_the_count_starts_over_is_not_counted(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model's thread emits while the loop starts the next session, so an
    # earlier session's chunk can land the moment the count starts over.
    started_runner.start_session({})
    started_runner.stop_session()
    await started_runner._drain_teardown()
    earlier = started_runner._sessions_posted
    output = started_runner._model_output
    reset = output.reset

    def reset_then_emit_late(session: int | None = None) -> None:
        reset(session)
        started_runner._emit_media(
            MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=4, session=earlier)
        )

    monkeypatch.setattr(output, "reset", reset_then_emit_late)

    started_runner.start_session({})

    assert output.take()["main"]["frames_emitted"] == 0.0


async def test_a_rejected_start_leaves_the_running_sessions_counts_alone(
    started_runner: Runner,
) -> None:
    started_runner.start_session({})
    started_runner._emit_media(MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=3))

    with pytest.raises(SessionTransitionError):
        started_runner.start_session({})

    assert started_runner._model_output.take()["main"]["frames_emitted"] == 3.0


async def test_media_reaches_the_connections_before_the_recorder(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Feeding the recorder can make the fan-out wait when the encoder is behind,
    # bounded but real, while a pacer that waits is throttling to the playout
    # rate it was asked for. Serving the connections first keeps a stalled
    # archive off the live path.
    started_runner.start_session({})
    served: list[str] = []
    monkeypatch.setattr(
        started_runner._connections,
        "broadcast_media",
        lambda *_a, **_k: served.append("connections"),
    )
    monkeypatch.setattr(started_runner._recorder, "on_chunk", lambda _c: served.append("recorder"))

    started_runner._emit_media(MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=1))

    assert served == ["connections", "recorder"]


async def test_a_flush_that_cuts_the_broadcast_short_still_records_the_chunk(
    started_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A playout cut is not an archive boundary. With the recorder served last,
    # a flush landing mid-broadcast abandons the rest of the fan-out, and the
    # archive still has to receive the whole chunk.
    started_runner.start_session({})
    recorded: list[MediaChunk] = []
    monkeypatch.setattr(started_runner._recorder, "on_chunk", lambda c: recorded.append(c))
    monkeypatch.setattr(
        started_runner._connections,
        "broadcast_media",
        lambda *_a, **_k: started_runner._flush_media(),
    )
    chunk = MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=1)

    started_runner._emit_media(chunk)

    assert recorded == [chunk]


async def test_every_output_track_the_model_declares_starts_at_zero(
    started_runner: Runner,
) -> None:
    # A track that has carried nothing reads zero, which is what tells a silent
    # track apart from a track this model does not have.
    assert _metric(started_runner, "runtime_media_frames_total", track="main") == 0.0


async def test_the_gap_between_two_emissions_is_measured(started_runner: Runner) -> None:
    started_runner.start_session({})
    chunk = MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=1)

    started_runner._emit_media(chunk)
    # The first emission has nothing to be measured against.
    assert _interval_count(started_runner) is None

    started_runner._emit_media(chunk)

    assert _interval_count(started_runner) == 1.0


async def test_no_gap_is_measured_across_a_session_boundary(started_runner: Runner) -> None:
    chunk = MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=1)
    started_runner.start_session({})
    started_runner._emit_media(chunk)
    started_runner.stop_session()
    await asyncio.sleep(0.01)

    started_runner.start_session({})
    started_runner._emit_media(chunk)

    # The wait between one client leaving and the next arriving is idle time, and
    # counting it would report the pause as the model's worst stall.
    assert _interval_count(started_runner) is None


def _interval_count(runner: Runner) -> float | None:
    """How many gaps between emissions were measured on FakeModel's one track."""
    return _metric(runner, "runtime_media_emit_interval_seconds_count", track="main")


def _video_bundle() -> MediaBundle:
    """A one-frame bundle on the track FakeModel declares."""
    return MediaBundle(
        tracks={
            "main": TrackData(
                info=TrackInfo(name="main", kind=TrackKind.VIDEO),
                data=np.zeros((2, 2, 3), dtype=np.uint8),
            )
        }
    )


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
    runner.request_shutdown = lambda *, failure=False: called.append(failure)
    await runner.start()
    assert called == [False]

    # An error journalled after the terminal move self-loops in TERMINATED
    # without asking the service to bring the process down a second time.
    assert runner._sm.send(SessionEvent.ERROR, message="late") is True
    assert runner._sm.current_state is SessionState.TERMINATED
    assert called == [False]


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


_LIVE_SESSION_ID = "0f0e0d0c-0b0a-0908-0706-050403020100"


def _stamped_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return *record*'s fields as the handler writes them, session context included."""
    log.SessionContextFilter().filter(record)
    return getattr(record, "reactor_fields", {})


async def test_client_stats_are_logged_with_session_and_connection_identity(
    started_runner: Runner, caplog: pytest.LogCaptureFixture
) -> None:
    stat = ClientTrackStat(
        timestamp=1_700_000_000_000,
        track_name="main_video",
        kind=TrackKind.VIDEO,
        direction=ClientTrackDirection.RECVONLY,
        codec="VP9",
        paused=False,
        metrics={
            "bitrate_bps": 950_000,
            "frames_per_second": 29.5,
            "packets_lost": 4,
            "packets_received": 996,
            "jitter_ms": 12.0,
            "round_trip_time_ms": 48.0,
            "frames_decoded": 900,
            "frames_dropped": 2,
            "frame_width": 1280,
            "frame_height": 720,
            "nack_count": 3,
            "keyframe_requests": 1,
        },
    )
    connection_stat = ClientConnectionStat(
        timestamp=1_700_000_000_000,
        metrics={"available_outgoing_bitrate_bps": 2_000_000, "time_to_connect_ms": 850},
    )
    batch = ClientStatsBatch(track_stats=[stat], connection_stat=connection_stat)
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    with caplog.at_level(logging.DEBUG, logger="reactor_runtime.runner.runner"):
        started_runner.client_stats_received(ConnId(3), batch)

    connection_record = next(
        r for r in caplog.records if r.getMessage() == "client connection stats"
    )
    connection_fields = _stamped_fields(connection_record)
    # The id the session is known by, not the fixed transport id.
    assert connection_fields["session_id"] == _LIVE_SESSION_ID
    assert connection_fields["conn_id"] == ConnId(3)
    assert connection_fields["metrics"] == {
        "available_outgoing_bitrate_bps": 2_000_000,
        "time_to_connect_ms": 850,
    }

    record = next(r for r in caplog.records if r.getMessage() == "client stats")
    # Both lines stay below the default log level.
    assert connection_record.levelno == logging.DEBUG
    assert record.levelno == logging.DEBUG
    fields = _stamped_fields(record)
    assert fields["session_id"] == _LIVE_SESSION_ID
    assert fields["conn_id"] == ConnId(3)
    assert fields["track_name"] == "main_video"
    assert fields["kind"] == "video"
    assert fields["direction"] == "recvonly"
    assert fields["codec"] == "VP9"
    assert fields["paused"] is False
    assert fields["metrics"] == dict(stat.metrics)


async def test_client_stats_metric_names_stay_inside_the_metrics_field(
    started_runner: Runner, caplog: pytest.LogCaptureFixture
) -> None:
    stat = ClientTrackStat(
        timestamp=1_700_000_000_000,
        track_name="main_video",
        kind=TrackKind.VIDEO,
        direction=ClientTrackDirection.RECVONLY,
        codec="VP9",
        paused=False,
        # Metric names are the client's to choose. Named like the runtime's own
        # fields, they stay data inside `metrics` and leave the real fields alone.
        metrics={"session_id": -1, "state": -1, "conn_id": -1, "msg": -1, "bitrate_bps": 950_000},
    )
    batch = ClientStatsBatch(track_stats=[stat], connection_stat=None)
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    with caplog.at_level(logging.DEBUG, logger="reactor_runtime.runner.runner"):
        started_runner.client_stats_received(ConnId(3), batch)

    record = next(r for r in caplog.records if r.getMessage() == "client stats")
    fields = _stamped_fields(record)
    assert fields["session_id"] == _LIVE_SESSION_ID
    assert fields["state"] != -1
    assert fields["conn_id"] == ConnId(3)
    assert fields["metrics"] == dict(stat.metrics)


async def test_client_stats_metric_names_cannot_break_a_text_log_line(
    started_runner: Runner, caplog: pytest.LogCaptureFixture
) -> None:
    stat = ClientTrackStat(
        timestamp=1_700_000_000_000,
        track_name="main_video",
        kind=TrackKind.VIDEO,
        direction=ClientTrackDirection.RECVONLY,
        codec="VP9",
        paused=False,
        metrics={"x\nforged=line": 1.0, "a b=c": 2.0},
    )
    batch = ClientStatsBatch(track_stats=[stat], connection_stat=None)
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    with caplog.at_level(logging.DEBUG, logger="reactor_runtime.runner.runner"):
        started_runner.client_stats_received(ConnId(3), batch)

    record = next(r for r in caplog.records if r.getMessage() == "client stats")
    line = log.TextFormatter().format(record)
    assert "\n" not in line
    assert "forged=line" not in line.split("metrics=", 1)[0]


async def test_client_stats_logs_no_connection_line_when_the_batch_carries_none(
    started_runner: Runner, caplog: pytest.LogCaptureFixture
) -> None:
    stat = ClientTrackStat(
        timestamp=1_700_000_000_000,
        track_name="main_video",
        kind=TrackKind.VIDEO,
        direction=ClientTrackDirection.RECVONLY,
        codec="VP9",
        paused=False,
        metrics={"bitrate_bps": 950_000},
    )
    batch = ClientStatsBatch(track_stats=[stat], connection_stat=None)
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    with caplog.at_level(logging.DEBUG, logger="reactor_runtime.runner.runner"):
        started_runner.client_stats_received(ConnId(3), batch)

    assert [r for r in caplog.records if r.getMessage() == "client connection stats"] == []
    assert [r for r in caplog.records if r.getMessage() == "client stats"]


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


def _one_track_batch(**metrics: float) -> ClientStatsBatch:
    return ClientStatsBatch(
        track_stats=[
            ClientTrackStat(
                timestamp=1_700_000_000_000,
                track_name="main_video",
                kind=TrackKind.VIDEO,
                direction=ClientTrackDirection.RECVONLY,
                codec="VP9",
                paused=False,
                metrics=metrics,
            )
        ],
        connection_stat=ClientConnectionStat(
            timestamp=1_700_000_000_000, metrics={"connection_rtt_ms": 25.0}
        ),
    )


def _client_stats_facts(runner: Runner) -> list[Transition]:
    metrics = _moves(runner, SessionEvent.METRIC)
    return [t for t in metrics if t.detail.get("name") == "client_stats"]


async def test_client_stats_are_journalled_as_a_metric_with_the_session_and_connection(
    started_runner: Runner,
) -> None:
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    state = started_runner._sm.current_state

    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=30.0))

    (fact,) = _client_stats_facts(started_runner)
    # A self-loop like every other journal fact: the session state is unchanged.
    assert fact.from_state is state
    assert fact.to_state is state
    assert dict(fact.detail) == {
        "name": "client_stats",
        "session_id": _LIVE_SESSION_ID,
        "conn_id": ConnId(3),
        "track_stats": [
            {
                "timestamp": 1_700_000_000_000,
                "track_name": "main_video",
                "kind": "video",
                "direction": "recvonly",
                "codec": "VP9",
                "paused": False,
                "metrics": {"frames_per_second": 30.0},
            }
        ],
        "connection_stat": {
            "timestamp": 1_700_000_000_000,
            "metrics": {"connection_rtt_ms": 25.0},
        },
    }


class MeasuredConnection(FakeConnection):
    """A fake whose transport measures its wire."""

    def __init__(self, cid: int, reading: TransportReading | None) -> None:
        super().__init__(cid)
        self.latest_reading = reading


def _runtime_stats_facts(runner: Runner) -> list[Transition]:
    metrics = _moves(runner, SessionEvent.METRIC)
    return [t for t in metrics if t.detail.get("name") == "runtime_stats"]


async def test_runtime_stats_are_journalled_while_a_session_runs(
    started_runner: Runner,
) -> None:
    started_runner._runtime_stats_interval = 0.01
    conn = MeasuredConnection(4, TransportReading(metrics={"connection_rtt_ms": 40.0}))
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    started_runner.connection_opened(conn)
    state = started_runner._sm.current_state
    started_runner._emit_media(MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=2))

    await asyncio.sleep(0.05)

    facts = _runtime_stats_facts(started_runner)
    assert facts, "no runtime_stats reading was journalled"
    first = facts[0]
    # A self-loop like every other journal fact: the session state is unchanged.
    assert first.from_state is state
    assert first.to_state is state
    assert first.detail["session_id"] == _LIVE_SESSION_ID
    (output,) = first.detail["model_output"]
    assert output["track_name"] == "main"
    assert output["metrics"]["frames_emitted"] == 2.0
    (connection,) = first.detail["connection_stats"]
    assert connection["conn_id"] == 4
    assert connection["metrics"]["connection_rtt_ms"] == pytest.approx(40.0)


async def test_runtime_stats_carry_the_model_output_past_a_connection_that_measures_nothing(
    started_runner: Runner,
) -> None:
    # A transport without the stats capability adds no connection reading, and
    # the session still reports what its model emitted.
    started_runner._runtime_stats_interval = 0.01
    started_runner.start_session({})
    unmeasured = FakeConnection(5)
    assert not isinstance(unmeasured, TransportStatsSource)
    started_runner.connection_opened(unmeasured)
    started_runner._emit_media(MediaChunk(bundle=_video_bundle(), fps=30.0, n_frames=1))

    await asyncio.sleep(0.05)

    facts = _runtime_stats_facts(started_runner)
    assert facts
    assert facts[0].detail["connection_stats"] == []
    assert facts[0].detail["model_output"][0]["metrics"]["frames_emitted"] == 1.0


async def test_runtime_stats_stop_when_the_session_ends(started_runner: Runner) -> None:
    started_runner._runtime_stats_interval = 0.01
    started_runner.start_session({})
    await asyncio.sleep(0.03)
    started_runner.stop_session()
    await started_runner._drain_teardown()
    journalled = len(_runtime_stats_facts(started_runner))

    await asyncio.sleep(0.05)

    assert journalled > 0
    assert len(_runtime_stats_facts(started_runner)) == journalled
    assert started_runner._runtime_stats_task is None


async def test_runtime_stats_are_not_journalled_without_a_session(started_runner: Runner) -> None:
    started_runner._runtime_stats_interval = 0.01

    await asyncio.sleep(0.05)

    assert _runtime_stats_facts(started_runner) == []


async def test_client_stats_are_not_journalled_before_a_session_starts(
    started_runner: Runner,
) -> None:
    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=30.0))

    assert _client_stats_facts(started_runner) == []


async def test_client_stats_leave_out_values_json_cannot_carry(started_runner: Runner) -> None:
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})

    started_runner.client_stats_received(
        ConnId(3),
        _one_track_batch(jitter_ms=float("nan"), bitrate_bps=float("inf"), packets_lost=4.0),
    )

    (fact,) = _client_stats_facts(started_runner)
    assert fact.detail["track_stats"][0]["metrics"] == {"packets_lost": 4.0}


async def test_client_stats_are_journalled_for_a_session_started_with_the_all_zero_id(
    started_runner: Runner,
) -> None:
    # The all-zero id is a value a caller may pass like any other; running is
    # judged by the session's state, not by its id.
    all_zero = "00000000-0000-0000-0000-000000000000"
    started_runner.start_session({"session_id": all_zero})

    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=30.0))

    (fact,) = _client_stats_facts(started_runner)
    assert fact.detail["session_id"] == all_zero


async def test_client_stats_are_not_journalled_once_the_session_is_closing(
    started_runner: Runner,
) -> None:
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    started_runner.stop_session()
    assert started_runner._sm.current_state is SessionState.CLOSING

    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=30.0))

    assert _client_stats_facts(started_runner) == []


def _pin_client_stats_clock(runner: Runner, start: float = 1000.0) -> list[float]:
    """Give the runner's client stats gate a clock the test moves by hand."""
    now = [start]
    runner._client_stats = ClientStatsGate(clock=lambda: now[0])
    return now


async def test_client_stats_a_batch_that_comes_too_soon_is_neither_journalled_nor_logged(
    started_runner: Runner, caplog: pytest.LogCaptureFixture
) -> None:
    # The client chooses its cadence; a fast one would push lifecycle facts
    # out of the bounded journal.
    now = _pin_client_stats_clock(started_runner)
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=30.0))
    now[0] += 0.5

    with caplog.at_level(logging.DEBUG, logger="reactor_runtime.runner.runner"):
        started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=29.0))

    assert len(_client_stats_facts(started_runner)) == 1
    messages = [r.getMessage() for r in caplog.records]
    assert "client stats" not in messages
    assert "client connection stats" not in messages


async def test_client_stats_a_batch_larger_than_the_sdk_sends_is_not_journalled(
    started_runner: Runner,
) -> None:
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})

    started_runner.client_stats_received(
        ConnId(3), _one_track_batch(**{f"metric_{i}": 1.0 for i in range(65)})
    )

    assert _client_stats_facts(started_runner) == []


async def test_client_stats_limit_restarts_with_a_new_connection_of_the_same_id(
    started_runner: Runner,
) -> None:
    _pin_client_stats_clock(started_runner)
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})

    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=30.0))
    started_runner.connection_closed(ConnId(3))
    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=29.0))

    assert len(_client_stats_facts(started_runner)) == 2


async def test_client_stats_outside_a_session_leave_the_limits_alone(
    started_runner: Runner,
) -> None:
    # A batch before the session starts is dropped, and must not hold back
    # the session's first reading.
    _pin_client_stats_clock(started_runner)
    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=30.0))

    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=29.0))

    assert len(_client_stats_facts(started_runner)) == 1
    started_runner.stop_session()
    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=28.0))
    assert started_runner._client_stats._accepted_at == {}


async def test_client_stats_limits_are_cleared_when_the_session_closes(
    started_runner: Runner,
) -> None:
    # A closing session forgets its connections' last batch times, so they
    # never outlive the session.
    _pin_client_stats_clock(started_runner)
    started_runner.start_session({"session_id": _LIVE_SESSION_ID})
    started_runner.client_stats_received(ConnId(3), _one_track_batch(frames_per_second=30.0))
    assert started_runner._client_stats._accepted_at

    started_runner.stop_session()

    assert started_runner._client_stats._accepted_at == {}
