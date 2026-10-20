"""The runner in a stepped session: step requests in, their replies out on the data channel."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from reactor_runtime import (
    ApplicationError,
    CommandError,
    InputField,
    InputState,
    MediaInput,
    Output,
    ReactorApp,
    ReadMode,
    StepCompleted,
    StepOutcome,
    UploadedFile,
    Video,
    event,
    protocol,
)
from reactor_runtime.core import (
    CompletedStep,
    ConnectionCapabilities,
    ConnId,
    InputFrame,
    MediaBundle,
    MediaChunk,
    RequestedStep,
    RuntimeConfig,
    SessionEvent,
    SessionState,
    StepResultsConfig,
)
from reactor_runtime.interface.internal.input_buffer import InputBuffer
from reactor_runtime.message_gateway import InboundStep
from reactor_runtime.protocol.common import dict_to_struct
from reactor_runtime.protocol.frames import FrameEncodingError
from reactor_runtime.runner import runner as runner_module
from reactor_runtime.runner.runner import Runner
from reactor_runtime.runner.session_start import InvalidSessionStartError
from reactor_runtime.upload_store import UnknownUploadError
from reactor_wire.v1 import common_pb2, control_pb2, data_pb2, model_pb2, platform_pb2, step_pb2

V1 = protocol.ProtocolVersion.V1
DATA = protocol.Channel.DATA
CONTROL = protocol.Channel.CONTROL
SERVER = protocol.Direction.SERVER
K = common_pb2.MessageKind


class Out(Output):
    main_video: Video


class Camera(MediaInput):
    webcam: Video


class State(InputState):
    prompt: str = InputField(default="a forest", min_length=1)
    image: UploadedFile = InputField(default=None)


class StepModel(ReactorApp):
    state: State
    camera: Camera

    def __init__(self) -> None:
        super().__init__()
        self.notes: list[str] = []
        self.inputs: list[tuple[str, list[str], list[int]]] = []
        models.append(self)

    @event(name="note")
    def note(self, text: str) -> None:
        self.notes.append(text)

    @event(name="reject_note")
    def reject_note(self) -> None:
        raise CommandError("no_notes", "notes are closed")

    @event(name="crash")
    def crash(self) -> None:
        raise RuntimeError("an unplanned failure with private detail")

    async def process_input(self) -> tuple[str, list[str], list[int]]:
        if self.state.prompt == "refuse":
            raise ApplicationError("not now")
        webcam = cast(InputBuffer, self.camera.webcam)
        frames = webcam.try_read(webcam.available, mode=ReadMode.FIFO)
        return self.state.prompt, list(self.notes), [int(f.data[0, 0, 0]) for f in frames or []]

    def generate(self, input: tuple[str, list[str], list[int]]) -> Out:
        self.inputs.append(input)
        if input[0] == "explode":
            raise RuntimeError("the model ran out of memory")
        return Out(main_video=np.zeros((2, 2, 3), dtype=np.uint8))

    async def process_output(self, outcome: StepOutcome, /) -> Output | StepCompleted | None:
        if outcome.error is not None:
            return StepCompleted(error=str(outcome.error))
        output = outcome.to_output()
        if self.state.prompt == "keep files":
            return StepCompleted(output=output, files=_KEPT_FILES)
        if self.state.prompt == "fail":
            return StepCompleted(output=output, files=_KEPT_FILES, error="the step went dark")
        return output


_KEPT_FILES = {"depth.png": b"\x89PNG....", "notes.bin": b"abc"}


class OwnLoop(StepModel):
    async def run(self) -> None:
        await asyncio.sleep(60)


models: list[StepModel] = []


class V1Connection:
    """A v1 connection that records what is sent down each channel."""

    def __init__(self, cid: int) -> None:
        self.id = ConnId(cid)
        self.capabilities = ConnectionCapabilities(carries_video=True)
        self.protocol_version = V1
        self.sent: list[bytes | str] = []
        self.control: list[bytes | str] = []
        self.media: list[MediaChunk] = []

    max_message_size = 64 * 1024 * 1024
    message_chunking = True

    def send_step_result(self, payload: bytes | str) -> None:
        self.send_message(payload)

    def send_message(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def send_control(self, payload: bytes | str) -> None:
        self.control.append(payload)

    def send_media(self, chunk: MediaChunk) -> None:
        self.media.append(chunk)

    def flush_media(self, *, hold: bool = False) -> None: ...

    def set_media_rate(self, fps: float) -> None: ...

    def set_media_depth(self, depth: int) -> None: ...

    def resume_track(self, name: str) -> None: ...

    def pause_track(self, name: str) -> None: ...

    async def close(self) -> None: ...

    def replies(self) -> list[data_pb2.DataServerMessage]:
        codec = protocol.select(V1)
        replies: list[data_pb2.DataServerMessage] = []
        for frame in self.sent:
            message = codec.decode(frame, DATA, SERVER)
            assert isinstance(message, data_pb2.DataServerMessage)
            replies.append(message)
        return replies


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(StepModel)


async def _runner(
    monkeypatch: pytest.MonkeyPatch, model: type = StepModel, *, steps_root: Path | None = None
) -> Runner:
    models.clear()
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: model)
    runner = Runner(
        RuntimeConfig(
            model_ref="fake:Model",
            step_results=StepResultsConfig(enabled=steps_root is not None),
        )
    )
    if steps_root is not None:
        assert runner.step_store is not None
        runner.step_store._root = steps_root
    await runner.start()
    return runner


@pytest.fixture
async def runner(monkeypatch: pytest.MonkeyPatch) -> Any:
    runner = await _runner(monkeypatch)
    try:
        yield runner
    finally:
        await runner.stop()


@pytest.fixture
async def saving_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    runner = await _runner(monkeypatch, steps_root=tmp_path / "steps")
    try:
        yield runner
    finally:
        await runner.stop()


def _frame(value: int) -> step_pb2.Frame:
    return step_pb2.Frame(
        encoding=step_pb2.FRAME_ENCODING_RAW,
        dtype="uint8",
        shape=[2, 2, 3],
        data=np.full((2, 2, 3), value, dtype=np.uint8).tobytes(),
    )


def _request_step(
    request_id: str,
    *,
    state: dict[str, Any] | None = None,
    commands: Sequence[tuple[str, dict[str, Any]]] = (),
    media_input: Sequence[step_pb2.TrackFrames] = (),
) -> bytes:
    message = data_pb2.DataClientMessage(
        request_id=request_id,
        kind=K.MESSAGE_KIND_REQUEST,
        request_step=step_pb2.RequestStep(
            state=dict_to_struct(state or {}),
            commands=[
                model_pb2.Command(type=name, data=dict_to_struct(args)) for name, args in commands
            ],
            media_input=media_input,
        ),
    )
    return message.SerializeToString()


def _command(request_id: str, name: str, **args: Any) -> bytes:
    return data_pb2.DataClientMessage(
        request_id=request_id,
        kind=K.MESSAGE_KIND_REQUEST,
        command=model_pb2.Command(type=name, data=dict_to_struct(args)),
    ).SerializeToString()


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(400):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached within 2 s")


async def _stepped(runner: Runner, **params: Any) -> V1Connection:
    runner.start_session({"generation": "stepped", **params})
    conn = V1Connection(1002)
    runner.connection_opened(conn)
    return conn


def _send(runner: Runner, conn: V1Connection, payload: bytes) -> None:
    runner.message_received(conn.id, payload, V1, DATA)


# -- starting a stepped session ------------------------------------------------


async def test_the_descriptor_echoes_the_generation_always(runner: Runner) -> None:
    assert runner.descriptor()["generation"] == "streamed"
    runner.start_session({"generation": "stepped"})
    assert runner.descriptor()["generation"] == "stepped"
    runner.stop_session()
    await _until(lambda: runner._sm.current_state is SessionState.READY)
    runner.start_session({})
    assert runner.descriptor()["generation"] == "streamed"


async def test_an_unknown_generation_is_refused_and_the_session_stays_ready(
    runner: Runner,
) -> None:
    with pytest.raises(InvalidSessionStartError, match="generation must be one of"):
        runner.start_session({"generation": "batch"})
    assert runner._sm.current_state is SessionState.READY


async def test_a_model_with_its_own_loop_cannot_start_a_stepped_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = await _runner(monkeypatch, OwnLoop)
    try:
        with pytest.raises(InvalidSessionStartError, match="drives its own run"):
            runner.start_session({"generation": "stepped"})
        assert runner._sm.current_state is SessionState.READY
        runner.start_session({})
        assert runner._sm.current_state is SessionState.WAITING
    finally:
        await runner.stop()


# -- one step, end to end ------------------------------------------------------


async def test_a_requested_step_is_answered_with_its_result_on_the_data_channel(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    _send(
        runner,
        conn,
        _request_step(
            "r1",
            state={"prompt": "a lake"},
            media_input=[
                step_pb2.TrackFrames(
                    track="webcam",
                    kind=platform_pb2.TRACK_KIND_VIDEO,
                    frames=[_frame(7), _frame(9)],
                )
            ],
        ),
    )
    await _until(lambda: len(conn.sent) == 1)
    await asyncio.sleep(0.02)

    [result] = conn.replies()
    assert result.request_id == "r1"
    assert result.kind == K.MESSAGE_KIND_RESPONSE
    assert result.WhichOneof("payload") == "step_result"
    assert result.step_result.step == 1
    assert result.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_UNSPECIFIED
    assert not result.step_result.HasField("error")
    assert conn.control == []
    assert models[-1].inputs == [("a lake", [], [7, 9])]
    assert conn.media == []


async def test_a_command_sent_before_a_step_is_applied_before_it(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _command("c1", "note", text="first"))
    _send(runner, conn, _request_step("r1", commands=[("note", {"text": "second"})]))
    _send(runner, conn, _command("c2", "note", text="third"))
    await _until(lambda: len(conn.sent) == 3)

    assert models[-1].inputs == [("a forest", ["first", "second"], [])]
    assert models[-1].notes == ["first", "second", "third"]
    kinds = [(m.request_id, m.WhichOneof("payload")) for m in conn.replies()]
    # The model acknowledges a command from its own thread as it applies it,
    # and the step's result is sent once the runtime has journalled the step,
    # so c1's acknowledgement precedes the result. A command applied after the
    # step may be acknowledged before or after the step's result goes out.
    assert sorted(kinds, key=str) == [("c1", None), ("c2", None), ("r1", "step_result")]
    assert kinds.index(("c1", None)) < kinds.index(("r1", "step_result"))


async def test_steps_are_numbered_from_one(runner: Runner) -> None:
    conn = await _stepped(runner)
    for request_id in ("r1", "r2", "r3"):
        _send(runner, conn, _request_step(request_id))
    await _until(lambda: len(conn.sent) == 3)
    results = [m for m in conn.replies() if m.WhichOneof("payload") == "step_result"]
    assert [(m.request_id, m.step_result.step) for m in results] == [
        ("r1", 1),
        ("r2", 2),
        ("r3", 3),
    ]
    facts = [
        t.detail["step"]
        for _, e in runner.events._history  # type: ignore[attr-defined]
        if (t := e.transition).event is SessionEvent.STEP_COMPLETED
    ]
    assert facts == [1, 2, 3]


async def test_an_empty_request_runs_a_step_on_the_current_state(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    [reply] = conn.replies()
    assert (reply.request_id, reply.WhichOneof("payload")) == ("r1", "step_result")
    assert reply.step_result.step == 1
    assert models[-1].inputs == [("a forest", [], [])]


async def test_a_standalone_command_before_an_empty_request_reaches_its_step(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _command("c1", "set_prompt", prompt="a lake"))
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 2)

    assert models[-1].inputs == [("a lake", [], [])]
    kinds = [(m.request_id, m.WhichOneof("payload")) for m in conn.replies()]
    assert kinds == [("c1", None), ("r1", "step_result")]


async def test_a_state_value_carries_a_file_by_inline_upload_reference(runner: Runner) -> None:
    conn = await _stepped(runner)
    upload_id = runner.uploads.create_slot("cat.png", "image/png", 3)
    runner.uploads.put(upload_id, b"cat")
    _send(runner, conn, _request_step("r1", state={"image": {"upload_id": upload_id}}))
    await _until(lambda: len(conn.sent) == 1)

    assert conn.replies()[0].WhichOneof("payload") == "step_result"
    assert models[-1].state.image == UploadedFile(
        name="cat.png", mime_type="image/png", data=b"cat"
    )


async def test_a_state_value_whose_upload_never_arrives_refuses_the_step(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = await _stepped(runner)

    async def never_arrives(upload_id: str, wait_seconds: float = 0.0) -> UploadedFile:
        raise UnknownUploadError(upload_id)

    monkeypatch.setattr(runner._uploads, "fetch", never_arrives)
    _send(runner, conn, _request_step("r1", state={"image": {"upload_id": "missing"}}))
    await _until(lambda: len(conn.sent) == 1)

    assert _refusal(conn.replies()[0]) == "state.image references an unresolved upload"
    assert models[-1].inputs == []
    assert _journalled_steps(runner) == []


async def test_a_request_waiting_for_an_upload_holds_back_the_requests_after_it(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    upload_id = runner.uploads.create_slot("cat.png", "image/png", 3)
    _send(runner, conn, _request_step("r1", state={"image": {"upload_id": upload_id}}))
    _send(runner, conn, _request_step("r2"))
    await asyncio.sleep(0.05)
    assert conn.sent == []

    runner.uploads.put(upload_id, b"cat")
    await _until(lambda: len(conn.sent) == 2)

    assert [(m.request_id, m.step_result.step) for m in conn.replies()] == [("r1", 1), ("r2", 2)]
    assert runner._step_order == {}


async def test_a_command_waiting_for_an_upload_holds_back_a_step_sent_after_it(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    upload_id = runner.uploads.create_slot("cat.png", "image/png", 3)
    _send(runner, conn, _command("c1", "set_image", image={"upload_id": upload_id}))
    _send(runner, conn, _request_step("r1"))
    await asyncio.sleep(0.05)
    assert conn.sent == []

    runner.uploads.put(upload_id, b"cat")
    await _until(lambda: len(conn.sent) == 2)

    kinds = [(m.request_id, m.WhichOneof("payload")) for m in conn.replies()]
    assert kinds == [("c1", None), ("r1", "step_result")]
    assert runner._step_order == {}


async def test_one_connections_upload_wait_does_not_hold_back_another_connection(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    other = V1Connection(1003)
    runner.connection_opened(other)
    upload_id = runner.uploads.create_slot("cat.png", "image/png", 3)
    _send(runner, conn, _request_step("r1", state={"image": {"upload_id": upload_id}}))
    _send(runner, other, _request_step("r2"))
    await _until(lambda: len(other.sent) == 1)

    assert conn.sent == []
    assert other.replies()[0].step_result.step == 1
    runner.uploads.put(upload_id, b"cat")
    await _until(lambda: len(conn.sent) == 1)


# -- refusals --------------------------------------------------------------------


def _refusal(
    message: data_pb2.DataServerMessage,
    failure_code: step_pb2.StepFailureCode = step_pb2.STEP_FAILURE_CODE_REFUSED,
) -> str:
    assert message.kind == K.MESSAGE_KIND_RESPONSE
    assert message.WhichOneof("payload") == "step_result"
    result = message.step_result
    assert result.failure_code == failure_code
    assert (result.step, result.elapsed_seconds, list(result.tracks), list(result.files)) == (
        0,
        0.0,
        [],
        [],
    )
    return result.error


def _journalled_errors(runner: Runner) -> list[str]:
    return [
        t.detail["message"]
        for _, e in runner.events._history  # type: ignore[attr-defined]
        if (t := e.transition).event is SessionEvent.ERROR
    ]


def _journalled_steps(runner: Runner) -> list[SessionEvent]:
    return [
        t.event
        for _, e in runner.events._history  # type: ignore[attr-defined]
        if (t := e.transition).event
        in (SessionEvent.STEP_COMPLETED, SessionEvent.STEP_RESULT_READY)
    ]


async def test_a_step_with_an_invalid_command_is_refused_and_changes_nothing(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    _send(
        runner,
        conn,
        _request_step(
            "r1",
            state={"prompt": "a lake"},
            commands=[("note", {"text": 3}), ("note", {"text": "never"})],
        ),
    )
    await _until(lambda: len(conn.sent) == 1)
    await asyncio.sleep(0.02)

    [reply] = conn.replies()
    assert reply.request_id == "r1"
    assert _refusal(reply).startswith("commands[0] is refused: text")
    assert reply.step_result.entry == "commands[0]"
    model = models[-1]
    assert model.state.prompt == "a forest"
    assert model.notes == []
    assert model.inputs == []
    [error] = _journalled_errors(runner)
    assert error.startswith("step request: commands[0] is refused: text")


async def test_a_step_with_an_invalid_state_value_names_the_key(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", state={"prompt": ""}))
    await _until(lambda: len(conn.sent) == 1)
    assert _refusal(conn.replies()[0]).startswith("state.prompt is refused: prompt")


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (
            step_pb2.TrackFrames(track="mic"),
            "media_input[0] names 'mic', which is not an input track",
        ),
        (
            step_pb2.TrackFrames(track="webcam", kind=platform_pb2.TRACK_KIND_AUDIO),
            "media_input[0] says 'webcam' is not a video track",
        ),
        (
            step_pb2.TrackFrames(track="webcam", frames=[step_pb2.Frame(dtype="uint8")]),
            "media_input[0].frames[0]: frame encoding FRAME_ENCODING_UNSPECIFIED is not supported",
        ),
    ],
)
async def test_a_step_whose_frames_do_not_fit_is_refused(
    runner: Runner, entry: step_pb2.TrackFrames, reason: str
) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", media_input=[entry]))
    await _until(lambda: len(conn.sent) == 1)
    assert _refusal(conn.replies()[0]).startswith(reason)
    assert conn.replies()[0].step_result.entry == "media_input[0]"
    assert _journalled_steps(runner) == []


async def test_a_step_with_more_frames_than_its_track_holds_is_refused(runner: Runner) -> None:
    conn = await _stepped(runner)
    frames = step_pb2.TrackFrames(track="webcam", frames=[_frame(i % 256) for i in range(129)])
    _send(runner, conn, _request_step("r1", state={"prompt": "a lake"}, media_input=[frames]))
    await _until(lambda: len(conn.sent) == 1)

    reason = _refusal(conn.replies()[0])
    assert (
        reason
        == "media_input[0] holds 129 frames for 'webcam', whose input buffer holds 128 frames"
    )
    assert models[-1].state.prompt == "a forest"
    assert models[-1].inputs == []


async def test_a_steps_frames_are_decoded_on_the_models_thread(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []
    decode = runner_module.decode_frame

    def recording_decode(frame: step_pb2.Frame, kind: Any) -> Any:
        threads.append(threading.get_ident())
        return decode(frame, kind)

    monkeypatch.setattr(runner_module, "decode_frame", recording_decode)
    conn = await _stepped(runner)
    frames = step_pb2.TrackFrames(track="webcam", frames=[_frame(7)])
    _send(runner, conn, _request_step("r1", media_input=[frames]))
    await _until(lambda: len(conn.sent) == 1)

    assert models[-1].inputs == [("a forest", [], [7])]
    assert threads
    assert threading.get_ident() not in threads


async def test_a_stepped_session_drops_live_track_frames(runner: Runner) -> None:
    conn = await _stepped(runner)
    runner.media_received(conn.id, "webcam", InputFrame(data=np.full((2, 2, 3), 9, np.uint8)))
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    assert models[-1].inputs == [("a forest", [], [])]


class _RecordingStore:
    """A step store that records the step numbers it is offered and saves none."""

    def __init__(self) -> None:
        self.admitted: list[int] = []

    def admit(self, recording_id: str, step: int, *args: Any) -> bool:
        self.admitted.append(step)
        return False

    def close(self) -> None: ...


async def test_a_step_the_model_refuses_is_answered_and_not_counted(runner: Runner) -> None:
    store = _RecordingStore()
    runner._step_store = cast(Any, store)
    conn = await _stepped(runner, steps=1)
    _send(runner, conn, _request_step("r1", state={"prompt": "refuse"}))
    await _until(lambda: len(conn.sent) == 1)
    assert _refusal(conn.replies()[0]) == "not now"
    assert runner._sm.current_state in (SessionState.STREAMING, SessionState.WAITING)
    assert models[-1].state.prompt == "refuse"
    assert _journalled_steps(runner) == []
    assert store.admitted == []

    _send(runner, conn, _request_step("r2", state={"prompt": "a lake"}))
    await _until(lambda: runner._sm.current_state is not SessionState.STREAMING)
    results = [m.step_result for m in conn.replies()]
    assert [(r.step, r.failure_code) for r in results] == [
        (0, step_pb2.STEP_FAILURE_CODE_REFUSED),
        (1, step_pb2.STEP_FAILURE_CODE_UNSPECIFIED),
    ]
    assert _journalled_steps(runner) == [SessionEvent.STEP_COMPLETED]
    assert store.admitted == [1]


async def test_a_failing_command_handler_answers_its_step_and_the_step_does_not_run(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    _send(
        runner,
        conn,
        _request_step(
            "r1",
            commands=[
                ("note", {"text": "kept"}),
                ("reject_note", {}),
                ("note", {"text": "skipped"}),
            ],
        ),
    )
    await _until(lambda: len(conn.sent) == 1)
    await asyncio.sleep(0.02)

    [reply] = conn.replies()
    assert reply.request_id == "r1"
    error = _refusal(reply, step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED)
    assert error == "reject_note failed (no_notes): notes are closed"
    assert (reply.step_result.code, reply.step_result.entry) == ("no_notes", "commands[1]")
    assert models[-1].notes == ["kept"]
    assert models[-1].inputs == []
    assert _journalled_steps(runner) == []


async def test_an_unexpected_handler_error_answers_its_step_without_its_detail(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", commands=[("crash", {})]))
    await _until(lambda: len(conn.sent) == 1)

    error = _refusal(conn.replies()[0], step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED)
    assert error.startswith("crash failed (internal_error):")
    assert "private detail" not in error
    assert models[-1].inputs == []


async def test_a_refused_request_after_a_failed_handler_leaves_nothing_behind(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    for index in range(5):
        _send(
            runner,
            conn,
            _request_step(f"r{index}", commands=[("reject_note", {}), ("note", {"text": 3})]),
        )
    _send(runner, conn, _request_step("last"))
    await _until(lambda: len(conn.sent) == 6)

    *refused, last = conn.replies()
    assert [_refusal(m).split(":")[0] for m in refused] == ["commands[1] is refused"] * 5
    assert last.step_result.step == 1
    assert models[-1]._failed_step_calls == {}


async def test_a_command_failure_answers_only_its_own_step(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", commands=[("reject_note", {})]))
    _send(runner, conn, _request_step("r2", commands=[("note", {"text": "fine"})]))
    await _until(lambda: len(conn.sent) == 2)
    await asyncio.sleep(0.02)

    first, second = conn.replies()
    assert (first.request_id, first.step_result.failure_code) == (
        "r1",
        step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED,
    )
    assert (second.request_id, second.step_result.failure_code, second.step_result.step) == (
        "r2",
        step_pb2.STEP_FAILURE_CODE_UNSPECIFIED,
        1,
    )
    assert models[-1].inputs == [("a forest", ["fine"], [])]


async def test_a_step_the_model_reports_as_failed_is_answered_failed_with_its_number(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", state={"prompt": "explode"}))
    await _until(lambda: len(conn.sent) == 1)

    [reply] = conn.replies()
    result = reply.step_result
    assert result.failure_code == step_pb2.STEP_FAILURE_CODE_FAILED
    assert result.step == 1
    assert result.error == "the model ran out of memory"


async def test_a_step_report_from_an_earlier_session_answers_session_ended(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    report = CompletedStep(
        bundle=None,
        files={},
        error=None,
        elapsed=0.1,
        session=runner._sessions_posted - 1,
        fps=24.0,
        request=RequestedStep(conn.id, "r-old", 1),
    )
    runner._record_step(report)

    error = _refusal(conn.replies()[0], step_pb2.STEP_FAILURE_CODE_SESSION_ENDED)
    assert error == "the session ended before the step's result was sent"
    assert _journalled_steps(runner) == []


async def test_a_stepped_session_closes_after_its_steps(runner: Runner) -> None:
    conn = await _stepped(runner, steps=2)
    _send(runner, conn, _request_step("r1"))
    _send(runner, conn, _request_step("r2"))
    await _until(lambda: runner._sm.current_state is SessionState.READY)
    results = [m for m in conn.replies() if m.WhichOneof("payload") == "step_result"]
    assert [m.step_result.step for m in results] == [1, 2]


async def test_requests_queued_beyond_the_sessions_steps_are_not_generated(
    runner: Runner,
) -> None:
    conn = await _stepped(runner, steps=1)
    for index in range(10):
        _send(runner, conn, _request_step(f"r{index}"))
    await _until(lambda: len(conn.sent) == 10)
    await _until(lambda: runner._sm.current_state is SessionState.READY)

    # Replies to different requests are correlated by request id, not order:
    # a refusal leaves from the model's thread, ahead of a result still on its
    # way through the runtime loop.
    replies = {m.request_id: m.step_result for m in conn.replies()}
    first = replies.pop("r0")
    rest = list(replies.values())
    assert (first.step, first.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_UNSPECIFIED)
    assert {m.failure_code for m in rest} <= {
        step_pb2.STEP_FAILURE_CODE_REFUSED,
        step_pb2.STEP_FAILURE_CODE_NOT_READY,
    }
    assert all(m.step == 0 for m in rest)
    assert len(models[-1].inputs) == 1
    assert runner._steps_completed == 1


async def test_requests_queued_beyond_the_sessions_steps_apply_none_of_their_calls(
    runner: Runner,
) -> None:
    conn = await _stepped(runner, steps=1)
    for index in range(10):
        _send(runner, conn, _request_step(f"r{index}", commands=[("note", {"text": f"n{index}"})]))
    # The session closes its client once its step is reported, so the replies
    # to the requests still in flight have no connection to go to.
    await _until(lambda: runner._sm.current_state is SessionState.READY)
    await asyncio.sleep(0.05)

    assert models[-1].notes == ["n0"]
    assert len(models[-1].inputs) == 1


async def test_a_stepped_session_with_steps_waits_for_a_client(runner: Runner) -> None:
    runner.start_session({"generation": "stepped", "steps": 2})
    await asyncio.sleep(0.02)

    assert runner._sm.current_state is SessionState.WAITING


async def test_a_stepped_session_with_steps_waits_again_after_its_client_leaves(
    runner: Runner,
) -> None:
    conn = await _stepped(runner, steps=2)
    await _until(lambda: runner._sm.current_state is SessionState.STREAMING)
    runner.connection_closed(conn.id)
    await asyncio.sleep(0.02)

    assert runner._sm.current_state is not SessionState.STREAMING


async def test_a_step_request_in_a_streamed_session_is_refused(runner: Runner) -> None:
    runner.start_session({})
    conn = V1Connection(1002)
    runner.connection_opened(conn)
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)
    assert "not stepped" in _refusal(conn.replies()[0])


async def test_a_step_request_after_the_session_stops_is_not_ready(runner: Runner) -> None:
    conn = await _stepped(runner)
    runner.stop_session()
    await runner._submit_step(
        InboundStep(
            state={},
            commands=(),
            media_input=(),
            conn_id=conn.id,
            request_id="r1",
            received_at=0.0,
        )
    )

    [reply] = conn.replies()
    assert reply.request_id == "r1"
    assert _refusal(reply, step_pb2.STEP_FAILURE_CODE_NOT_READY) == (
        "the session is not running a step yet"
    )
    assert models[-1].inputs == []
    assert _journalled_steps(runner) == []
    assert _journalled_errors(runner) == []


async def test_a_step_request_after_a_streamed_session_stops_is_not_ready(
    runner: Runner,
) -> None:
    runner.start_session({})
    conn = V1Connection(1002)
    runner.connection_opened(conn)
    runner.stop_session()
    await runner._submit_step(
        InboundStep(
            state={},
            commands=(),
            media_input=(),
            conn_id=conn.id,
            request_id="r1",
            received_at=0.0,
        )
    )

    [reply] = conn.replies()
    assert _refusal(reply, step_pb2.STEP_FAILURE_CODE_NOT_READY) == (
        "the session is not running a step yet"
    )


# -- what a stepped session does not do ------------------------------------------


async def test_a_stepped_session_streams_no_media(runner: Runner) -> None:
    conn = await _stepped(runner)
    runner._emit_media(MediaChunk(bundle=MediaBundle(), fps=30.0))
    assert conn.media == []


async def test_a_stepped_session_refuses_clips(runner: Runner) -> None:
    conn = await _stepped(runner)
    runner.clip_requested(conn.id, 5.0, "c1")
    runner.recording_requested(conn.id, "c2")
    codec = protocol.select(V1)
    reasons = []
    for frame in conn.control:
        reply = codec.decode(frame, CONTROL, SERVER)
        assert isinstance(reply, control_pb2.ControlServerMessage)
        reasons.append(reply.clip_failed.reason)
    assert reasons == ["a stepped session is not recorded, so it has no clips"] * 2


# -- the step result on the wire -------------------------------------------------


async def test_the_step_result_carries_the_steps_media_per_output_track(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    result = conn.replies()[0].step_result
    [track] = result.tracks
    assert (track.track, track.kind) == ("main_video", platform_pb2.TRACK_KIND_VIDEO)
    [frame] = track.frames
    assert (frame.encoding, frame.dtype, list(frame.shape)) == (
        step_pb2.FRAME_ENCODING_RAW,
        "uint8",
        [2, 2, 3],
    )
    assert frame.data == bytes(12)
    assert track.rate > 0
    assert result.elapsed_seconds > 0
    assert not result.HasField("error")
    assert list(result.files) == []


async def test_the_step_result_references_the_files_a_saved_step_keeps(
    saving_runner: Runner,
) -> None:
    conn = await _stepped(saving_runner)
    _send(saving_runner, conn, _request_step("r1", state={"prompt": "keep files"}))
    await _until(lambda: len(conn.sent) == 1)

    [reply] = conn.replies()
    result = reply.step_result
    assert [(f.name, f.content_type, f.size) for f in result.files] == [
        ("depth.png", "image/png", len(_KEPT_FILES["depth.png"])),
        ("notes.bin", "application/octet-stream", 3),
    ]
    assert not result.HasField("error")


async def test_the_step_result_references_no_files_when_step_results_are_off(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", state={"prompt": "keep files"}))
    await _until(lambda: len(conn.sent) == 1)

    [reply] = conn.replies()
    assert reply.WhichOneof("payload") == "step_result"
    assert list(reply.step_result.files) == []


async def test_a_failed_step_is_answered_with_its_error_and_no_files(
    saving_runner: Runner,
) -> None:
    conn = await _stepped(saving_runner)
    _send(saving_runner, conn, _request_step("r1", state={"prompt": "fail"}))
    await _until(lambda: len(conn.sent) == 1)
    await asyncio.sleep(0.02)

    [reply] = conn.replies()
    assert reply.kind == K.MESSAGE_KIND_RESPONSE
    result = reply.step_result
    assert (result.step, result.failure_code, result.error) == (
        1,
        step_pb2.STEP_FAILURE_CODE_FAILED,
        "the step went dark",
    )
    assert list(result.files) == []
    assert _journalled_steps(saving_runner)[0] is SessionEvent.STEP_COMPLETED


async def test_a_result_too_large_for_the_connection_is_answered_without_media_and_counted(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "_STEP_MIN_MESSAGE_SIZE", 0)
    conn = await _stepped(runner)
    conn.max_message_size = 16
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    [reply] = conn.replies()
    assert (reply.request_id, reply.kind) == ("r1", K.MESSAGE_KIND_RESPONSE)
    result = reply.step_result
    assert (result.step, result.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE)
    assert result.elapsed_seconds > 0
    assert re.fullmatch(
        r"step 1's result is (at least )?\d+ bytes; this connection takes at most 16", result.error
    )
    assert (list(result.tracks), list(result.files)) == ([], [])
    assert _journalled_steps(runner) == [SessionEvent.STEP_COMPLETED]

    conn.max_message_size = 64 * 1024 * 1024
    _send(runner, conn, _request_step("r2"))
    await _until(lambda: len(conn.sent) == 2)
    assert conn.replies()[1].step_result.step == 2
    assert runner._sm.current_state is SessionState.STREAMING


async def test_media_that_outgrows_the_connection_is_answered_without_encoding_it(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_encode(*_: object, **__: object) -> list[step_pb2.TrackFrames]:
        raise AssertionError("a result whose frames cannot fit was encoded")

    monkeypatch.setattr(runner_module, "track_frames", must_not_encode)
    monkeypatch.setattr(runner_module, "_STEP_MIN_MESSAGE_SIZE", 0)
    conn = await _stepped(runner)
    # The model's 2x2 RGB frame is 12 bytes of media alone.
    conn.max_message_size = 8
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    result = conn.replies()[0].step_result
    assert (result.step, result.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE)
    assert re.fullmatch(
        r"step 1's result is at least \d+ bytes; this connection takes at most 8", result.error
    )


async def test_a_step_result_s_frames_are_encoded_off_the_runtime_loop(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    encoded_on: list[int] = []
    encode = runner_module.track_frames

    def recording(*args: Any, **kwargs: Any) -> list[step_pb2.TrackFrames]:
        encoded_on.append(threading.get_ident())
        return encode(*args, **kwargs)

    monkeypatch.setattr(runner_module, "track_frames", recording)
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    assert conn.replies()[0].step_result.step == 1
    assert encoded_on
    assert loop_thread not in encoded_on


async def test_a_connection_whose_messages_are_too_small_is_refused_a_step(runner: Runner) -> None:
    conn = await _stepped(runner)
    conn.max_message_size = 16
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    error = _refusal(conn.replies()[0])
    assert error == (
        "this connection's data-channel chunking takes at most 16 bytes a message; "
        "a stepped session needs at least 262144"
    )


async def test_a_connection_without_chunking_is_refused_a_step(runner: Runner) -> None:
    conn = await _stepped(runner)
    conn.message_chunking = False
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)
    await asyncio.sleep(0.02)

    assert "negotiated data-channel chunking" in _refusal(conn.replies()[0])
    assert models[-1].inputs == []
    assert _journalled_steps(runner) == []


async def test_a_step_whose_output_cannot_be_encoded_is_answered_failed(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unencodable(bundle: Any, **_: Any) -> list[step_pb2.TrackFrames]:
        raise FrameEncodingError("float64 video is not supported")

    monkeypatch.setattr("reactor_runtime.runner.runner.track_frames", unencodable)
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    result = conn.replies()[0].step_result
    assert (result.step, result.failure_code, result.error) == (
        1,
        step_pb2.STEP_FAILURE_CODE_FAILED,
        "step 1's output could not be encoded: float64 video is not supported",
    )
    assert result.elapsed_seconds > 0
    assert (list(result.tracks), list(result.files)) == ([], [])
    assert _journalled_steps(runner) == [SessionEvent.STEP_COMPLETED]
