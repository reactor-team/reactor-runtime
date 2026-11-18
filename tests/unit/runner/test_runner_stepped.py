"""The runner in a stepped session: step requests in, their replies out on the data channel."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import reactor_webrtc as rw

from reactor_runtime import (
    ApplicationError,
    Audio,
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
    DecodedStep,
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
from reactor_runtime.interface.tracks.output import OUTPUT_REGISTRY
from reactor_runtime.message_gateway import InboundStep
from reactor_runtime.protocol.common import dict_to_struct
from reactor_runtime.protocol.frames import FrameEncodingError
from reactor_runtime.runner import stepped as stepped_module
from reactor_runtime.runner.runner import Runner
from reactor_runtime.runner.session_start import InvalidSessionStartError
from reactor_runtime.transport import step_frames as step_frames_module
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
        self.frames: list[InputFrame] = []
        models.append(self)

    @event(name="note")
    def note(self, text: str) -> None:
        self.notes.append(text)

    @event(name="reject_note")
    def reject_note(self) -> None:
        raise CommandError("no_notes", "notes are closed")

    @event(name="crash")
    def unplanned_failure(self) -> None:
        raise RuntimeError("an unplanned failure with private detail")

    async def process_input(self) -> tuple[str, list[str], list[int]]:
        if self.state.prompt == "refuse":
            raise ApplicationError("not now")
        webcam = cast(InputBuffer, self.camera.webcam)
        frames = webcam.try_read(webcam.available, mode=ReadMode.FIFO)
        self.frames += frames or []
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
        if self.state.prompt == "data":
            return StepCompleted(output=output, files=_KEPT_FILES, data=_DATA)
        if self.state.prompt == "data dict":
            return StepCompleted(output=output, data={"actions": [[0.25, -1]], "done": False})
        if self.state.prompt == "large data":
            return StepCompleted(output=output, data=bytes(4096))
        if self.state.prompt == "fail with data":
            return StepCompleted(output=output, data=_DATA, error="the step went dark")
        return output


_KEPT_FILES = {"depth.png": b"\x89PNG....", "notes.bin": b"abc"}
# Msgpack for {"actions": [1, 2]}, as a policy model sends its action chunk.
_DATA = b"\x81\xa7actions\x92\x01\x02"


class VideoAndAudio(Output):
    main_video: Video
    main_audio: Audio


class SilentStepModel(StepModel):
    """Returns video and no audio, an empty ``(1, 0)`` array, as examples/echo does."""

    # It declares an audio output the base model does not have.
    def generate(  # type: ignore[ty:invalid-method-override]
        self, input: tuple[str, list[str], list[int]]
    ) -> VideoAndAudio:
        self.inputs.append(input)
        return VideoAndAudio(
            main_video=np.zeros((2, 2, 3), dtype=np.uint8),
            main_audio=np.zeros((1, 0), dtype=np.int16),
        )


class DataOnlyModel(ReactorApp):
    """Declares no output track and returns its step's actions as data alone."""

    state: State

    async def process_input(self) -> str:
        return self.state.prompt

    def generate(self, input: str) -> dict[str, Any]:
        return {"prompt": input, "actions": [1, 2]}

    async def process_output(self, outcome: StepOutcome, /) -> StepCompleted:
        return StepCompleted(data=outcome.result)


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

    async def send_step_result(self, payload: bytes) -> bool:
        self.send_message(payload)
        return True

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


async def test_the_descriptor_reports_a_stepped_session_as_not_recorded(runner: Runner) -> None:
    # Recording is on by default, but a stepped session is never recorded, so
    # a consumer that reads the descriptor at start must not expect a recording.
    assert runner.descriptor()["recording"]["enabled"] is True
    runner.start_session({"generation": "stepped"})
    assert runner.descriptor()["recording"]["enabled"] is False
    runner.stop_session()
    await _until(lambda: runner._sm.current_state is SessionState.READY)
    runner.start_session({})
    assert runner.descriptor()["recording"]["enabled"] is True


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


async def test_a_session_started_as_soon_as_the_runtime_is_ready_runs_its_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No settle between start() and the session: the model thread may not have
    # run yet, and the session must still reach the model.
    runner = await _runner(monkeypatch)
    try:
        conn = await _stepped(runner)
        _send(runner, conn, _request_step("r1"))
        await _until(lambda: len(conn.sent) == 1)

        result = conn.replies()[0].step_result
        assert (result.step, result.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_UNSPECIFIED)
    finally:
        await runner.stop()


async def test_a_step_whose_audio_is_empty_keeps_its_video_and_does_not_fail(
    monkeypatch: pytest.MonkeyPatch, register_model: Callable[[type], None]
) -> None:
    register_model(SilentStepModel)
    runner = await _runner(monkeypatch, SilentStepModel)
    try:
        conn = await _stepped(runner)
        _send(runner, conn, _request_step("r1"))
        await _until(lambda: len(conn.sent) == 1)

        result = conn.replies()[0].step_result
        assert (result.step, result.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_UNSPECIFIED)
        assert not result.HasField("error")
        assert [(t.track, len(t.frames)) for t in result.tracks] == [
            ("main_video", 1),
            ("main_audio", 0),
        ]
    finally:
        await runner.stop()


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
    assert len(runner._steps.order) == 0


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
    assert len(runner._steps.order) == 0


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
    decode = stepped_module.decode_frame

    def recording_decode(frame: step_pb2.Frame, kind: Any) -> Any:
        threads.append(threading.get_ident())
        return decode(frame, kind)

    monkeypatch.setattr(stepped_module, "decode_frame", recording_decode)
    conn = await _stepped(runner)
    frames = step_pb2.TrackFrames(track="webcam", frames=[_frame(7)])
    _send(runner, conn, _request_step("r1", media_input=[frames]))
    await _until(lambda: len(conn.sent) == 1)

    assert models[-1].inputs == [("a forest", [], [7])]
    assert threads
    assert threading.get_ident() not in threads


def _h264_frame(**fields: Any) -> step_pb2.Frame:
    return step_pb2.Frame(
        encoding=step_pb2.FRAME_ENCODING_H264,
        dtype="uint8",
        shape=[2, 2, 3],
        data=b"an access unit",
        **fields,
    )


async def test_a_decoded_step_reads_each_encoded_frame_as_its_picture(runner: Runner) -> None:
    conn = await _stepped(runner)
    frames = step_pb2.TrackFrames(
        track="webcam", frames=[_h264_frame(metadata=b"tag", capture_time_us=42), _frame(5)]
    )
    picture = np.full((2, 2, 3), 77, dtype=np.uint8)
    runner.decoded_step_received(
        conn.id,
        DecodedStep(payload=_request_step("r1", media_input=[frames]), pictures={(0, 0): picture}),
        V1,
    )
    await _until(lambda: len(conn.sent) == 1)

    assert conn.replies()[0].step_result.step == 1
    assert models[-1].inputs == [("a forest", [], [77, 5])]
    decoded, raw = models[-1].frames
    # The picture itself, not a copy, with what its encoded frame carried.
    assert decoded.data is picture
    assert (decoded.metadata, decoded.capture_time_us) == (b"tag", 42)
    assert (raw.metadata, raw.capture_time_us) == (None, None)


async def test_an_encoded_frame_for_an_audio_track_is_refused(
    monkeypatch: pytest.MonkeyPatch, register_model: Callable[[type], None]
) -> None:
    class Microphone(MediaInput):
        mic: Audio

    class ListeningStepModel(StepModel):
        microphone: Microphone

    register_model(ListeningStepModel)
    runner = await _runner(monkeypatch, ListeningStepModel)
    try:
        conn = await _stepped(runner)
        frames = step_pb2.TrackFrames(track="mic", frames=[_h264_frame()])
        runner.decoded_step_received(
            conn.id,
            DecodedStep(
                payload=_request_step("r1", media_input=[frames]),
                pictures={(0, 0): np.zeros((2, 2, 3), dtype=np.uint8)},
            ),
            V1,
        )
        await _until(lambda: len(conn.sent) == 1)

        assert _refusal(conn.replies()[0]) == (
            "media_input[0].frames[0]: an encoded video frame cannot go to audio track 'mic'"
        )
        assert conn.replies()[0].step_result.entry == "media_input[0]"
        assert models[-1].inputs == []
    finally:
        await runner.stop()


async def test_a_decoded_step_keeps_its_place_behind_a_command_sent_before_it(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _command("c1", "note", text="first"))
    frames = step_pb2.TrackFrames(track="webcam", frames=[_h264_frame()])
    runner.decoded_step_received(
        conn.id,
        DecodedStep(
            payload=_request_step("r1", media_input=[frames]),
            pictures={(0, 0): np.full((2, 2, 3), 3, dtype=np.uint8)},
        ),
        V1,
    )
    await _until(lambda: any(r.WhichOneof("payload") == "step_result" for r in conn.replies()))

    assert models[-1].inputs == [("a forest", ["first"], [3])]


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
    # The session closes its client once its one step is reported, so a
    # refusal still on its way when it does has no connection to go to: only
    # the first request is sure of an answer.
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
    await runner._steps.submit(
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
    await runner._steps.submit(
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


async def test_the_data_a_step_returns_arrives_as_it_is_with_its_frames(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", state={"prompt": "data"}))
    await _until(lambda: len(conn.sent) == 1)

    result = conn.replies()[0].step_result
    assert (result.step, result.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_UNSPECIFIED)
    assert result.data == _DATA
    assert [t.track for t in result.tracks] == ["main_video"]


async def test_a_dict_a_step_returns_arrives_as_json(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", state={"prompt": "data dict"}))
    await _until(lambda: len(conn.sent) == 1)

    result = conn.replies()[0].step_result
    assert json.loads(result.data) == {"actions": [[0.25, -1]], "done": False}


async def test_a_step_that_returns_no_data_leaves_the_field_unset(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    assert not conn.replies()[0].step_result.HasField("data")


async def test_a_failed_step_carries_no_data(runner: Runner) -> None:
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1", state={"prompt": "fail with data"}))
    await _until(lambda: len(conn.sent) == 1)

    result = conn.replies()[0].step_result
    assert (result.failure_code, result.error) == (
        step_pb2.STEP_FAILURE_CODE_FAILED,
        "the step went dark",
    )
    assert not result.HasField("data")


async def test_a_model_without_output_tracks_returns_its_data_with_no_frames(
    monkeypatch: pytest.MonkeyPatch, register_model: Callable[[type], None]
) -> None:
    register_model(DataOnlyModel)
    OUTPUT_REGISTRY.clear()
    runner = await _runner(monkeypatch, DataOnlyModel)
    try:
        conn = await _stepped(runner)
        _send(runner, conn, _request_step("r1", state={"prompt": "reach"}))
        await _until(lambda: len(conn.sent) == 1)

        result = conn.replies()[0].step_result
        assert (result.step, result.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_UNSPECIFIED)
        assert json.loads(result.data) == {"prompt": "reach", "actions": [1, 2]}
        assert list(result.tracks) == []
        assert conn.media == []
    finally:
        await runner.stop()


async def test_data_that_does_not_fit_the_connection_is_answered_result_too_large(
    runner: Runner,
) -> None:
    conn = await _stepped(runner)
    # The model's 2x2 RGB frame fits; its 4 KiB of data does not.
    conn.max_message_size = 1024
    _send(runner, conn, _request_step("r1", state={"prompt": "large data"}))
    await _until(lambda: len(conn.sent) == 1)

    result = conn.replies()[0].step_result
    assert (result.step, result.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE)
    assert re.fullmatch(
        r"step 1's result is at least \d+ bytes, more than this connection takes in one message",
        result.error,
    )
    assert not result.HasField("data")
    assert list(result.tracks) == []


async def test_a_saved_step_keeps_its_files_and_not_its_data(saving_runner: Runner) -> None:
    conn = await _stepped(saving_runner)
    _send(saving_runner, conn, _request_step("r1", state={"prompt": "data"}))
    await _until(lambda: len(conn.sent) == 1)
    assert conn.replies()[0].step_result.data == _DATA

    store = saving_runner.step_store
    assert store is not None
    assert store.root is not None
    root = store.root
    await _until(lambda: any(root.glob("*/1/result.json")))
    [step_dir] = [path.parent for path in root.glob("*/1/result.json")]
    saved = json.loads((step_dir / "result.json").read_text())
    assert "data" not in saved
    assert sorted(path.name for path in step_dir.iterdir()) == sorted(
        ["result.json", "output.mp4", *_KEPT_FILES]
    )


async def test_a_result_too_large_for_the_connection_is_answered_without_media_and_counted(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        r"step 1's result is (at least )?\d+ bytes, more than this connection takes in one message",
        result.error,
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

    monkeypatch.setattr(stepped_module, "track_frames", must_not_encode)
    conn = await _stepped(runner)
    # The model's 2x2 RGB frame is 12 bytes of media alone.
    conn.max_message_size = 8
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    result = conn.replies()[0].step_result
    assert (result.step, result.failure_code) == (1, step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE)
    assert re.fullmatch(
        r"step 1's result is at least \d+ bytes, more than this connection takes in one message",
        result.error,
    )


async def test_a_step_result_s_frames_are_encoded_off_the_runtime_loop(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    encoded_on: list[int] = []
    encode = stepped_module.track_frames

    def recording(*args: Any, **kwargs: Any) -> list[step_pb2.TrackFrames]:
        encoded_on.append(threading.get_ident())
        return encode(*args, **kwargs)

    monkeypatch.setattr(stepped_module, "track_frames", recording)
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    assert conn.replies()[0].step_result.step == 1
    assert encoded_on
    assert loop_thread not in encoded_on


async def test_a_step_result_with_media_is_serialized_off_the_runtime_loop(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_thread = threading.get_ident()
    serialized_on: list[int] = []
    codec = runner._steps._codec
    encode = codec.encode_step_result

    def recording(request_id: str, result: step_pb2.StepResult) -> Any:
        if result.tracks:
            serialized_on.append(threading.get_ident())
        return encode(request_id, result)

    monkeypatch.setattr(codec, "encode_step_result", recording)
    conn = await _stepped(runner)
    _send(runner, conn, _request_step("r1"))
    await _until(lambda: len(conn.sent) == 1)

    assert list(conn.replies()[0].step_result.tracks)
    assert serialized_on
    assert loop_thread not in serialized_on


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

    monkeypatch.setattr("reactor_runtime.runner.stepped.track_frames", unencodable)
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


# -- encoded output frames -----------------------------------------------------


def _output_codec() -> tuple[rw.VideoCodec, int]:
    """The first codec, H.264 first, this platform both encodes and decodes."""
    capabilities = rw.VideoCodecs().capabilities()
    for codec, encoding in (
        (rw.VideoCodec.H264, step_pb2.FRAME_ENCODING_H264),
        (rw.VideoCodec.Vp8, step_pb2.FRAME_ENCODING_VP8),
        (rw.VideoCodec.Vp9, step_pb2.FRAME_ENCODING_VP9),
        (rw.VideoCodec.Av1, step_pb2.FRAME_ENCODING_AV1),
    ):
        if capabilities.can_encode(codec) and capabilities.can_decode(codec):
            return codec, encoding
    pytest.skip("this platform has no video codec to encode step frames in")


def _picture(step: int, height: int, width: int) -> np.ndarray:
    """A gradient that moves a little each step, as a model's output does."""
    rows = np.arange(height, dtype=np.int32)[:, None]
    cols = np.arange(width, dtype=np.int32)[None, :]
    picture = np.empty((height, width, 3), dtype=np.uint8)
    picture[..., 0] = (rows + 2 * step) % 256
    picture[..., 1] = (cols + 3 * step) % 256
    picture[..., 2] = (rows + cols) % 256
    return picture


class PictureModel(StepModel):
    """Returns a moving picture of :attr:`size` each step, or no frames when told to."""

    size = (64, 96)

    def generate(self, input: tuple[str, list[str], list[int]]) -> Out:
        self.inputs.append(input)
        return Out(main_video=_picture(len(self.inputs), *self.size))

    async def process_output(self, outcome: StepOutcome, /) -> Output | StepCompleted | None:
        if self.state.prompt == "no frames":
            return StepCompleted()
        return await super().process_output(outcome)


class EncodingConnection(V1Connection):
    """A v1 connection that negotiated *encoding* for the model's output video."""

    def __init__(self, cid: int, encoding: int) -> None:
        super().__init__(cid)
        self.step_frame_encodings = {"main_video": encoding, "webcam": step_pb2.FRAME_ENCODING_RAW}
        self.refuse_sends = 0
        # Set to hold a refused send's outcome until it is set.
        self.refusal_gate: asyncio.Event | None = None

    async def send_step_result(self, payload: bytes) -> bool:
        if self.refuse_sends:
            self.refuse_sends -= 1
            if self.refusal_gate is not None:
                await self.refusal_gate.wait()
            return False
        return await super().send_step_result(payload)


@pytest.fixture
async def picture_runner(
    monkeypatch: pytest.MonkeyPatch, register_model: Callable[[type], None]
) -> Any:
    register_model(PictureModel)
    runner = await _runner(monkeypatch, PictureModel)
    try:
        yield runner
    finally:
        await runner.stop()


async def _encoding(runner: Runner, encoding: int, cid: int = 1002) -> EncodingConnection:
    if runner._sm.current_state is SessionState.READY:
        runner.start_session({"generation": "stepped"})
    conn = EncodingConnection(cid, encoding)
    runner.connection_opened(conn)
    return conn


async def _step(
    runner: Runner, conn: V1Connection, request_id: str, **fields: Any
) -> step_pb2.StepResult:
    """Request one step on *conn* and return its result."""
    sent = len(conn.sent)
    _queue_step(runner, conn, request_id, **fields)
    await _until(lambda: len(conn.sent) > sent)
    return conn.replies()[sent].step_result


def _queue_step(runner: Runner, conn: V1Connection, request_id: str, **fields: Any) -> None:
    """Request one step on *conn* without waiting for its result."""
    message = data_pb2.DataClientMessage(
        request_id=request_id,
        kind=K.MESSAGE_KIND_REQUEST,
        request_step=step_pb2.RequestStep(state=dict_to_struct(fields.pop("state", {})), **fields),
    )
    _send(runner, conn, message.SerializeToString())


def _result_of(conn: V1Connection, request_id: str) -> step_pb2.StepResult:
    [reply] = [reply for reply in conn.replies() if reply.request_id == request_id]
    return reply.step_result


def _video(result: step_pb2.StepResult) -> list[step_pb2.Frame]:
    [entry] = [entry for entry in result.tracks if entry.track == "main_video"]
    return list(entry.frames)


def _keyframes(result: step_pb2.StepResult) -> list[bool]:
    return [frame.keyframe for frame in _video(result)]


async def test_an_encoded_output_track_is_one_stream_across_steps(picture_runner: Runner) -> None:
    codec, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    first = await _step(picture_runner, conn, "r1")
    second = await _step(picture_runner, conn, "r2")

    assert (_keyframes(first), _keyframes(second)) == ([True], [False])
    decoder = rw.VideoCodecs().decoder(codec)
    for result in (first, second):
        [frame] = _video(result)
        assert (frame.encoding, frame.dtype, list(frame.shape)) == (encoding, "uint8", [64, 96, 3])
        picture = decoder.decode(frame.data)
        assert picture is not None
        assert picture.shape == (64, 96, 3)
    expected = _picture(2, 64, 96).astype(np.int16)
    assert np.abs(picture.astype(np.int16) - expected).mean() < 8


async def test_a_new_connection_starts_its_stream_at_a_keyframe(picture_runner: Runner) -> None:
    _, encoding = _output_codec()
    first = await _encoding(picture_runner, encoding)
    await _step(picture_runner, first, "r1")
    await _step(picture_runner, first, "r2")

    # Another client of the same session, an observer, gets its own stream.
    observer = await _encoding(picture_runner, encoding, cid=1003)
    assert _keyframes(await _step(picture_runner, observer, "o1")) == [True]
    assert _keyframes(await _step(picture_runner, first, "r3")) == [False]
    assert _keyframes(await _step(picture_runner, observer, "o2")) == [False]


async def test_a_reconnect_starts_the_stream_over_at_a_keyframe(picture_runner: Runner) -> None:
    _, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    await _step(picture_runner, conn, "r1")

    again = await _encoding(picture_runner, encoding)
    assert _keyframes(await _step(picture_runner, again, "r2")) == [True]


async def test_a_result_too_large_restarts_the_stream(
    picture_runner: Runner,
) -> None:
    _, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    await _step(picture_runner, conn, "r1")

    # The raw frame would be 18 KiB; only the encoded one is measured.
    conn.max_message_size = 64
    too_large = await _step(picture_runner, conn, "r2")
    assert too_large.failure_code == step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE

    conn.max_message_size = 64 * 1024 * 1024
    after = await _step(picture_runner, conn, "r3")
    assert after.failure_code == step_pb2.STEP_FAILURE_CODE_UNSPECIFIED
    assert _keyframes(after) == [True]


async def test_a_result_the_connection_refuses_restarts_the_stream(
    picture_runner: Runner,
) -> None:
    _, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    await _step(picture_runner, conn, "r1")

    conn.refuse_sends = 1
    refused = await _step(picture_runner, conn, "r2")
    assert refused.failure_code == step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE
    await asyncio.sleep(0.02)

    assert _keyframes(await _step(picture_runner, conn, "r3")) == [True]


async def test_a_refused_send_restarts_the_stream_of_a_step_requested_behind_it(
    picture_runner: Runner,
) -> None:
    codec, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    await _step(picture_runner, conn, "r1")

    # Two steps in flight; the first one's send is refused only after the
    # second has run.
    conn.refuse_sends = 1
    conn.refusal_gate = asyncio.Event()
    _queue_step(picture_runner, conn, "r2")
    _queue_step(picture_runner, conn, "r3")
    await _until(lambda: len(models[-1].inputs) == 3)
    await asyncio.sleep(0.05)
    conn.refusal_gate.set()
    await _until(lambda: len(conn.sent) == 3)

    refused = _result_of(conn, "r2")
    assert refused.failure_code == step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE
    after = _result_of(conn, "r3")
    assert _keyframes(after) == [True]
    assert rw.VideoCodecs().decoder(codec).decode(_video(after)[0].data) is not None


async def test_resuming_a_track_starts_its_next_frames_at_a_keyframe(
    picture_runner: Runner,
) -> None:
    codec, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    await _step(picture_runner, conn, "r1")
    assert _keyframes(await _step(picture_runner, conn, "r2")) == [False]

    # A client whose decoder lost the stream pauses and resumes the track.
    picture_runner.pause_track(conn.id, "main_video")
    picture_runner.resume_track(conn.id, "main_video")
    after = await _step(picture_runner, conn, "r3")

    assert _keyframes(after) == [True]
    assert rw.VideoCodecs().decoder(codec).decode(_video(after)[0].data) is not None
    assert _keyframes(await _step(picture_runner, conn, "r4")) == [False]


async def test_resuming_another_track_leaves_the_stream_where_it_was(
    picture_runner: Runner,
) -> None:
    _, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    await _step(picture_runner, conn, "r1")

    picture_runner.resume_track(conn.id, "webcam")

    assert _keyframes(await _step(picture_runner, conn, "r2")) == [False]


async def test_resuming_a_track_on_another_connection_leaves_this_stream_where_it_was(
    picture_runner: Runner,
) -> None:
    _, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    other = await _encoding(picture_runner, encoding, cid=1003)
    await _step(picture_runner, conn, "r1")
    await _step(picture_runner, other, "o1")

    picture_runner.resume_track(other.id, "main_video")

    assert _keyframes(await _step(picture_runner, conn, "r2")) == [False]
    assert _keyframes(await _step(picture_runner, other, "o2")) == [True]


async def test_a_step_without_frames_leaves_the_stream_where_it_was(
    picture_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    codec, encoding = _output_codec()
    encoded: list[int] = []
    encode = step_frames_module.TrackEncoder.encode

    def counting(self: Any, pictures: list[np.ndarray], rate: float | None) -> Any:
        encoded.append(len(pictures))
        return encode(self, pictures, rate)

    monkeypatch.setattr(step_frames_module.TrackEncoder, "encode", counting)
    conn = await _encoding(picture_runner, encoding)
    first = await _step(picture_runner, conn, "r1")
    empty = await _step(picture_runner, conn, "r2", state={"prompt": "no frames"})
    third = await _step(picture_runner, conn, "r3", state={"prompt": "a forest"})

    assert list(empty.tracks) == []
    assert encoded == [1, 1]
    assert _keyframes(third) == [False]
    decoder = rw.VideoCodecs().decoder(codec)
    for result in (first, third):
        assert decoder.decode(_video(result)[0].data) is not None


async def test_a_connection_that_negotiated_no_encoding_gets_raw_frames(
    picture_runner: Runner,
) -> None:
    # An older client: no encoded step frames in its offer.
    plain = await _stepped(picture_runner)
    raw = await _encoding(picture_runner, step_pb2.FRAME_ENCODING_RAW, cid=1003)

    for conn in (plain, raw):
        [frame] = _video(await _step(picture_runner, conn, "r1"))
        assert (frame.encoding, frame.keyframe, list(frame.shape)) == (
            step_pb2.FRAME_ENCODING_RAW,
            False,
            [64, 96, 3],
        )
        assert len(frame.data) == 64 * 96 * 3


async def test_a_one_frame_720p_step_is_tens_of_kilobytes(
    monkeypatch: pytest.MonkeyPatch, register_model: Callable[[type], None]
) -> None:
    _, encoding = _output_codec()
    monkeypatch.setattr(PictureModel, "size", (720, 1280))
    register_model(PictureModel)
    runner = await _runner(monkeypatch, PictureModel)
    try:
        conn = await _encoding(runner, encoding)
        await _step(runner, conn, "r1")
        await _step(runner, conn, "r2")
        assert len(conn.sent[-1]) < 100_000
        assert _keyframes(conn.replies()[-1].step_result) == [False]
    finally:
        await runner.stop()


async def test_a_replaced_connections_step_leaves_its_replacements_stream_alone(
    picture_runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    codec, encoding = _output_codec()
    release = threading.Event()
    generate = PictureModel.generate

    def held(self: PictureModel, input: tuple[str, list[str], list[int]]) -> Any:
        release.wait(5)
        return generate(self, input)

    monkeypatch.setattr(PictureModel, "generate", held)
    old = await _encoding(picture_runner, encoding)
    _queue_step(picture_runner, old, "r1")
    await _until(lambda: len(models) > 0)
    await asyncio.sleep(0.05)

    # The same id reconnects and asks for a step while the old one still runs.
    new = await _encoding(picture_runner, encoding)
    _queue_step(picture_runner, new, "r2")
    await asyncio.sleep(0.05)
    release.set()
    await _until(lambda: len(new.sent) == 1)
    await asyncio.sleep(0.05)

    assert [reply.request_id for reply in new.replies()] == ["r2"]
    result = _result_of(new, "r2")
    assert _keyframes(result) == [True]
    assert rw.VideoCodecs().decoder(codec).decode(_video(result)[0].data) is not None


async def test_closing_a_connection_frees_its_encoders(picture_runner: Runner) -> None:
    _, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    await _step(picture_runner, conn, "r1")
    assert len(picture_runner._steps.streams) == 1

    picture_runner.connection_closed(conn.id)
    assert len(picture_runner._steps.streams) == 0


async def test_ending_the_session_frees_every_connections_encoders(
    picture_runner: Runner,
) -> None:
    _, encoding = _output_codec()
    conn = await _encoding(picture_runner, encoding)
    await _step(picture_runner, conn, "r1")
    assert len(picture_runner._steps.streams) == 1

    # The runtime closes the connection itself, which reports no close back.
    picture_runner.stop_session()
    await _until(lambda: picture_runner._sm.current_state is SessionState.READY)
    assert len(picture_runner._steps.streams) == 0
