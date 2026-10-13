"""Surface and behaviour checks for the stepped echo example.

The model is driven two ways. Most checks run it in a stepped session through
its own step loop and inbound queue, with the outbound sinks recording each
step's report. The rest drive it through a runner, the way a client does: step
requests in on the data channel, step results out, and saved steps on disk.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime import StepOutcome, UploadedFile, protocol
from reactor_runtime.core import (
    ConnectionCapabilities,
    ConnId,
    MediaChunk,
    RuntimeConfig,
    StepResultsConfig,
)
from reactor_runtime.core.model import ClientConnected, SessionStarted
from reactor_runtime.core.values import CompletedStep, InputFrame, StepRefusal
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope, StepEnvelope
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config
from reactor_runtime.protocol.common import dict_to_struct
from reactor_runtime.runner.runner import Runner
from reactor_wire.v1 import common_pb2, data_pb2, model_pb2, platform_pb2, step_pb2
from stepped_echo import REPORT_FILE, SteppedEcho, SteppedEchoOutput

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "stepped_echo"
CLIENT = ConnId(1002)


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(SteppedEcho)


def _frame(value: int, size: tuple[int, int] = (4, 6)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


# -- the client contract ------------------------------------------------------


def test_commands_are_the_state_setters_tag_and_fail() -> None:
    assert set(ModelContract.of(SteppedEcho).commands) == {
        "set_label",
        "set_overlay",
        "tag",
        "fail",
    }


def test_tag_and_fail_each_take_one_string() -> None:
    commands = ModelContract.of(SteppedEcho).commands
    assert set(commands["tag"].command.__command_fields__) == {"value"}
    assert set(commands["fail"].command.__command_fields__) == {"reason"}


def test_tracks_are_webcam_in_and_main_video_out() -> None:
    tracks = ModelContract.of(SteppedEcho).tracks
    assert {name: (t.kind.value, t.direction.value) for name, t in tracks.items()} == {
        "webcam": ("video", "in"),
        "main_video": ("video", "out"),
    }


def test_the_model_runs_the_default_loop_so_it_can_be_stepped() -> None:
    assert "run" not in vars(SteppedEcho)
    assert SteppedEcho.supports_stepped()


def test_manifest_resolves_to_the_app_class_and_keeps_every_step() -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert import_model_class(cfg.model_ref) is SteppedEcho
    assert cfg.model_name == "stepped-echo"
    assert cfg.step_results.enabled


# -- steps, through the model's own loop ---------------------------------------


class Harness:
    """The app in a stepped session, its inbound queue drained and its loop running."""

    def __init__(self) -> None:
        self.app = SteppedEcho()
        self.reports: list[CompletedStep] = []
        self.refusals: list[StepRefusal] = []
        self.app._on_loop_ready()
        self.app.bind_output(
            broadcast=lambda message: None,
            addressed=lambda *args: None,
            media=lambda chunk: None,
            step=self.reports.append,
            step_refusal=self.refusals.append,
        )
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.app._inbound_q = self.queue
        self.tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self.tasks = [
            asyncio.create_task(self.app._inbound_loop()),
            asyncio.create_task(self.app.run()),
        ]
        self.queue.put_nowait(SessionStarted("s", stepped=True))
        self.queue.put_nowait(ClientConnected(CLIENT, 1))
        await self.settle()

    def command(self, name: str, **args: Any) -> None:
        command = ModelContract.of(SteppedEcho).validate(name, args)
        self.queue.put_nowait(CommandEnvelope(command, CLIENT, None))

    def step(
        self,
        request_id: str,
        *commands: tuple[str, dict[str, Any]],
        frames: Sequence[InputFrame] = (),
    ) -> None:
        """Post a step request the way the runtime does: its commands, then the step."""
        for name, args in commands:
            self.command(name, **args)
        inputs = {"webcam": tuple(frames)} if frames else {}
        self.queue.put_nowait(StepEnvelope(inputs=inputs, conn_id=CLIENT, request_id=request_id))

    async def settle(self) -> None:
        for _ in range(40):
            await asyncio.sleep(0.005)
            if self.queue.empty() and self.app._pending_step is None:
                return

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


@pytest.fixture
async def harness() -> Any:
    h = Harness()
    await h.start()
    try:
        yield h
    finally:
        await h.stop()


def _video(step: CompletedStep) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the one frame a step sent on ``main_video``, and its metadata as JSON."""
    assert step.bundle is not None
    assert list(step.bundle.tracks) == ["main_video"]
    assert step.bundle.frame_count == 1
    track = step.bundle.tracks["main_video"]
    assert isinstance(track.metadata, bytes)
    return track.data, json.loads(track.metadata.decode("utf-8"))


async def test_a_step_sends_back_its_last_webcam_frame_untouched(harness: Harness) -> None:
    last = np.random.default_rng(0).integers(0, 255, (4, 6, 3), dtype=np.uint8)
    harness.step(
        "r1",
        frames=[
            InputFrame(data=_frame(1), metadata=b"first"),
            InputFrame(data=last, metadata='{"seq": 2, "note": "é"}'.encode()),
        ],
    )
    await harness.settle()

    [step] = harness.reports
    frame, metadata = _video(step)
    np.testing.assert_array_equal(frame, last)
    assert metadata["inputs"] == 2
    assert metadata["input_metadata"] == '{"seq": 2, "note": "é"}'
    assert step.error is None


async def test_a_step_without_webcam_frames_sends_a_black_frame(harness: Harness) -> None:
    harness.step("r1")
    await harness.settle()

    frame, metadata = _video(harness.reports[0])
    assert frame.shape == (16, 16, 3)
    assert frame.dtype == np.uint8
    assert not frame.any()
    assert (metadata["inputs"], metadata["input_metadata"]) == (0, None)


async def test_a_last_frame_without_metadata_reports_null(harness: Harness) -> None:
    harness.step("r1", frames=[InputFrame(data=_frame(1), metadata=b"x"), InputFrame(_frame(2))])
    await harness.settle()

    _, metadata = _video(harness.reports[0])
    assert metadata["input_metadata"] is None


async def test_the_report_holds_exactly_what_the_step_saw(harness: Harness) -> None:
    harness.step(
        "r1",
        ("set_label", {"label": "hello"}),
        ("tag", {"value": "a"}),
        ("tag", {"value": "b"}),
        frames=[InputFrame(data=_frame(3), metadata=b"m")],
    )
    await harness.settle()

    _, metadata = _video(harness.reports[0])
    assert metadata == {
        "label": "hello",
        "tags": ["a", "b"],
        "inputs": 1,
        "input_metadata": "m",
        "overlay": None,
        "error": None,
    }


async def test_tags_belong_to_the_step_they_were_sent_with(harness: Harness) -> None:
    harness.step("r1", ("tag", {"value": "one"}), ("tag", {"value": "two"}))
    harness.step("r2", ("tag", {"value": "three"}))
    harness.step("r3")
    await harness.settle()

    assert [_video(step)[1]["tags"] for step in harness.reports] == [
        ["one", "two"],
        ["three"],
        [],
    ]


async def test_a_tag_sent_between_steps_lands_on_the_next_one(harness: Harness) -> None:
    harness.step("r1")
    harness.command("tag", value="between")
    harness.step("r2")
    await harness.settle()

    assert [_video(step)[1]["tags"] for step in harness.reports] == [[], ["between"]]


async def test_the_label_holds_across_steps_including_an_empty_one(harness: Harness) -> None:
    harness.step("r1", ("set_label", {"label": "kept"}), frames=[InputFrame(_frame(1))])
    harness.step("r2")
    harness.step("r3", ("set_label", {"label": "changed"}))
    await harness.settle()

    labels = [_video(step)[1]["label"] for step in harness.reports]
    assert labels == ["kept", "kept", "changed"]
    assert [r.request.index for r in harness.reports if r.request] == [1, 2, 3]


async def test_the_label_starts_empty(harness: Harness) -> None:
    harness.step("r1")
    await harness.settle()

    assert _video(harness.reports[0])[1]["label"] == ""


async def test_each_step_keeps_its_report_as_report_json(harness: Harness) -> None:
    harness.step("r1", ("tag", {"value": "x"}), frames=[InputFrame(_frame(1), metadata=b"m")])
    await harness.settle()

    [step] = harness.reports
    assert set(step.files) == {REPORT_FILE}
    assert step.bundle is not None
    assert step.files[REPORT_FILE] == step.bundle.tracks["main_video"].metadata
    assert json.loads(step.files[REPORT_FILE])["tags"] == ["x"]


async def test_fail_completes_the_step_with_its_reason_and_no_report_file(
    harness: Harness,
) -> None:
    harness.step("r1", ("fail", {"reason": "asked to fail"}), frames=[InputFrame(_frame(5))])
    harness.step("r2")
    await harness.settle()

    assert harness.refusals == []
    failed, after = harness.reports
    assert failed.request is not None
    assert failed.request.index == 1
    assert failed.error == "asked to fail"
    assert dict(failed.files) == {}
    frame, metadata = _video(failed)
    np.testing.assert_array_equal(frame, _frame(5))
    assert metadata["error"] == "asked to fail"
    # The failure belonged to that step alone.
    assert after.error is None
    assert set(after.files) == {REPORT_FILE}


async def test_an_overlay_upload_reaches_the_step_and_its_report(harness: Harness) -> None:
    overlay = UploadedFile(name="cat.png", mime_type="image/png", data=b"\x89PNG..")
    harness.step("r1", ("set_overlay", {"overlay": overlay}))
    harness.step("r2")
    await harness.settle()

    assert harness.app.state.overlay == overlay
    reported = [_video(step)[1]["overlay"] for step in harness.reports]
    assert reported == [{"name": "cat.png", "mime_type": "image/png", "size": 6}] * 2


async def test_a_streamed_turn_sends_what_arrived_since_the_last_one_or_black() -> None:
    app = SteppedEcho()
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda message: None, addressed=lambda *args: None, media=lambda c: None
    )
    await app._dispatch_reactor_event(SessionStarted("s"))

    async def turn() -> tuple[np.ndarray, dict[str, Any]]:
        input = await app.process_input()
        report = await app.process_output(StepOutcome(result=app.generate(input)))
        assert isinstance(report.output, SteppedEchoOutput)
        metadata = report.output.__metadata__["main_video"]
        assert isinstance(metadata, bytes)
        return np.asarray(report.output.main_video), json.loads(metadata)

    app._input_buffers["webcam"].push(InputFrame(data=_frame(1)))
    app._input_buffers["webcam"].push(InputFrame(data=_frame(2)))
    frame, metadata = await turn()
    assert (int(frame[0, 0, 0]), metadata["inputs"]) == (2, 2)
    frame, metadata = await turn()
    assert (frame.shape, metadata["inputs"]) == ((16, 16, 3), 0)


async def test_output_is_one_frame_on_main_video() -> None:
    assert list(SteppedEchoOutput.__tracks__) == ["main_video"]


# -- steps, through a runner ----------------------------------------------------

V1 = protocol.ProtocolVersion.V1
DATA = protocol.Channel.DATA
SERVER = protocol.Direction.SERVER


class Connection:
    """A v1 connection that records the data-channel messages sent to it."""

    max_message_size = 64 * 1024 * 1024
    message_chunking = True

    def __init__(self) -> None:
        self.id = CLIENT
        self.capabilities = ConnectionCapabilities(carries_video=True)
        self.protocol_version = V1
        self.sent: list[bytes | str] = []

    def send_step_result(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def send_message(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def send_control(self, payload: bytes | str) -> None: ...

    def send_media(self, chunk: MediaChunk) -> None: ...

    def flush_media(self) -> None: ...

    def set_media_rate(self, fps: float) -> None: ...

    def set_media_depth(self, depth: int) -> None: ...

    def resume_track(self, name: str) -> None: ...

    def pause_track(self, name: str) -> None: ...

    async def close(self) -> None: ...

    def results(self) -> list[data_pb2.DataServerMessage]:
        codec = protocol.select(V1)
        replies: list[data_pb2.DataServerMessage] = []
        for frame in self.sent:
            reply = codec.decode(frame, DATA, SERVER)
            assert isinstance(reply, data_pb2.DataServerMessage)
            replies.append(reply)
        return replies


@pytest.fixture
async def runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: SteppedEcho)
    runner = Runner(
        RuntimeConfig(
            model_ref="stepped_echo:SteppedEcho",
            step_results=StepResultsConfig(enabled=True),
        )
    )
    assert runner.step_store is not None
    runner.step_store._root = tmp_path / "steps"
    await runner.start()
    try:
        yield runner
    finally:
        await runner.stop()


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(400):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached within 2 s")


def _request_step(
    request_id: str,
    *,
    state: dict[str, Any] | None = None,
    commands: Sequence[tuple[str, dict[str, Any]]] = (),
    frames: Sequence[tuple[np.ndarray, bytes | None]] = (),
) -> bytes:
    inputs = []
    if frames:
        inputs.append(
            step_pb2.TrackFrames(
                track="webcam",
                kind=platform_pb2.TRACK_KIND_VIDEO,
                frames=[
                    step_pb2.Frame(
                        encoding=step_pb2.FRAME_ENCODING_RAW,
                        dtype="uint8",
                        shape=list(data.shape),
                        data=data.tobytes(),
                        metadata=metadata,
                    )
                    for data, metadata in frames
                ],
            )
        )
    return data_pb2.DataClientMessage(
        request_id=request_id,
        kind=common_pb2.MessageKind.MESSAGE_KIND_REQUEST,
        request_step=step_pb2.RequestStep(
            state=dict_to_struct(state or {}),
            commands=[
                model_pb2.Command(type=name, data=dict_to_struct(args)) for name, args in commands
            ],
            inputs=inputs,
        ),
    ).SerializeToString()


async def _stepped(runner: Runner) -> Connection:
    runner.start_session({"generation": "stepped"})
    conn = Connection()
    runner.connection_opened(conn)
    return conn


async def test_a_client_step_is_answered_with_the_frame_its_report_and_report_json(
    runner: Runner, tmp_path: Path
) -> None:
    conn = await _stepped(runner)
    upload_id = runner.uploads.create_slot("cat.png", "image/png", 3)
    runner.uploads.put(upload_id, b"cat")
    frame = _frame(9)
    runner.message_received(
        conn.id,
        _request_step(
            "r1",
            state={"label": "hi", "overlay": {"upload_id": upload_id}},
            commands=[("tag", {"value": "t"})],
            frames=[(_frame(1), None), (frame, b"seq-2")],
        ),
        V1,
        DATA,
    )
    await _until(lambda: len(conn.sent) == 1)

    [reply] = conn.results()
    result = reply.step_result
    assert (reply.request_id, result.index) == ("r1", 1)
    assert not result.HasField("error")
    [track] = result.tracks
    [sent] = track.frames
    assert track.track == "main_video"
    assert np.frombuffer(sent.data, np.uint8).reshape(tuple(sent.shape)).tolist() == frame.tolist()
    expected = {
        "label": "hi",
        "tags": ["t"],
        "inputs": 2,
        "input_metadata": "seq-2",
        "overlay": {"name": "cat.png", "mime_type": "image/png", "size": 3},
        "error": None,
    }
    assert json.loads(sent.metadata) == expected
    assert [(f.name, f.content_type) for f in result.files] == [(REPORT_FILE, "application/json")]

    saved = tmp_path / "steps"
    await _until(lambda: any(saved.glob(f"*/1/{REPORT_FILE}")))
    [report] = saved.glob(f"*/1/{REPORT_FILE}")
    assert report.read_bytes() == sent.metadata


async def test_a_failed_client_step_carries_its_error_and_no_files(runner: Runner) -> None:
    conn = await _stepped(runner)
    runner.message_received(
        conn.id, _request_step("r1", commands=[("fail", {"reason": "nope"})]), V1, DATA
    )
    await _until(lambda: len(conn.sent) == 1)

    result = conn.results()[0].step_result
    assert (result.index, result.error) == (1, "nope")
    assert list(result.files) == []
    assert len(result.tracks[0].frames) == 1
