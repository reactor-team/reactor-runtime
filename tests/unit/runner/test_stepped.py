"""The stepped path on its own: driven through a fake host, with no runner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from reactor_runtime import protocol
from reactor_runtime.core import (
    CompletedStep,
    ConnId,
    MediaBundle,
    RequestedStep,
    SessionEvent,
    StepRefusal,
    StepRefusalKind,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.interface.internal.bridge import CommandOutcome
from reactor_runtime.interface.internal.reactor_core import FrameSource
from reactor_runtime.message_gateway import InboundStep, StepCommand
from reactor_runtime.runner.session_start import Generation
from reactor_runtime.runner.stepped import (
    StepHost,
    StepInputError,
    StepOrder,
    StepRequests,
)
from reactor_runtime.upload_store import UnknownUploadError
from reactor_wire.v1 import data_pb2, platform_pb2, step_pb2

V1 = protocol.ProtocolVersion.V1
CLIENT = ConnId(1002)
WEBCAM = TrackInfo("webcam", TrackKind.VIDEO, direction=TrackDirection.IN)
MIC = TrackInfo("mic", TrackKind.AUDIO, direction=TrackDirection.IN)
SCREEN = TrackInfo("screen", TrackKind.VIDEO)


def _decode(frame: bytes | str) -> data_pb2.DataServerMessage:
    message = protocol.select(V1).decode(frame, protocol.Channel.DATA, protocol.Direction.SERVER)
    assert isinstance(message, data_pb2.DataServerMessage)
    return message


class FakeConnections:
    """Records each step reply and step result, decoded as the client reads it."""

    def __init__(self, *, max_message_size: int | None = None) -> None:
        self.replies: list[data_pb2.DataServerMessage] = []
        self.results: list[data_pb2.DataServerMessage] = []
        self.chunking = True
        self._max_message_size = max_message_size

    def message_chunking(self, cid: ConnId) -> bool:
        return self.chunking

    def max_message_size(self, cid: ConnId) -> int | None:
        return self._max_message_size

    def send_step_reply(
        self,
        cid: ConnId,
        encode: Callable[[protocol.ProtocolVersion], tuple[protocol.Channel, bytes | str]],
    ) -> None:
        _, frame = encode(V1)
        self.replies.append(_decode(frame))

    async def send_step_result(
        self, cid: ConnId, frame: bytes, too_large: Callable[[int], bytes]
    ) -> None:
        self.results.append(_decode(frame))


class FakeMetrics:
    """Records each count, as its outcome and the command's label."""

    def __init__(self) -> None:
        self.counted: list[tuple[str, str]] = []

    def accepted(self, command: str, *, since: float) -> None:
        self.counted.append(("accepted", command))

    def rejected(self, command: str, *, since: float) -> None:
        self.counted.append(("rejected", command))

    def unresolved_upload(self, command: str) -> None:
        self.counted.append(("unresolved_upload", command))


@dataclass(frozen=True)
class FakeContract:
    tracks: Mapping[str, TrackInfo]


@dataclass(frozen=True)
class Submitted:
    calls: list[tuple[str, str, dict[str, Any]]]
    media_input: dict[str, list[FrameSource]]
    conn_id: ConnId
    request_id: str
    max_message_size: int | None


class FakeBridge:
    """A bridge whose contract declares *tracks*, and that records each step it admits."""

    def __init__(
        self,
        tracks: Sequence[TrackInfo] = (WEBCAM, MIC, SCREEN),
        *,
        overflow: str | None = None,
        rejection: tuple[str, CommandOutcome] | None = None,
    ) -> None:
        self.contract = FakeContract({info.name: info for info in tracks})
        self.overflow = overflow
        self.rejection = rejection
        self.submitted: list[Submitted] = []

    def input_overflow(self, track: str, frames: int, samples: int = 0) -> str | None:
        return self.overflow

    def submit_step(
        self,
        calls: Sequence[tuple[str, str, dict[str, Any]]],
        media_input: Mapping[str, Sequence[FrameSource]],
        *,
        conn_id: ConnId,
        request_id: str,
        max_message_size: int | None = None,
    ) -> tuple[str, CommandOutcome] | None:
        if self.rejection is None:
            self.submitted.append(
                Submitted(
                    list(calls),
                    {track: list(frames) for track, frames in media_input.items()},
                    conn_id,
                    request_id,
                    max_message_size,
                )
            )
        return self.rejection


async def _resolve_uploads(
    name: str, args: Mapping[str, Any], uploads: Mapping[str, str], wait: float
) -> dict[str, Any]:
    """Resolve each upload to its id's file, unless the id is ``missing``."""
    if "missing" in uploads.values():
        raise UnknownUploadError("missing")
    return {**args, **{arg: f"file:{upload_id}" for arg, upload_id in uploads.items()}}


@dataclass
class Harness:
    steps: StepRequests
    connections: FakeConnections
    bridge: FakeBridge
    metrics: FakeMetrics
    journal: list[tuple[SessionEvent, dict[str, Any]]]
    spawned: list[Coroutine[Any, Any, None]]


def _requests(
    *,
    running: bool = True,
    generation: Generation = Generation.STEPPED,
    sessions: int = 1,
    bridge: FakeBridge | None = None,
    max_message_size: int | None = None,
) -> Harness:
    connections = FakeConnections(max_message_size=max_message_size)
    bridge = bridge or FakeBridge()
    metrics = FakeMetrics()
    journalled: list[tuple[SessionEvent, dict[str, Any]]] = []
    spawned: list[Coroutine[Any, Any, None]] = []

    def journal(event: SessionEvent, /, **detail: Any) -> None:
        journalled.append((event, detail))

    host = StepHost(
        bridge=lambda: bridge,
        session_running=lambda: running,
        generation=lambda: generation,
        sessions_posted=lambda: sessions,
        starting_input_done=asyncio.Event,
        connections=connections,
        journal=journal,
        resolve_uploads=_resolve_uploads,
        command_metrics=metrics,
        command_label=lambda name: name,
        codec_for=protocol.select,
        spawn=spawned.append,
    )
    return Harness(
        StepRequests(host, upload_wait=0.0), connections, bridge, metrics, journalled, spawned
    )


def _step(
    request_id: str = "r1",
    *,
    state: Mapping[str, Any] | None = None,
    commands: Sequence[StepCommand] = (),
    media_input: Sequence[step_pb2.TrackFrames] = (),
) -> InboundStep:
    return InboundStep(
        state=state or {},
        commands=tuple(commands),
        media_input=tuple(media_input),
        conn_id=CLIENT,
        request_id=request_id,
        received_at=0.0,
    )


def _frame(value: int) -> step_pb2.Frame:
    return step_pb2.Frame(
        encoding=step_pb2.FRAME_ENCODING_RAW,
        dtype="uint8",
        shape=[2, 2, 3],
        data=np.full((2, 2, 3), value, dtype=np.uint8).tobytes(),
    )


async def test_a_connections_messages_take_its_order_one_at_a_time_in_arrival_order() -> None:
    order = StepOrder()
    ran: list[str] = []
    release = asyncio.Event()

    async def first() -> None:
        async with order.hold(CLIENT):
            await release.wait()
            ran.append("first")

    async def second() -> None:
        async with order.hold(CLIENT):
            ran.append("second")

    tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
    await asyncio.sleep(0)
    assert len(order) == 1
    assert ran == []
    release.set()
    await asyncio.gather(*tasks)

    assert ran == ["first", "second"]
    assert len(order) == 0, "the entry goes once nothing holds or waits for it"


async def test_another_connection_does_not_wait_for_a_held_order() -> None:
    order = StepOrder()
    release = asyncio.Event()

    async def held() -> None:
        async with order.hold(CLIENT):
            await release.wait()

    task = asyncio.create_task(held())
    await asyncio.sleep(0)
    async with order.hold(ConnId(1003)):
        assert len(order) == 2
    release.set()
    await task


async def test_a_request_outside_a_running_session_is_not_ready_and_journals_nothing() -> None:
    harness = _requests(running=False)

    await harness.steps.submit(_step())

    [reply] = harness.connections.replies
    assert reply.request_id == "r1"
    assert reply.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_NOT_READY
    assert reply.step_result.step == 0
    assert harness.journal == []


async def test_a_request_in_a_streamed_session_is_refused() -> None:
    harness = _requests(generation=Generation.STREAMED)

    await harness.steps.submit(_step())

    [reply] = harness.connections.replies
    assert reply.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_REFUSED
    assert "not stepped" in reply.step_result.error
    assert harness.journal == []


async def test_a_request_passing_every_check_reaches_the_model_and_is_journalled() -> None:
    harness = _requests(max_message_size=4096)
    step = _step(
        state={"seed": 3, "prompt": "a lake"},
        commands=[
            StepCommand("note", {"text": "hi"}, {}),
            StepCommand("show", {}, {"image": "u1"}),
        ],
        media_input=[step_pb2.TrackFrames(track="webcam", frames=[_frame(7)])],
    )

    await harness.steps.submit(step)

    assert harness.connections.replies == []
    [submitted] = harness.bridge.submitted
    assert submitted.calls == [
        ("state.prompt", "set_prompt", {"prompt": "a lake"}),
        ("state.seed", "set_seed", {"seed": 3}),
        ("commands[0]", "note", {"text": "hi"}),
        ("commands[1]", "show", {"image": "file:u1"}),
    ]
    assert list(submitted.media_input) == ["webcam"]
    assert int(submitted.media_input["webcam"][0]().data[0, 0, 0]) == 7
    assert (submitted.conn_id, submitted.request_id, submitted.max_message_size) == (
        CLIENT,
        "r1",
        4096,
    )
    names = ["set_prompt", "set_seed", "note", "show"]
    assert harness.metrics.counted == [("accepted", name) for name in names]
    assert [(event, detail["name"]) for event, detail in harness.journal] == [
        (SessionEvent.COMMAND, name) for name in names
    ]


async def test_a_request_with_an_unresolved_upload_is_refused_naming_its_entry() -> None:
    harness = _requests()
    step = _step(commands=[StepCommand("show", {}, {"image": "missing"})])

    await harness.steps.submit(step)

    [reply] = harness.connections.replies
    assert reply.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_REFUSED
    assert reply.step_result.entry == "commands[0]"
    assert reply.step_result.error == "commands[0] references an unresolved upload"
    assert harness.bridge.submitted == []
    assert harness.metrics.counted == [("unresolved_upload", "show")]
    assert [event for event, _ in harness.journal] == [SessionEvent.ERROR]


async def test_a_request_the_contract_rejects_is_refused_naming_its_entry() -> None:
    rejection = ("state.prompt", CommandOutcome.reject("prompt", "must not be empty"))
    harness = _requests(bridge=FakeBridge(rejection=rejection))

    await harness.steps.submit(_step(state={"prompt": ""}))

    [reply] = harness.connections.replies
    assert reply.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_REFUSED
    assert reply.step_result.entry == "state.prompt"
    assert reply.step_result.error == "state.prompt is refused: prompt: must not be empty"
    assert harness.metrics.counted == [("rejected", "set_prompt")]
    assert [event for event, _ in harness.journal] == [SessionEvent.ERROR]


async def test_a_request_with_bad_media_input_is_refused_naming_its_entry() -> None:
    harness = _requests()
    step = _step(media_input=[step_pb2.TrackFrames(track="screen")])

    await harness.steps.submit(step)

    [reply] = harness.connections.replies
    assert reply.step_result.entry == "media_input[0]"
    assert harness.bridge.submitted == []
    assert [event for event, _ in harness.journal] == [SessionEvent.ERROR]


@pytest.mark.parametrize(
    ("media_input", "pictures", "error"),
    [
        (
            [step_pb2.TrackFrames(track="screen")],
            {},
            "media_input[0] names 'screen', which is not an input track of the model",
        ),
        (
            [step_pb2.TrackFrames(track="webcam"), step_pb2.TrackFrames(track="webcam")],
            {},
            "media_input[1] names 'webcam' a second time",
        ),
        (
            [step_pb2.TrackFrames(track="webcam", kind=platform_pb2.TRACK_KIND_AUDIO)],
            {},
            "media_input[0] says 'webcam' is not a video track",
        ),
        (
            [step_pb2.TrackFrames(track="mic", frames=[step_pb2.Frame()])],
            {(0, 0): np.zeros((2, 2, 3), dtype=np.uint8)},
            "media_input[0].frames[0]: an encoded video frame cannot go to audio track 'mic'",
        ),
    ],
)
def test_media_input_a_track_cannot_take_is_refused_naming_its_entry(
    media_input: list[step_pb2.TrackFrames],
    pictures: dict[tuple[int, int], Any],
    error: str,
) -> None:
    harness = _requests()

    with pytest.raises(StepInputError) as refused:
        harness.steps.check_media_input(tuple(media_input), pictures)

    assert str(refused.value) == error
    assert refused.value.entry == f"media_input[{len(media_input) - 1}]"


def test_media_input_that_overflows_its_tracks_buffer_is_refused() -> None:
    harness = _requests(bridge=FakeBridge(overflow="2 frames"))
    frames = step_pb2.TrackFrames(track="webcam", frames=[_frame(1), _frame(2), _frame(3)])

    with pytest.raises(StepInputError) as refused:
        harness.steps.check_media_input((frames,), {})

    assert refused.value.entry == "media_input[0]"
    assert str(refused.value) == (
        "media_input[0] holds 3 frames for 'webcam', whose input buffer holds 2 frames"
    )


def test_a_refusal_carries_its_code_and_entry() -> None:
    harness = _requests()

    harness.steps.refuse(
        CLIENT,
        "r1",
        "notes are closed",
        failure_code=step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED,
        code="no_notes",
        entry="commands[1]",
    )

    result = harness.connections.replies[0].step_result
    assert (result.failure_code, result.error, result.code, result.entry) == (
        step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED,
        "notes are closed",
        "no_notes",
        "commands[1]",
    )
    assert harness.journal == []


def test_a_model_refusal_from_an_earlier_session_is_answered_as_ended() -> None:
    harness = _requests(sessions=2)

    harness.steps.on_refusal(StepRefusal(CLIENT, "r1", session=1, reason="not now"))

    assert harness.connections.replies[0].step_result.failure_code == (
        step_pb2.STEP_FAILURE_CODE_SESSION_ENDED
    )


def test_a_model_refusal_keeps_its_kind_as_its_wire_code() -> None:
    harness = _requests(sessions=1)

    harness.steps.on_refusal(
        StepRefusal(CLIENT, "r1", session=1, reason="not ready", kind=StepRefusalKind.NOT_READY)
    )

    assert harness.connections.replies[0].step_result.failure_code == (
        step_pb2.STEP_FAILURE_CODE_NOT_READY
    )


def test_a_contract_refusal_names_the_entry_and_the_field() -> None:
    outcome = CommandOutcome.reject("prompt", "must not be empty")

    assert outcome.refusal("state.prompt") == ("state.prompt is refused: prompt: must not be empty")


def _completed(
    data: np.ndarray[Any, Any],
    *,
    max_message_size: int | None = None,
    files: Mapping[str, bytes] | None = None,
) -> CompletedStep:
    return CompletedStep(
        bundle=MediaBundle({"screen": TrackData(SCREEN, data)}),
        files=files or {},
        elapsed=0.25,
        request=RequestedStep(CLIENT, "r1", 4, max_message_size=max_message_size),
    )


def test_a_step_no_client_requested_is_not_encoded() -> None:
    harness = _requests()

    step = CompletedStep(bundle=None)

    assert harness.steps.encode_result(step) is None


def test_oversized_media_is_answered_too_large_without_being_encoded() -> None:
    harness = _requests()
    step = _completed(np.zeros((4, 4, 3), dtype=np.uint8), max_message_size=10)

    encoded = harness.steps.encode_result(step)
    assert encoded is not None
    assert (encoded.frame, encoded.oversized) == (None, 48)
    harness.steps.send_result(step, 4, saved=False, encoded=encoded)

    [reply] = harness.connections.replies
    result = reply.step_result
    assert result.failure_code == step_pb2.STEP_FAILURE_CODE_RESULT_TOO_LARGE
    assert (result.step, result.tracks) == (4, [])
    assert "at least 48 bytes" in result.error
    assert harness.spawned == []


def test_media_that_cannot_be_encoded_is_answered_failed() -> None:
    harness = _requests()
    step = _completed(np.zeros((2, 2), dtype=np.uint8))

    harness.steps.send_result(step, 4, saved=False)

    [reply] = harness.connections.replies
    result = reply.step_result
    assert result.failure_code == step_pb2.STEP_FAILURE_CODE_FAILED
    assert result.error.startswith("step 4's output could not be encoded: ")
    assert harness.spawned == []


async def test_a_saved_steps_files_join_its_result() -> None:
    harness = _requests()
    step = _completed(
        np.zeros((2, 2, 3), dtype=np.uint8), files={"depth.png": b"png", "notes.bin": b"abc"}
    )

    harness.steps.send_result(step, 4, saved=True)
    for send in harness.spawned:
        await send

    assert harness.connections.replies == []
    [reply] = harness.connections.results
    result = reply.step_result
    assert (result.step, result.elapsed_seconds) == (4, 0.25)
    assert [track.track for track in result.tracks] == ["screen"]
    assert [(ref.name, ref.size) for ref in result.files] == [("depth.png", 3), ("notes.bin", 3)]


async def test_an_unsaved_steps_result_names_no_files() -> None:
    harness = _requests()
    step = _completed(np.zeros((2, 2, 3), dtype=np.uint8), files={"depth.png": b"png"})

    harness.steps.send_result(step, 4, saved=False)
    for send in harness.spawned:
        await send

    [reply] = harness.connections.results
    assert list(reply.step_result.files) == []
