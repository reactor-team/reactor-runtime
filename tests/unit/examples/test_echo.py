"""Surface and behaviour checks for the echo example.

The model half is pure OpenCV, so it is tested as it is. The application half
is driven through its hooks the way the runtime's loop drives them, with frames
pushed into its input buffers the way the transport pushes them. A stepped
session is driven two ways: through the model's own step loop and inbound
queue, and through a runner, the way a client drives it, with step requests in
on the data channel and step results out.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from echo_model import EFFECTS, EchoInput, EchoModel, EchoResult

from echo import Echo, EchoMedia, EchoOutput
from reactor_runtime import ApplicationError, InputFrame, StepOutcome, TrackPayload, protocol
from reactor_runtime.core import ConnectionCapabilities, ConnId, MediaChunk, RuntimeConfig
from reactor_runtime.core.model import ClientConnected, EndReason, SessionEnded, SessionStarted
from reactor_runtime.core.values import CompletedStep, StepRefusal
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope, StepEnvelope
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config
from reactor_runtime.protocol.common import dict_to_struct
from reactor_runtime.runner.runner import Runner
from reactor_wire.v1 import common_pb2, data_pb2, platform_pb2, step_pb2

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "echo"
CLIENT = ConnId(1002)


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Echo)


def _frame(value: int = 0, size: tuple[int, int] = (8, 8)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


def _audio(samples: int) -> np.ndarray:
    return np.arange(samples, dtype=np.int16).reshape(1, -1)


# -- the client contract ------------------------------------------------------


def test_commands_are_exactly_the_state_setters() -> None:
    assert set(ModelContract.of(Echo).commands) == {
        "set_effect",
        "set_intensity",
        "set_caption",
        "set_burst",
    }


def test_set_effect_offers_every_effect() -> None:
    info = ModelContract.of(Echo).commands["set_effect"].command.__command_fields__["effect"]
    assert info.info.choices == EFFECTS


def test_the_setters_carry_the_field_bounds() -> None:
    commands = ModelContract.of(Echo).commands
    intensity = commands["set_intensity"].command.__command_fields__["intensity"].info
    caption = commands["set_caption"].command.__command_fields__["caption"].info
    burst = commands["set_burst"].command.__command_fields__["burst"].info
    assert (intensity.ge, intensity.le) == (0.0, 1.0)
    assert caption.max_length == 200
    assert (burst.ge, burst.le) == (1, 120)  # four seconds at the model's 30 fps


def test_only_the_free_text_field_asks_for_moderation() -> None:
    commands = ModelContract.of(Echo).commands
    assert commands["set_caption"].command.__command_fields__["caption"].info.moderate is True
    assert commands["set_effect"].command.__command_fields__["effect"].info.moderate is False


def test_tracks_are_bidirectional_audio_and_video() -> None:
    tracks = ModelContract.of(Echo).tracks
    assert {name: (t.kind.value, t.direction.value) for name, t in tracks.items()} == {
        "webcam": ("video", "in"),
        "mic": ("audio", "in"),
        "main_video": ("video", "out"),
        "main_audio": ("audio", "out"),
    }


def test_manifest_resolves_to_the_app_class() -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    assert import_model_class(cfg.model_ref) is Echo
    assert cfg.config_path is None


def test_the_model_runs_the_default_loop_so_it_can_be_stepped() -> None:
    assert "run" not in vars(Echo)
    assert Echo.supports_stepped()


# -- the model half -----------------------------------------------------------


@pytest.mark.parametrize("effect", EFFECTS)
def test_every_effect_returns_a_frame_of_the_same_shape(effect: str) -> None:
    model = EchoModel()
    model.load()
    frame = np.random.default_rng(0).integers(0, 255, (16, 24, 3), dtype=np.uint8)
    result = model.generate(EchoInput(frame=frame, effect=effect, intensity=1.0, caption=""))
    assert result.frame.shape == frame.shape
    assert result.frame.dtype == np.uint8


def test_no_effect_and_zero_intensity_leave_the_frame_alone() -> None:
    model = EchoModel()
    frame = _frame(77)
    untouched = model.generate(EchoInput(frame=frame, effect="none", intensity=1.0, caption=""))
    dry = model.generate(EchoInput(frame=frame, effect="invert", intensity=0.0, caption=""))
    assert untouched.frame is frame
    assert dry.frame is frame


def test_invert_at_full_intensity_inverts() -> None:
    result = EchoModel().generate(
        EchoInput(frame=_frame(10), effect="invert", intensity=1.0, caption="")
    )
    assert int(result.frame[0, 0, 0]) == 245


def test_a_caption_changes_the_frame() -> None:
    frame = _frame(0, size=(64, 128))
    result = EchoModel().generate(
        EchoInput(frame=frame, effect="none", intensity=1.0, caption="hi")
    )
    assert result.frame.any()


# -- the application half -----------------------------------------------------


class RecordingModel:
    """A model half that records its inputs and hands each frame straight back."""

    def __init__(self) -> None:
        self.inputs: list[EchoInput] = []
        self.resets = 0

    def generate(self, input: EchoInput) -> EchoResult:
        self.inputs.append(input)
        return EchoResult(frame=input.frame)

    def reset(self) -> None:
        self.resets += 1


async def _app() -> tuple[Echo, RecordingModel]:
    app = Echo()
    app.load(None)
    model = RecordingModel()
    app.engine = model  # type: ignore[ty:invalid-assignment]  # the model half by shape
    app._on_loop_ready()
    app.bind_output(
        broadcast=lambda *args: None, addressed=lambda *args: None, media=lambda c: None
    )
    await app._dispatch_reactor_event(SessionStarted("s"))
    return app, model


def _push_webcam(app: Echo, frame: np.ndarray, metadata: bytes | None = None) -> None:
    app._input_buffers["webcam"].push(InputFrame(data=frame, pts=0.0, metadata=metadata))


def _push_mic(app: Echo, *chunks: np.ndarray) -> None:
    for chunk in chunks:
        app._input_buffers["mic"].push(InputFrame(data=chunk, pts=0.0))


async def _step(app: Echo, frame: np.ndarray, metadata: bytes | None = None) -> EchoOutput | None:
    """Drive one step the way the loop does: input, generate, output."""
    _push_webcam(app, frame, metadata)
    input = await app.process_input()
    return await app.process_output(StepOutcome(result=app.generate(input)))


def test_the_media_holder_is_bound_to_the_declared_tracks() -> None:
    app = Echo()
    assert isinstance(app.media, EchoMedia)
    assert set(app._input_buffers) == {"webcam", "mic"}


async def test_process_input_refuses_until_a_webcam_frame_arrives() -> None:
    app, _ = await _app()
    with pytest.raises(ApplicationError, match="webcam"):
        await app.process_input()


async def test_process_input_takes_the_newest_frame_and_the_settings() -> None:
    app, _ = await _app()
    app.state.effect = "sepia"
    app.state.intensity = 0.5
    app.state.caption = "hello"
    _push_webcam(app, _frame(1))
    _push_webcam(app, _frame(2), metadata=b'{"seq":2}')
    input = await app.process_input()
    assert int(input.frame[0, 0, 0]) == 2  # LATEST: the backlog is dropped
    assert (input.effect, input.intensity, input.caption) == ("sepia", 0.5, "hello")
    assert app.state._metadata == b'{"seq":2}'


async def test_process_input_drains_the_microphone_in_arrival_order() -> None:
    app, _ = await _app()
    _push_webcam(app, _frame())
    _push_mic(app, _audio(3), _audio(2))
    await app.process_input()
    assert app.state._audio is not None
    assert app.state._audio.tolist() == [[0, 1, 2, 0, 1]]


async def test_process_input_trims_an_audio_backlog_from_the_head() -> None:
    app, _ = await _app()
    _push_webcam(app, _frame())
    head = np.full((1, 2000), 1, dtype=np.int16)
    kept = np.full((1, 1000), 7, dtype=np.int16)
    _push_mic(app, head, kept, kept)  # 4000 samples, over two frames' worth at 48 kHz
    await app.process_input()
    assert app.state._audio is not None
    assert app.state._audio.shape == (1, 2000)
    assert int(app.state._audio[0, 0]) == 7  # the oldest chunk went, the rest stayed in order


async def test_a_frame_without_audio_pairs_with_silence() -> None:
    app, _ = await _app()
    output = await _step(app, _frame())
    assert output is not None
    assert cast(np.ndarray, output.main_audio).shape == (1, 0)


async def test_generate_forwards_to_the_model_half() -> None:
    app, model = await _app()
    input = EchoInput(frame=_frame(), effect="blur", intensity=1.0, caption="")
    app.generate(input)
    assert model.inputs == [input]


async def test_a_burst_of_one_emits_every_frame_with_its_metadata() -> None:
    app, _ = await _app()
    output = await _step(app, _frame(5), metadata=b'{"seq":9}')
    assert isinstance(output, EchoOutput)
    assert cast(np.ndarray, output.main_video).shape == (8, 8, 3)
    assert output.__metadata__["main_video"] == b'{"seq":9}'


async def test_an_untagged_frame_goes_back_untagged() -> None:
    app, _ = await _app()
    output = await _step(app, _frame())
    assert output is not None
    assert "main_video" not in output.__metadata__


async def test_a_burst_gathers_frames_and_emits_them_as_one_batch() -> None:
    app, _ = await _app()
    app.state.burst = 3
    assert await _step(app, _frame(1), metadata=b"a") is None
    assert await _step(app, _frame(2)) is None
    output = await _step(app, _frame(3), metadata=b"c")
    assert output is not None
    assert cast(np.ndarray, output.main_video).shape == (3, 8, 8, 3)
    # One entry per frame; a frame that carried nothing takes an empty trailer.
    assert output.__metadata__["main_video"] == [b"a", b"", b"c"]
    assert len(app.burst) == 0


async def test_a_resolution_change_ends_the_batch_and_opens_the_next() -> None:
    app, _ = await _app()
    app.state.burst = 4
    await _step(app, _frame(1))
    await _step(app, _frame(2))
    early = await _step(app, _frame(3, size=(4, 4)))
    assert early is not None
    assert cast(np.ndarray, early.main_video).shape == (2, 8, 8, 3)
    assert [f.shape for f in app.burst.frames] == [(4, 4, 3)]


async def test_process_output_reraises_a_model_error() -> None:
    app, _ = await _app()
    with pytest.raises(RuntimeError, match="boom"):
        await app.process_output(StepOutcome(error=RuntimeError("boom")))


async def test_a_session_end_drops_the_batch_and_resets_the_model_half() -> None:
    app, model = await _app()
    app.state.burst = 4
    await _step(app, _frame())
    await app._dispatch_reactor_event(SessionEnded("s", EndReason.STOPPED))
    assert len(app.burst) == 0
    assert model.resets == 1


def test_output_carries_the_metadata_a_frame_arrived_with() -> None:
    output = EchoOutput(
        main_video=TrackPayload(_frame(), metadata=b'{"seq":3}'),
        main_audio=np.zeros((1, 4), np.int16),
    )
    assert output.__metadata__["main_video"] == b'{"seq":3}'


# -- a stepped session, through the model's own loop ---------------------------


async def _until(predicate: Callable[[], bool]) -> None:
    # Generous: a first step on a fresh runner starts the model's thread, which
    # a loaded CI host can take seconds over. A passing test returns at the
    # first poll that sees the condition.
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached within 10 s")


class Harness:
    """The app in a stepped session, its inbound queue drained and its loop running."""

    def __init__(self) -> None:
        self.app = Echo()
        self.app.load(None)
        self.steps: list[CompletedStep] = []
        self.refusals: list[StepRefusal] = []
        self.app._on_loop_ready()
        self.app.bind_output(
            broadcast=lambda message: None,
            addressed=lambda *args: None,
            media=lambda chunk: None,
            step=self.steps.append,
        )
        self.app._bind_step_refusal(self.refusals.append)
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

    def step(
        self,
        request_id: str,
        *commands: tuple[str, dict[str, Any]],
        media: Mapping[str, Sequence[InputFrame]] | None = None,
    ) -> None:
        """Post a step request the way the runtime does: its commands, then the step."""
        for name, args in commands:
            command = ModelContract.of(Echo).validate(name, args)
            self.queue.put_nowait(CommandEnvelope(command, CLIENT, None))
        media_input = {
            track: tuple(_source(frame) for frame in frames)
            for track, frames in (media or {}).items()
        }
        self.queue.put_nowait(
            StepEnvelope(media_input=media_input, conn_id=CLIENT, request_id=request_id)
        )

    async def settle(self) -> None:
        await asyncio.sleep(0.005)
        await _until(lambda: self.queue.empty() and self.app._pending_step is None)

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        results = await asyncio.gather(*self.tasks, return_exceptions=True)
        # Cancelling is how these end; anything else is a failure the test must show.
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result


def _source(frame: InputFrame) -> Callable[[], InputFrame]:
    """Wrap a frame as the checked source a step envelope carries."""
    return lambda: frame


def _webcam(*frames: InputFrame) -> dict[str, Sequence[InputFrame]]:
    return {"webcam": frames}


@pytest.fixture
async def harness() -> Any:
    h = Harness()
    await h.start()
    try:
        yield h
    finally:
        await h.stop()


def _video(step: CompletedStep) -> tuple[np.ndarray, Any]:
    """Return what a step sent on ``main_video``, and the metadata it carried."""
    assert step.bundle is not None
    track = step.bundle.tracks["main_video"]
    return np.asarray(track.data), track.metadata


async def test_a_step_answers_with_its_webcam_frame_processed(harness: Harness) -> None:
    harness.step(
        "r1",
        ("set_effect", {"effect": "invert"}),
        media=_webcam(InputFrame(data=_frame(10), metadata=b'{"seq":1}')),
    )
    await harness.settle()

    [step] = harness.steps
    assert step.error is None
    assert step.bundle is not None
    assert set(step.bundle.tracks) == {"main_video", "main_audio"}
    frame, metadata = _video(step)
    assert frame.shape == (8, 8, 3)
    assert int(frame[0, 0, 0]) == 245
    assert metadata == b'{"seq":1}'


async def test_a_step_takes_the_newest_webcam_frame_it_carried(harness: Harness) -> None:
    harness.step("r1", media=_webcam(InputFrame(_frame(1)), InputFrame(_frame(2))))
    await harness.settle()

    frame, _ = _video(harness.steps[0])
    assert int(frame[0, 0, 0]) == 2


async def test_a_step_plays_the_microphone_audio_it_carried(harness: Harness) -> None:
    harness.step(
        "r1",
        media={"webcam": [InputFrame(_frame())], "mic": [InputFrame(_audio(3))]},
    )
    await harness.settle()

    [step] = harness.steps
    assert step.bundle is not None
    assert np.asarray(step.bundle.tracks["main_audio"].data).tolist() == [[0, 1, 2]]


async def test_a_step_without_a_webcam_frame_is_refused_and_takes_no_number(
    harness: Harness,
) -> None:
    harness.step("r1")
    harness.step("r2", media=_webcam(InputFrame(_frame(3))))
    await harness.settle()

    [refusal] = harness.refusals
    assert refusal.request_id == "r1"
    assert "waiting for a webcam frame" in refusal.reason
    [step] = harness.steps
    assert step.request is not None
    assert (step.request.request_id, step.request.index) == ("r2", 1)


async def test_the_settings_hold_across_steps(harness: Harness) -> None:
    harness.step("r1", ("set_effect", {"effect": "invert"}), media=_webcam(InputFrame(_frame(10))))
    harness.step("r2", media=_webcam(InputFrame(_frame(20))))
    await harness.settle()

    assert [int(_video(step)[0][0, 0, 0]) for step in harness.steps] == [245, 235]


async def test_a_burst_gathers_across_steps_and_the_last_step_carries_it(
    harness: Harness,
) -> None:
    harness.step("r1", ("set_burst", {"burst": 2}), media=_webcam(InputFrame(_frame(1))))
    harness.step("r2", media=_webcam(InputFrame(_frame(2))))
    await harness.settle()

    gathering, full = harness.steps
    assert gathering.bundle is None
    assert full.bundle is not None
    assert full.bundle.frame_count == 2
    frames, _ = _video(full)
    assert [int(f[0, 0, 0]) for f in frames] == [1, 2]


# -- a stepped session, through a runner ----------------------------------------

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

    async def send_step_result(self, payload: bytes) -> bool:
        self.sent.append(payload)
        return True

    def send_message(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def send_control(self, payload: bytes | str) -> None: ...

    def send_media(self, chunk: MediaChunk) -> None: ...

    def flush_media(self, *, hold: bool = False) -> None: ...

    def set_media_rate(self, fps: float) -> None: ...

    def set_media_depth(self, depth: int) -> None: ...

    def resume_track(self, name: str) -> None: ...

    def pause_track(self, name: str) -> None: ...

    async def close(self) -> None: ...

    def results(self) -> list[step_pb2.StepResult]:
        codec = protocol.select(V1)
        results: list[step_pb2.StepResult] = []
        for frame in self.sent:
            reply = codec.decode(frame, DATA, SERVER)
            assert isinstance(reply, data_pb2.DataServerMessage)
            results.append(reply.step_result)
        return results


@pytest.fixture
async def runner(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr("reactor_runtime.runner.runner.import_model_class", lambda ref: Echo)
    runner = Runner(RuntimeConfig(model_ref="echo:Echo"))
    await runner.start()
    try:
        yield runner
    finally:
        await runner.stop()


def _request_step(
    request_id: str,
    *,
    state: dict[str, Any] | None = None,
    frames: Sequence[tuple[np.ndarray, bytes | None]] = (),
) -> bytes:
    media_input = []
    if frames:
        media_input.append(
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
            media_input=media_input,
        ),
    ).SerializeToString()


def _stepped(runner: Runner) -> Connection:
    runner.start_session({"generation": "stepped"})
    conn = Connection()
    runner.connection_opened(conn)
    return conn


async def test_a_client_step_is_answered_with_the_processed_frame(runner: Runner) -> None:
    conn = _stepped(runner)
    runner.message_received(
        conn.id,
        _request_step("r1", state={"effect": "invert"}, frames=[(_frame(10), b"seq-1")]),
        V1,
        DATA,
    )
    await _until(lambda: len(conn.sent) == 1)

    [result] = conn.results()
    assert result.step == 1
    assert result.failure_code == step_pb2.STEP_FAILURE_CODE_UNSPECIFIED
    video = next(track for track in result.tracks if track.track == "main_video")
    [sent] = video.frames
    pixels = np.frombuffer(sent.data, np.uint8).reshape(tuple(sent.shape))
    assert pixels.tolist() == np.full((8, 8, 3), 245, np.uint8).tolist()
    assert sent.metadata == b"seq-1"


async def test_a_client_step_without_a_webcam_frame_is_refused(runner: Runner) -> None:
    conn = _stepped(runner)
    runner.message_received(conn.id, _request_step("r1"), V1, DATA)
    await _until(lambda: len(conn.sent) == 1)

    [result] = conn.results()
    assert result.failure_code == step_pb2.STEP_FAILURE_CODE_REFUSED
    assert "waiting for a webcam frame" in result.error
    assert list(result.tracks) == []
