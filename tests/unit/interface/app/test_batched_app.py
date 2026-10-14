"""The batched application: several sessions, one model, one ``generate_batch()`` per step.

Each test runs a small batched app on its own model thread, the way the
runtime does, and posts session-addressed facts and commands into it. The
outbound sinks record what came out and which session key the model was
serving when it did, which is how the runtime routes it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest

from reactor_runtime import (
    BatchedReactorApp,
    InputBuffer,
    InputField,
    InputState,
    JoinPolicy,
    MediaInput,
    Output,
    StepCompleted,
    StepOutcome,
    TrackPayload,
    Video,
    session_ended,
    session_started,
)
from reactor_runtime.core.model import (
    ClientConnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import CompletedStep, ConnId, InputFrame, MediaChunk
from reactor_runtime.interface.app.batched_app import _Session, current_session_key
from reactor_runtime.interface.model.contract import ModelContract


class Frame(Output):
    main_video: Video


class Camera(MediaInput):
    webcam: Video


class State(InputState):
    value: int = InputField(default=0)


def _frame(value: int) -> Frame:
    return Frame(main_video=np.full((2, 2, 3), value % 256, dtype=np.uint8))


class Summing(BatchedReactorApp):
    """Paints each session's ``value`` plus its step count; records every batch."""

    state: State
    max_sessions = 4

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []
        self.counts: dict[str, int] = {}
        self.started: list[str] = []
        self.ended: list[str] = []

    @session_started
    def begin(self) -> None:
        assert self.session_id is not None
        self.started.append(self.session_id)
        self.counts[self.session_id] = 0

    @session_ended
    def end(self) -> None:
        assert self.session_id is not None
        self.ended.append(self.session_id)

    def generate_batch(self, inputs: dict[str, State]) -> Mapping[str, Any]:
        self.batches.append(list(inputs))
        results: dict[str, Any] = {}
        for sid, state in inputs.items():
            self.counts[sid] += 1
            results[sid] = _frame(state.value * 10 + self.counts[sid] % 10)
        return results


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Summing)


@dataclass
class Sinks:
    """What the model handed the runtime, each with the session key it was serving."""

    media: list[tuple[int | None, MediaChunk]] = field(default_factory=list)
    steps: list[tuple[int | None, CompletedStep]] = field(default_factory=list)
    replies: list[tuple[int | None, ConnId]] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)
    crashed: list[BaseException] = field(default_factory=list)

    def frames(self, key: int) -> list[int]:
        return [
            int(chunk.bundle.tracks["main_video"].data[0, 0, 0])
            for owner, chunk in self.media
            if owner == key
        ]


def _serve(app: BatchedReactorApp) -> Sinks:
    sinks = Sinks()
    app.bind_output(
        broadcast=lambda message: None,
        addressed=lambda conn, message, request: sinks.replies.append(
            (current_session_key(), conn)
        ),
        media=lambda chunk: sinks.media.append((current_session_key(), chunk)),
        step=lambda step: sinks.steps.append((current_session_key(), step)),
    )
    app.bind_failure(sinks.crashed.append)
    app.bind_session_failure(lambda key, error: sinks.failed.append((key, error)))
    app.start_thread()
    _wait(lambda: app._inbound_q is not None)
    return sinks


def _open(app: BatchedReactorApp, key: int, sid: str) -> None:
    app.post_session_event(key, SessionStarted(sid))
    app.post_session_event(key, ClientConnected(ConnId(1000 + key), 1))


def _command(app: BatchedReactorApp, key: int, name: str, **args: Any) -> None:
    command = ModelContract.of(type(app)).validate(name, args)
    app.submit_session_command(key, command, ConnId(1000 + key), "r1")


def _wait(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


@pytest.fixture
def app() -> Any:
    model = Summing()
    yield model
    model.stop()
    if model._thread is not None:
        model._thread.join(timeout=5)


def test_one_generate_batch_serves_every_live_session(app: Summing) -> None:
    sinks = _serve(app)
    _open(app, 1, "a")
    _open(app, 2, "b")
    _open(app, 3, "c")

    _wait(lambda: any(len(batch) == 3 for batch in app.batches))
    assert any(sorted(batch) == ["a", "b", "c"] for batch in app.batches)
    _wait(lambda: all(sinks.frames(key) for key in (1, 2, 3)))


def test_each_session_gets_its_own_output_routed_by_its_key(app: Summing) -> None:
    sinks = _serve(app)
    _open(app, 1, "a")
    _open(app, 2, "b")
    _command(app, 1, "set_value", value=1)
    _command(app, 2, "set_value", value=2)

    _wait(lambda: any(f >= 20 for f in sinks.frames(2)) and any(f >= 10 for f in sinks.frames(1)))
    # A session's frames carry its own value: a command reached the state of
    # the session it was sent to, and its output came back under its key.
    assert all(f < 20 for f in sinks.frames(1))
    assert not any(10 <= f < 20 for f in sinks.frames(2))
    assert {key for key, _ in sinks.steps} == {1, 2}


def test_a_handler_reply_goes_to_its_own_session(app: Summing) -> None:
    sinks = _serve(app)
    _open(app, 1, "a")
    _open(app, 2, "b")
    _command(app, 2, "set_value", value=5)

    _wait(lambda: bool(sinks.replies))
    assert sinks.replies[0] == (2, ConnId(1002))


def test_hooks_run_per_session_with_its_id(app: Summing) -> None:
    _serve(app)
    _open(app, 1, "a")
    _open(app, 2, "b")
    _wait(lambda: len(app.started) == 2)
    app.post_session_event(1, SessionEnded("a", EndReason.STOPPED))

    _wait(lambda: app.ended == ["a"])
    _wait(lambda: 1 not in app._sessions)
    batches_before = len(app.batches)
    _wait(lambda: len(app.batches) > batches_before + 2)
    assert app.batches[-1] == ["b"]


def test_a_session_without_a_client_takes_no_step(app: Summing) -> None:
    _serve(app)
    app.post_session_event(1, SessionStarted("a"))
    _open(app, 2, "b")

    _wait(lambda: len(app.batches) > 3)
    assert all(batch == ["b"] for batch in app.batches)


class FailsOne(Summing):
    def generate_batch(self, inputs: dict[str, State]) -> Mapping[str, Any]:
        results = dict(super().generate_batch(inputs))
        if "bad" in results:
            results["bad"] = ValueError("bad latent")
        return results


def test_a_per_session_error_fails_that_session_alone(
    register_model: Callable[[type], None],
) -> None:
    register_model(FailsOne)
    app = FailsOne()
    try:
        sinks = _serve(app)
        _open(app, 1, "good")
        _open(app, 2, "bad")

        _wait(lambda: bool(sinks.failed))
        assert sinks.failed[0] == (2, "bad latent")
        frames_after_failure = len(sinks.frames(1))
        _wait(lambda: len(sinks.frames(1)) > frames_after_failure + 3)
        assert not sinks.crashed
        assert app.batches[-1] == ["good"]
    finally:
        app.stop()


class EmitsBadMetadata(Summing):
    async def process_output(self, outcome: StepOutcome) -> Output | StepCompleted | None:
        if self.session_id == "bad":
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            return Frame(main_video=TrackPayload(frame, metadata={"latent": object()}))
        return await super().process_output(outcome)


def test_an_emit_error_fails_that_session_alone(
    register_model: Callable[[type], None],
) -> None:
    register_model(EmitsBadMetadata)
    app = EmitsBadMetadata()
    try:
        sinks = _serve(app)
        _open(app, 1, "good")
        _open(app, 2, "bad")

        _wait(lambda: bool(sinks.failed))
        assert [key for key, _ in sinks.failed] == [2]
        frames_after_failure = len(sinks.frames(1))
        _wait(lambda: len(sinks.frames(1)) > frames_after_failure + 3)
        assert not sinks.crashed
        assert app.batches[-1] == ["good"]
    finally:
        app.stop()


class RaisesBatch(Summing):
    def generate_batch(self, inputs: dict[str, State]) -> Mapping[str, Any]:
        raise RuntimeError("device lost")


def test_a_raise_from_generate_batch_crashes_the_model(
    register_model: Callable[[type], None],
) -> None:
    register_model(RaisesBatch)
    app = RaisesBatch()
    try:
        sinks = _serve(app)
        _open(app, 1, "a")
        _open(app, 2, "b")

        _wait(lambda: bool(sinks.crashed))
        assert str(sinks.crashed[0]) == "device lost"
        assert not sinks.failed
    finally:
        app.stop()


class RecoversBatch(RaisesBatch):
    async def process_output(self, outcome: StepOutcome) -> Output | None:
        return None


def test_a_batch_error_the_app_recovers_from_keeps_serving(
    register_model: Callable[[type], None],
) -> None:
    register_model(RecoversBatch)
    app = RecoversBatch()
    try:
        sinks = _serve(app)
        _open(app, 1, "a")

        _wait(lambda: len(sinks.steps) > 3)
        assert not sinks.crashed
    finally:
        app.stop()


class Aligned(Summing):
    join_policy = JoinPolicy.BOUNDARY

    def at_step_boundary(self) -> bool:
        return len(self.batches) % 4 == 0


def test_a_boundary_policy_holds_a_new_session_until_the_boundary(
    register_model: Callable[[type], None],
) -> None:
    register_model(Aligned)
    app = Aligned()
    try:
        _serve(app)
        _open(app, 1, "a")
        _wait(lambda: len(app.batches) >= 2)
        _open(app, 2, "b")

        _wait(lambda: any("b" in batch for batch in app.batches))
        first = next(index for index, batch in enumerate(app.batches) if "b" in batch)
        assert first % 4 == 0
    finally:
        app.stop()


class Watching(Summing):
    camera: Camera

    async def process_input(self) -> Any:
        frames = cast(InputBuffer, self.camera.webcam).try_read()
        value = int(frames[-1].data[0, 0, 0]) if frames else 0
        return State(value=value)


def test_media_reaches_the_buffers_of_its_own_session(
    register_model: Callable[[type], None],
) -> None:
    register_model(Watching)
    app = Watching()
    try:
        sinks = _serve(app)
        _open(app, 1, "a")
        _open(app, 2, "b")
        pixel = np.full((2, 2, 3), 7, dtype=np.uint8)
        _wait(lambda: 2 in app._sessions)
        app.push_session_media(2, "webcam", InputFrame(data=pixel, pts=0.0))

        _wait(lambda: any(f >= 70 for f in sinks.frames(2)))
        assert all(f < 70 for f in sinks.frames(1))
    finally:
        app.stop()


def test_max_sessions_must_be_positive() -> None:
    with pytest.raises(TypeError, match="max_sessions"):

        class Broken(BatchedReactorApp):
            max_sessions = 0


def test_a_single_session_runtime_drives_it_through_the_plain_entrypoints(app: Summing) -> None:
    sinks = _serve(app)
    app.post_reactor_event(SessionStarted("only"))
    app.post_reactor_event(ClientConnected(ConnId(1001), 1))
    command = ModelContract.of(Summing).validate("set_value", {"value": 4})
    app.submit_command(command, ConnId(1001), None)

    _wait(lambda: any(40 <= f < 50 for f in sinks.frames(0)))
    assert app.started == ["only"]


def test_a_session_started_right_after_one_ends_on_the_same_key_steps(app: Summing) -> None:
    sinks = _serve(app)
    app.post_reactor_event(SessionStarted("first"))
    app.post_reactor_event(ClientConnected(ConnId(1001), 1))
    _wait(lambda: bool(sinks.frames(0)))
    app.post_reactor_event(SessionEnded("first", EndReason.STOPPED))
    app.post_reactor_event(SessionStarted("second"))
    app.post_reactor_event(ClientConnected(ConnId(1002), 1))

    _wait(lambda: app.started == ["first", "second"] and app.ended == ["first"])
    _wait(lambda: bool(app.batches) and app.batches[-1] == ["second"])
    assert 0 in app._sessions


def test_a_session_registered_while_the_loop_scans_the_table_keeps_it_running(
    app: Summing, monkeypatch: pytest.MonkeyPatch
) -> None:
    sinks = _serve(app)
    _open(app, 1, "a")
    _wait(lambda: bool(sinks.frames(1)))
    live = _Session.live
    registered: list[int] = []

    # A start posted from the runtime thread registers its session while the
    # model thread walks the table; this lands one inside that walk.
    def live_with_a_start_racing_in(session: _Session) -> bool:
        if not registered:
            registered.append(2)
            app.post_session_event(2, SessionStarted("b"))
            app.post_session_event(2, ClientConnected(ConnId(1002), 1))
        return live.fget(session)

    monkeypatch.setattr(_Session, "live", property(live_with_a_start_racing_in))

    _wait(lambda: bool(sinks.frames(2)))
    assert not sinks.crashed
