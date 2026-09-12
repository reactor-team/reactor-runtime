from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reactor_runtime import InputState, ModelMessage, Output, Video
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import ConnId
from reactor_runtime.interface.internal.reactor_core import MediaOps
from reactor_runtime.realtime import AdvancementMode, RealtimePipeline, RealtimeStepError
from reactor_runtime.realtime import pipeline as pipeline_module


@dataclass
class FakeAttach:
    session_id: str
    context: dict[str, Any] = field(default_factory=dict)
    pacing: Any = None


@dataclass
class FakeDetach:
    session_id: str


@dataclass
class FakeReset:
    session_id: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeAction:
    session_id: str
    conditioning: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeAdvance:
    session_id: str
    request_id: str
    conditioning: dict[str, Any] | None = None


class FakePacingMode(StrEnum):
    ENGINE = "engine"
    CLIENT = "client"


@dataclass
class FakeSetPacing:
    session_id: str
    pacing: FakePacingMode


@dataclass
class FakeChunk:
    session_id: str
    output: Any
    request_id: str | None = None


@dataclass
class FakeStepFailed:
    session_id: str
    request_id: str
    code: Any
    message: str
    reason: str | None = None


class FakeInbox:
    def __init__(self) -> None:
        self.messages: list[Any] = []
        self.changed = asyncio.Event()

    def put_nowait(self, message: Any) -> None:
        self.messages.append(message)
        self.changed.set()


class FakeOutbox:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[FakeChunk | FakeStepFailed] = asyncio.Queue()

    async def get(self) -> FakeChunk | FakeStepFailed:
        return await self.queue.get()


class FakeEngine:
    def manifest(self) -> SimpleNamespace:
        return SimpleNamespace(rate_hz=1000.0, frames_per_chunk=2)


class DemoState(InputState):
    prompt: str = "start"


class DemoOutput(Output):
    main_video: Video


class DemoMessage(ModelMessage):
    value: int


class DemoPipeline(RealtimePipeline):
    state: DemoState

    def __init__(self, engine: FakeEngine) -> None:
        self.test_engine = engine
        super().__init__()

    def build_engine(self, config_path: Path | None) -> Any:
        return self.test_engine


class ExternallyPacedDemoPipeline(DemoPipeline):
    advancement_mode = AdvancementMode.EXTERNAL


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> pipeline_module._EngineAPI:
    async def run_engine(
        engine: Any,
        inbox: FakeInbox,
        outbox: FakeOutbox,
        *,
        stop: asyncio.Event,
    ) -> None:
        await stop.wait()

    api = pipeline_module._EngineAPI(
        action=FakeAction,
        advance=FakeAdvance,
        attach=FakeAttach,
        chunk=FakeChunk,
        detach=FakeDetach,
        inbox=FakeInbox,
        outbox=FakeOutbox,
        pacing_mode=FakePacingMode,
        reset=FakeReset,
        set_pacing=FakeSetPacing,
        step_failed=FakeStepFailed,
        run_engine=run_engine,
    )
    monkeypatch.setattr(pipeline_module, "_load_engine_api", lambda: api)
    return api


@pytest.fixture
def pipe(fake_api: pipeline_module._EngineAPI) -> DemoPipeline:
    model = DemoPipeline(FakeEngine())
    model.load(None)
    model._on_loop_ready()
    return model


@pytest.mark.asyncio
async def test_lifecycle_pauses_without_detaching(pipe: DemoPipeline) -> None:
    await pipe._dispatch_reactor_event(SessionStarted("session"))
    engine_session_id = pipe._engine_session_id
    assert engine_session_id is not None
    assert pipe._engine_inbox.messages == [
        FakeAttach(engine_session_id, {"prompt": "start"}, FakePacingMode.CLIENT),
    ]

    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1), 1))
    pipe.state.prompt = "changed"
    await pipe._dispatch_reactor_event(ClientDisconnected(ConnId(1), 0))
    assert pipe.state.prompt == "changed"
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(2), 1))
    assert pipe._engine_session_id == engine_session_id

    await pipe._dispatch_reactor_event(SessionEnded("session", EndReason.STOPPED))
    assert pipe._engine_inbox.messages[1:] == [
        FakeSetPacing(engine_session_id, FakePacingMode.ENGINE),
        FakeSetPacing(engine_session_id, FakePacingMode.CLIENT),
        FakeSetPacing(engine_session_id, FakePacingMode.ENGINE),
        FakeSetPacing(engine_session_id, FakePacingMode.CLIENT),
        FakeDetach(engine_session_id),
    ]
    assert pipe.state is None


@pytest.mark.asyncio
async def test_conditioning_is_change_driven_and_replayed_after_reconnect(
    pipe: DemoPipeline,
) -> None:
    await pipe._dispatch_reactor_event(SessionStarted("session"))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1), 1))
    task = asyncio.create_task(pipe._conditioning_loop())

    await _wait_for_actions(pipe, 1)
    pipe.state.prompt = "next"
    await _wait_for_actions(pipe, 2)
    await pipe._dispatch_reactor_event(ClientDisconnected(ConnId(1), 0))
    pipe.state.prompt = "while-away"
    await asyncio.sleep(0.005)
    assert len(_actions(pipe)) == 2

    await pipe._dispatch_reactor_event(ClientConnected(ConnId(2), 1))
    await _wait_for_actions(pipe, 3)
    assert [message.conditioning for message in _actions(pipe)] == [
        {"prompt": "start"},
        {"prompt": "next"},
        {"prompt": "while-away"},
    ]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_reset_clears_playout_and_preserves_attachment(pipe: DemoPipeline) -> None:
    flushes: list[None] = []
    pipe.bind_output(
        broadcast=lambda message: None,
        addressed=lambda conn_id, message, request_id: None,
        media=lambda chunk: None,
        media_ops=MediaOps(
            flush=lambda: flushes.append(None),
            set_rate=lambda rate: None,
            set_depth=lambda depth: None,
        ),
    )
    await pipe._dispatch_reactor_event(SessionStarted("session"))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1), 1))
    engine_session_id = pipe._engine_session_id
    assert engine_session_id is not None
    pipe.reset_engine({"seed": 7})

    assert flushes == [None]
    assert pipe._engine_session_id == engine_session_id
    assert pipe._engine_inbox.messages[-1] == FakeReset(engine_session_id, {"seed": 7})


@pytest.mark.asyncio
async def test_default_output_routing(pipe: DemoPipeline) -> None:
    broadcasts: list[ModelMessage] = []
    media: list[Any] = []
    pipe.bind_output(
        broadcast=broadcasts.append,
        addressed=lambda conn_id, message, request_id: None,
        media=media.append,
    )

    output = DemoOutput(main_video=np.zeros((2, 2, 3), dtype=np.uint8))
    await pipe.on_output(output)
    await pipe.on_output(DemoMessage(value=3))

    assert len(media) == 1
    assert broadcasts == [DemoMessage(value=3)]
    with pytest.raises(TypeError, match="override on_output"):
        await pipe.on_output(object())


@pytest.mark.asyncio
async def test_external_advance_routes_and_returns_correlated_output(
    fake_api: pipeline_module._EngineAPI,
) -> None:
    pipe = ExternallyPacedDemoPipeline(FakeEngine())
    pipe.load(None)
    pipe._on_loop_ready()
    broadcasts: list[ModelMessage] = []
    pipe.bind_output(
        broadcast=broadcasts.append,
        addressed=lambda conn_id, message, request_id: None,
        media=lambda chunk: None,
    )
    await pipe._dispatch_reactor_event(SessionStarted("session"))
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1), 1))
    session_id = pipe._engine_session_id
    assert session_id is not None
    assert pipe._engine_inbox.messages == [
        FakeAttach(session_id, {"prompt": "start"}, FakePacingMode.CLIENT)
    ]

    output_task = asyncio.create_task(pipe._output_loop())
    advance_task = asyncio.create_task(pipe.advance_engine({"prompt": "next"}))
    request = await _wait_for_advance(pipe)
    output = DemoMessage(value=7)
    await pipe._engine_outbox.queue.put(
        FakeChunk(session_id, output, request_id=request.request_id)
    )

    assert await advance_task is output
    assert broadcasts == [output]
    assert request.conditioning == {"prompt": "next"}

    output_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await output_task


@pytest.mark.asyncio
async def test_external_advance_surfaces_terminal_step_failure(
    fake_api: pipeline_module._EngineAPI,
) -> None:
    pipe = ExternallyPacedDemoPipeline(FakeEngine())
    pipe.load(None)
    pipe._on_loop_ready()
    await pipe._dispatch_reactor_event(SessionStarted("session"))
    session_id = pipe._engine_session_id
    assert session_id is not None

    output_task = asyncio.create_task(pipe._output_loop())
    advance_task = asyncio.create_task(pipe.advance_engine())
    request = await _wait_for_advance(pipe)
    await pipe._engine_outbox.queue.put(
        FakeStepFailed(
            session_id,
            request.request_id,
            "request_rejected",
            "conditioning is invalid",
            reason="invalid_conditioning",
        )
    )

    with pytest.raises(RealtimeStepError, match="conditioning is invalid") as raised:
        await advance_task
    assert raised.value.code == "request_rejected"
    assert raised.value.reason == "invalid_conditioning"

    output_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await output_task


@pytest.mark.asyncio
async def test_reset_fails_an_outstanding_external_advance(
    fake_api: pipeline_module._EngineAPI,
) -> None:
    pipe = ExternallyPacedDemoPipeline(FakeEngine())
    pipe.load(None)
    pipe._on_loop_ready()
    broadcasts: list[ModelMessage] = []
    pipe.bind_output(
        broadcast=broadcasts.append,
        addressed=lambda conn_id, message, request_id: None,
        media=lambda chunk: None,
    )
    await pipe._dispatch_reactor_event(SessionStarted("session"))
    session_id = pipe._engine_session_id
    assert session_id is not None

    advance_task = asyncio.create_task(pipe.advance_engine())
    request = await _wait_for_advance(pipe)
    pipe.reset_engine()

    with pytest.raises(RealtimeStepError, match="engine session was reset") as raised:
        await advance_task
    assert raised.value.code == "session_reset"

    output_task = asyncio.create_task(pipe._output_loop())
    await pipe._engine_outbox.queue.put(
        FakeChunk(session_id, DemoMessage(value=9), request_id=request.request_id)
    )
    await asyncio.sleep(0)
    assert broadcasts == []
    assert not output_task.done()
    output_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await output_task


@pytest.mark.asyncio
async def test_run_stops_engine_cleanly_when_cancelled(pipe: DemoPipeline) -> None:
    task = asyncio.create_task(pipe.run())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert pipe._engine_stop is not None
    assert pipe._engine_stop.is_set()


def test_numpy_conditioning_comparison() -> None:
    first = {"image": np.array([1.0, np.nan])}
    second = {"image": np.array([1.0, np.nan])}
    changed = {"image": np.array([2.0, np.nan])}

    assert pipeline_module._values_equal(first, second)
    assert not pipeline_module._values_equal(first, changed)


async def _wait_for_actions(pipe: DemoPipeline, count: int) -> None:
    async with asyncio.timeout(0.2):
        while len(_actions(pipe)) < count:
            pipe._engine_inbox.changed.clear()
            await pipe._engine_inbox.changed.wait()


async def _wait_for_advance(pipe: RealtimePipeline) -> FakeAdvance:
    async with asyncio.timeout(0.2):
        while True:
            for message in pipe._engine_inbox.messages:
                if isinstance(message, FakeAdvance):
                    return message
            pipe._engine_inbox.changed.clear()
            await pipe._engine_inbox.changed.wait()


def _actions(pipe: DemoPipeline) -> list[FakeAction]:
    return [message for message in pipe._engine_inbox.messages if isinstance(message, FakeAction)]
