from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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
from reactor_runtime.realtime import RealtimePipeline
from reactor_runtime.realtime import pipeline as pipeline_module


@dataclass
class FakeAttach:
    session_id: str
    context: dict[str, Any] = field(default_factory=dict)


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
class FakeSetActive:
    session_id: str
    active: bool


@dataclass
class FakeChunk:
    session_id: str
    output: Any


class FakeInbox:
    def __init__(self) -> None:
        self.messages: list[Any] = []
        self.changed = asyncio.Event()

    def put_nowait(self, message: Any) -> None:
        self.messages.append(message)
        self.changed.set()


class FakeOutbox:
    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = maxsize
        self.queue: asyncio.Queue[FakeChunk] = asyncio.Queue(maxsize=maxsize)

    async def get(self) -> FakeChunk:
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
        attach=FakeAttach,
        detach=FakeDetach,
        inbox=FakeInbox,
        outbox=FakeOutbox,
        reset=FakeReset,
        set_active=FakeSetActive,
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
        FakeAttach(engine_session_id, {"prompt": "start"}),
        FakeSetActive(engine_session_id, False),
    ]

    await pipe._dispatch_reactor_event(ClientConnected(ConnId(1), 1))
    pipe.state.prompt = "changed"
    await pipe._dispatch_reactor_event(ClientDisconnected(ConnId(1), 0))
    assert pipe.state.prompt == "changed"
    await pipe._dispatch_reactor_event(ClientConnected(ConnId(2), 1))
    assert pipe._engine_session_id == engine_session_id

    await pipe._dispatch_reactor_event(SessionEnded("session", EndReason.STOPPED))
    assert pipe._engine_inbox.messages[2:] == [
        FakeSetActive(engine_session_id, True),
        FakeSetActive(engine_session_id, False),
        FakeSetActive(engine_session_id, True),
        FakeSetActive(engine_session_id, False),
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


def _actions(pipe: DemoPipeline) -> list[FakeAction]:
    return [message for message in pipe._engine_inbox.messages if isinstance(message, FakeAction)]
