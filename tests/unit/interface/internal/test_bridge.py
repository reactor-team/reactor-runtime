import asyncio
import time

import numpy as np
import pytest

from reactor_runtime import (
    Input,
    InputField,
    ModelMessage,
    Output,
    ReactorModel,
    Video,
    event,
    session_started,
)
from reactor_runtime.core import SessionStarted
from reactor_runtime.core.values import (
    CommandFailure,
    ConnId,
    InputFrame,
    MediaChunk,
)
from reactor_runtime.interface.internal.bridge import ModelBridge
from reactor_runtime.interface.internal.reactor_core import MediaOps, RequestId
from reactor_runtime.interface.model.contract import ModelContract


class Reply(ModelMessage):
    text: str


class Out(Output):
    main: Video


class In(Input):
    camera: Video


class EchoModel(ReactorModel):
    input: In

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []
        self.sessions_started: list[str] = []

    @event(name="set_prompt")
    async def set_prompt(self, prompt: str = InputField(min_length=1)) -> None:
        self.prompts.append(prompt)

    @session_started
    async def _on_session_started(self) -> None:
        self.sessions_started.append("started")

    async def run(self) -> None:
        await asyncio.sleep(60)


class Sinks:
    def __init__(self) -> None:
        self.broadcast: list[ModelMessage] = []
        self.addressed: list[
            tuple[ConnId, ModelMessage | CommandFailure | None, RequestId | None]
        ] = []
        self.media: list[MediaChunk] = []

    def bind(self, bridge: ModelBridge) -> None:
        bridge.bind_outbound(
            broadcast=self.broadcast.append, addressed=self._addr, media=self.media.append
        )

    def _addr(
        self,
        conn: ConnId,
        message: ModelMessage | CommandFailure | None,
        request_id: RequestId | None,
    ) -> None:
        self.addressed.append((conn, message, request_id))


def make_bridge() -> tuple[ModelBridge, EchoModel]:
    model = EchoModel()
    return ModelBridge(model, ModelContract.of(EchoModel)), model


def frame() -> InputFrame:
    return InputFrame(data=np.zeros((2, 2, 3), dtype=np.uint8), pts=0.0)


# --- validation at the inbound command face ------------------------------


def test_submit_command_accepts_a_valid_payload() -> None:
    bridge, _ = make_bridge()
    outcome = asyncio.run(
        bridge.submit_command("set_prompt", {"prompt": "hi"}, conn_id=None, request_id="r1")
    )
    assert outcome.accepted is True


def test_submit_command_rejects_a_constraint_violation() -> None:
    bridge, _ = make_bridge()
    outcome = asyncio.run(
        bridge.submit_command("set_prompt", {"prompt": ""}, conn_id=None, request_id="r1")
    )
    assert outcome.accepted is False
    assert outcome.field == "prompt"
    assert outcome.reason is not None


def test_submit_command_rejects_an_unknown_command() -> None:
    bridge, _ = make_bridge()
    outcome = asyncio.run(bridge.submit_command("nope", {}, conn_id=None, request_id="r1"))
    assert outcome.accepted is False
    assert outcome.field == "nope"


# --- routing across the three inbound faces ------------------------------


def test_a_valid_command_is_dispatched_to_its_handler() -> None:
    bridge, model = make_bridge()
    Sinks().bind(bridge)
    bridge.start()
    try:
        time.sleep(0.1)
        asyncio.run(
            bridge.submit_command(
                "set_prompt", {"prompt": "hi"}, conn_id=ConnId(1), request_id="r1"
            )
        )
        time.sleep(0.1)
        assert model.prompts == ["hi"]
    finally:
        asyncio.run(bridge.stop())
        time.sleep(0.1)


def test_a_reactor_event_is_routed_unvalidated() -> None:
    bridge, model = make_bridge()
    Sinks().bind(bridge)
    bridge.start()
    try:
        time.sleep(0.1)
        bridge.dispatch_reactor_event(SessionStarted(session_id="s-1"))
        time.sleep(0.1)
        assert model.sessions_started == ["started"]
    finally:
        asyncio.run(bridge.stop())
        time.sleep(0.1)


def test_push_media_routes_to_the_track_buffer() -> None:
    bridge, model = make_bridge()
    bridge.push_media("camera", frame())
    assert model._input_buffers["camera"].available == 1


# --- outbound binding ----------------------------------------------------


def test_bind_outbound_wires_the_broadcast_sink() -> None:
    bridge, model = make_bridge()
    sinks = Sinks()
    sinks.bind(bridge)
    asyncio.run(model.send(Reply(text="x")))
    assert sinks.broadcast == [Reply(text="x")]


def test_emit_reaches_the_media_sink_as_a_chunk() -> None:
    bridge, model = make_bridge()
    sinks = Sinks()
    sinks.bind(bridge)

    asyncio.run(model.emit(Out(main=np.full((2, 2, 3), 1, np.uint8)), compute_time=0.05))
    assert len(sinks.media) == 1
    chunk = sinks.media[0]
    assert chunk.n_frames == 1
    assert chunk.fps == 20.0  # one frame in 0.05s
    assert chunk.bundle.tracks["main"].data.shape == (2, 2, 3)


def test_bound_media_ops_reach_the_model_output_handle() -> None:
    bridge, model = make_bridge()
    calls: list[str] = []
    bridge.bind_outbound(
        broadcast=lambda msg: None,
        addressed=lambda conn, msg, req: None,
        media=lambda chunk: None,
        media_ops=MediaOps(
            flush=lambda: calls.append("flush"),
            set_rate=lambda fps: calls.append(f"rate:{fps}"),
            set_depth=lambda depth: calls.append(f"depth:{depth}"),
        ),
    )
    model.output.flush()
    model.output.fps = 24
    assert calls == ["flush", "rate:24.0"]


# --- lifecycle + surface -------------------------------------------------


def test_start_requires_outbound_to_be_bound() -> None:
    bridge, _ = make_bridge()
    with pytest.raises(RuntimeError, match="bind_outbound"):
        bridge.start()


def test_bind_outbound_rejects_a_second_call() -> None:
    bridge, _ = make_bridge()
    Sinks().bind(bridge)
    with pytest.raises(RuntimeError, match="once"):
        Sinks().bind(bridge)


def test_contract_is_exposed() -> None:
    bridge, _ = make_bridge()
    assert bridge.contract is ModelContract.of(EchoModel)
