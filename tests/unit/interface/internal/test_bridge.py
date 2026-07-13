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
    ConnId,
    InputFrame,
    MediaBundle,
    TrackData,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.interface.internal.bridge import ModelBridge
from reactor_runtime.interface.internal.reactor_core import RequestId
from reactor_runtime.interface.model.contract import ModelContract


class Reply(ModelMessage):
    text: str


class Out(Output):
    main: Video


class In(Input):
    camera: Video


class EchoModel(ReactorModel):
    output: Out
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
        self.addressed: list[tuple[ConnId, ModelMessage | None, RequestId | None]] = []
        self.media: list[tuple[MediaBundle, bool]] = []

    def bind(self, bridge: ModelBridge) -> None:
        bridge.bind_outbound(
            broadcast=self.broadcast.append, addressed=self._addr, media=self._media
        )

    def _addr(
        self, conn: ConnId, message: ModelMessage | None, request_id: RequestId | None
    ) -> None:
        self.addressed.append((conn, message, request_id))

    def _media(self, bundle: MediaBundle, is_fresh_black: bool) -> None:
        self.media.append((bundle, is_fresh_black))


def make_bridge() -> tuple[ModelBridge, EchoModel]:
    model = EchoModel()
    return ModelBridge(model, ModelContract.of(EchoModel)), model


def video_bundle(value: int = 0) -> MediaBundle:
    info = TrackInfo(name="main", kind=TrackKind.VIDEO, direction=TrackDirection.OUT)
    data = np.full((2, 2, 3), value, np.uint8)
    return MediaBundle(tracks={"main": TrackData(info=info, data=data)})


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


def test_emit_reaches_the_media_sink_with_the_fresh_black_flag() -> None:
    bridge, model = make_bridge()
    sinks = Sinks()
    sinks.bind(bridge)

    model.output_buffer.submit(video_bundle(1), drop=True)
    model.output_buffer._emit_one_tick()
    assert sinks.media[-1][1] is False  # a real frame

    model.output_buffer.flush()
    model.output_buffer._emit_one_tick()
    assert sinks.media[-1][1] is True  # the session-boundary black


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


def test_contract_and_output_buffer_are_exposed() -> None:
    bridge, model = make_bridge()
    assert bridge.contract is ModelContract.of(EchoModel)
    assert bridge.output_buffer is model.output_buffer
