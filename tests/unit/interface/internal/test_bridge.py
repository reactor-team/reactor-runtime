import asyncio
import time
from typing import Any

import numpy as np
import pytest

from reactor_runtime import (
    InputField,
    MediaInput,
    ModelMessage,
    Output,
    ReactorApp,
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
from reactor_runtime.interface.internal.reactor_core import (
    AddressedSink,
    BroadcastSink,
    CommandEnvelope,
    MediaOps,
    MediaSink,
    RequestId,
    StepSink,
)
from reactor_runtime.interface.model.contract import ModelContract


class Reply(ModelMessage):
    text: str


class Out(Output):
    main: Video


class In(MediaInput):
    camera: Video


class EchoModel(ReactorApp):
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


def test_check_command_validates_without_admitting(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge, model = make_bridge()
    admitted: list[object] = []
    monkeypatch.setattr(model, "submit_command", lambda *args: admitted.append(args))

    valid = bridge.check_command("set_prompt", {"prompt": "hi"})
    refused = bridge.check_command("set_prompt", {"prompt": ""})

    assert valid.accepted is True
    assert (refused.accepted, refused.field) == (False, "prompt")
    assert admitted == []


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
    # Stamped with the session the model was in, so a late chunk of an earlier
    # session can be told apart downstream.
    assert chunk.session == model._sessions_started


def test_bound_media_ops_reach_the_model_output_handle() -> None:
    bridge, model = make_bridge()
    calls: list[str] = []
    bridge.bind_outbound(
        broadcast=lambda msg: None,
        addressed=lambda conn, msg, req: None,
        media=lambda chunk: None,
        media_ops=MediaOps(
            flush=lambda hold: calls.append(f"flush:{hold}"),
            set_rate=lambda fps: calls.append(f"rate:{fps}"),
            set_depth=lambda depth: calls.append(f"depth:{depth}"),
        ),
    )
    model.output.flush()
    model.output.flush(hold=True)
    model.output.fps = 24
    assert calls == ["flush:False", "flush:True", "rate:24.0"]


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


def test_a_model_overriding_bind_output_with_its_old_signature_still_binds() -> None:
    class OwnBind(EchoModel):
        def bind_output(
            self,
            *,
            broadcast: BroadcastSink,
            addressed: AddressedSink,
            media: MediaSink,
            media_ops: MediaOps | None = None,
            step: StepSink | None = None,
        ) -> None:
            super().bind_output(
                broadcast=broadcast,
                addressed=addressed,
                media=media,
                media_ops=media_ops,
                step=step,
            )

    model = OwnBind()
    bridge = ModelBridge(model, ModelContract.of(OwnBind))
    timing = lambda conn_id, session, times: None  # noqa: E731

    bridge.bind_outbound(
        broadcast=lambda m: None,
        addressed=lambda *a: None,
        media=lambda c: None,
        command_timing=timing,
    )

    assert model._out_command_timing is timing


def test_a_model_that_overrides_submit_command_still_gets_its_commands() -> None:
    class OwnSubmit(EchoModel):
        def __init__(self) -> None:
            super().__init__()
            self.submitted: list[tuple[str, ConnId | None, RequestId | None]] = []

        def submit_command(
            self, command: Any, conn_id: ConnId | None, request_id: RequestId | None
        ) -> None:
            self.submitted.append((type(command).__name__, conn_id, request_id))

    model = OwnSubmit()
    bridge = ModelBridge(model, ModelContract.of(OwnSubmit))

    outcome = asyncio.run(
        bridge.submit_command(
            "set_prompt", {"prompt": "hi"}, conn_id=ConnId(1), request_id="r1", received_at=1.0
        )
    )

    assert outcome.accepted is True
    assert [(conn, request) for _, conn, request in model.submitted] == [(ConnId(1), "r1")]


def test_a_command_reaches_the_model_with_when_it_arrived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, model = make_bridge()
    queued: list[CommandEnvelope] = []
    monkeypatch.setattr(model, "_enqueue", queued.append)

    asyncio.run(
        bridge.submit_command(
            "set_prompt", {"prompt": "hi"}, conn_id=ConnId(1), request_id="r1", received_at=1.5
        )
    )

    [envelope] = queued
    assert envelope.received_at == 1.5
