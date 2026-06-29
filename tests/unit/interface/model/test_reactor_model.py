import asyncio
import time
from typing import Any

from reactor_runtime import (
    ClientInfo,
    ModelMessage,
    Output,
    ReactorModel,
    Video,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.core.model import (
    ClientConnected,
    ClientDisconnected,
    Command,
    EndReason,
    SessionEnded,
    SessionStarted,
)
from reactor_runtime.core.values import ConnId
from reactor_runtime.interface.internal.reactor_core import CommandEnvelope, RequestId

Addressed = list[tuple[ConnId, ModelMessage, RequestId | None]]


class Out(Output):
    main: Video


class BrightnessSet(ModelMessage):
    value: int


class Model(ReactorModel):
    output: Out

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Any, ...]] = []
        self.value: int | None = None

    @event(name="set_value")
    async def set_value(self, value: int = 0) -> BrightnessSet:
        self.value = value
        return BrightnessSet(value=value)

    @event(name="touch")
    async def touch(self, client: ClientInfo) -> None:
        self.calls.append(("touch", client.id))

    @event(name="boom")
    async def boom(self) -> None:
        raise RuntimeError("kaboom")

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        self.calls.append(("connected", client.id))

    @disconnected
    def on_disconnect(self) -> None:
        self.calls.append(("disconnected",))

    @session_started
    def on_start(self) -> None:
        self.calls.append(("session_started",))

    @session_ended
    async def on_end(self) -> None:
        self.calls.append(("session_ended",))

    async def run(self) -> None:
        await asyncio.Event().wait()


def _ready(model: Model) -> Addressed:
    """Bring a model's loop-bound state up and capture its addressed sends."""
    addressed: Addressed = []
    model._on_loop_ready()
    model.bind_output(
        broadcast=lambda message: None,
        addressed=lambda conn_id, message, request_id: addressed.append(
            (conn_id, message, request_id)
        ),
    )
    return addressed


def _cmd(name: str, **args: Any) -> Command:
    """Validate raw args into the typed command the contract resolves."""
    return Model.__reactor_contract__.validate(name, args)


async def test_command_invokes_handler_and_replies_to_sender() -> None:
    model = Model()
    addressed = _ready(model)
    envelope = CommandEnvelope(_cmd("set_value", value=7), ConnId(1001), "req-1")
    await model._dispatch_command(envelope)
    assert model.value == 7
    assert addressed == [(ConnId(1001), BrightnessSet(value=7), "req-1")]


async def test_command_returning_none_sends_nothing() -> None:
    model = Model()
    addressed = _ready(model)
    await model._dispatch_command(CommandEnvelope(_cmd("touch"), ConnId(1001), "req-2"))
    assert model.calls == [("touch", ConnId(1001))]
    assert addressed == []


async def test_reserved_client_is_the_sending_connection() -> None:
    model = Model()
    _ready(model)
    await model._dispatch_command(CommandEnvelope(_cmd("touch"), ConnId(1234), "req-3"))
    assert model.calls == [("touch", ConnId(1234))]


async def test_handler_exception_is_swallowed() -> None:
    model = Model()
    addressed = _ready(model)
    await model._dispatch_command(CommandEnvelope(_cmd("boom"), ConnId(1001), "req-4"))
    assert addressed == []


async def test_return_value_is_dropped_without_an_originating_connection() -> None:
    model = Model()
    addressed = _ready(model)
    await model._dispatch_command(CommandEnvelope(_cmd("set_value", value=3), None, "req-5"))
    assert model.value == 3
    assert addressed == []


async def test_client_connected_sets_the_event_and_fires_the_hook() -> None:
    model = Model()
    _ready(model)
    await model._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    assert model.connected.is_set()
    assert ("connected", ConnId(1001)) in model.calls
    assert ConnId(1001) in model._clients


async def test_client_disconnected_clears_the_event_and_forgets_the_client() -> None:
    model = Model()
    _ready(model)
    await model._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    await model._dispatch_reactor_event(ClientDisconnected(ConnId(1001), 0))
    assert not model.connected.is_set()
    assert ("disconnected",) in model.calls
    assert ConnId(1001) not in model._clients


async def test_session_started_and_ended_hooks_fire() -> None:
    model = Model()
    _ready(model)
    await model._dispatch_reactor_event(SessionStarted("s-1"))
    await model._dispatch_reactor_event(SessionEnded("s-1", EndReason.STOPPED))
    assert ("session_started",) in model.calls
    assert ("session_ended",) in model.calls


async def test_session_end_clears_occupancy_without_a_per_client_close() -> None:
    model = Model()
    _ready(model)
    await model._dispatch_reactor_event(ClientConnected(ConnId(1001), 1))
    assert model.connected.is_set()
    await model._dispatch_reactor_event(SessionEnded("s-1", EndReason.STOPPED))
    assert not model.connected.is_set()
    assert model._clients == {}


async def test_client_send_routes_addressed_without_a_request_id() -> None:
    model = Model()
    addressed = _ready(model)
    client = model._client_for(ConnId(1001))
    assert client is not None
    await client.send(BrightnessSet(value=9))
    assert addressed == [(ConnId(1001), BrightnessSet(value=9), None)]


def test_loops_drain_the_queues_on_the_model_thread() -> None:
    model = Model()
    addressed: Addressed = []
    model.bind_output(
        broadcast=lambda message: None,
        addressed=lambda conn_id, message, request_id: addressed.append(
            (conn_id, message, request_id)
        ),
    )
    model.start_thread()
    try:
        time.sleep(0.15)  # let the loop bootstrap its queues and dispatchers
        model.post_reactor_event(ClientConnected(ConnId(1001), 1))
        model.submit_command(_cmd("set_value", value=42), ConnId(1001), "req-9")
        time.sleep(0.25)
        assert model.value == 42
        assert (ConnId(1001), BrightnessSet(value=42), "req-9") in addressed
        assert model.connected.is_set()
    finally:
        model.stop()
        time.sleep(0.1)
