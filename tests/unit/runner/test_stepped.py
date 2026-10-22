"""The stepped path on its own: driven through a fake host, with no runner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

from reactor_runtime import protocol
from reactor_runtime.core import ConnId, StepRefusal, StepRefusalKind
from reactor_runtime.interface.internal.bridge import CommandOutcome
from reactor_runtime.message_gateway import InboundStep
from reactor_runtime.runner.session_start import Generation
from reactor_runtime.runner.stepped import StepHost, StepOrder, StepRequests, refused
from reactor_wire.v1 import data_pb2, step_pb2

V1 = protocol.ProtocolVersion.V1
CLIENT = ConnId(1002)


class FakeConnections:
    """Records each step reply, decoded as the client reads it."""

    def __init__(self) -> None:
        self.replies: list[data_pb2.DataServerMessage] = []

    def send_step_reply(
        self, cid: ConnId, encode: Callable[[protocol.ProtocolVersion], tuple[Any, bytes | str]]
    ) -> None:
        _, frame = encode(V1)
        message = protocol.select(V1).decode(
            frame, protocol.Channel.DATA, protocol.Direction.SERVER
        )
        assert isinstance(message, data_pb2.DataServerMessage)
        self.replies.append(message)


def _requests(
    *, running: bool = True, generation: Generation = Generation.STEPPED, sessions: int = 1
) -> tuple[StepRequests, FakeConnections, list[tuple[Any, dict[str, Any]]]]:
    connections = FakeConnections()
    journal: list[tuple[Any, dict[str, Any]]] = []

    def unused(*_: Any) -> Any:
        raise AssertionError("a refused request must not reach this")

    host = StepHost(
        # These requests are refused before the bridge is used; any object stands in.
        bridge=lambda: cast(Any, object()),
        session_running=lambda: running,
        generation=lambda: generation,
        sessions_posted=lambda: sessions,
        starting_input_done=asyncio.Event,
        connections=cast(Any, connections),
        journal=lambda event, **detail: journal.append((event, detail)),
        resolve_uploads=unused,
        command_metrics=cast(Any, None),
        command_label=lambda name: name,
        codec_for=protocol.select,
        spawn=unused,
    )
    return StepRequests(host, upload_wait=0.0), connections, journal


def _step(request_id: str = "r1") -> InboundStep:
    return InboundStep(
        state={},
        commands=(),
        media_input=(),
        conn_id=CLIENT,
        request_id=request_id,
        received_at=0.0,
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
    steps, connections, journal = _requests(running=False)

    await steps.submit(_step())

    [reply] = connections.replies
    assert reply.request_id == "r1"
    assert reply.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_NOT_READY
    assert reply.step_result.step == 0
    assert journal == []


async def test_a_request_in_a_streamed_session_is_refused() -> None:
    steps, connections, _ = _requests(generation=Generation.STREAMED)

    await steps.submit(_step())

    [reply] = connections.replies
    assert reply.step_result.failure_code == step_pb2.STEP_FAILURE_CODE_REFUSED
    assert "not stepped" in reply.step_result.error


def test_a_refusal_carries_its_code_and_entry() -> None:
    steps, connections, _ = _requests()

    steps.refuse(
        CLIENT,
        "r1",
        "notes are closed",
        failure_code=step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED,
        code="no_notes",
        entry="commands[1]",
    )

    result = connections.replies[0].step_result
    assert (result.failure_code, result.error, result.code, result.entry) == (
        step_pb2.STEP_FAILURE_CODE_COMMAND_FAILED,
        "notes are closed",
        "no_notes",
        "commands[1]",
    )


def test_a_model_refusal_from_an_earlier_session_is_answered_as_ended() -> None:
    steps, connections, _ = _requests(sessions=2)

    steps.on_refusal(StepRefusal(CLIENT, "r1", session=1, reason="not now"))

    assert connections.replies[0].step_result.failure_code == (
        step_pb2.STEP_FAILURE_CODE_SESSION_ENDED
    )


def test_a_model_refusal_keeps_its_kind_as_its_wire_code() -> None:
    steps, connections, _ = _requests(sessions=1)

    steps.on_refusal(
        StepRefusal(CLIENT, "r1", session=1, reason="not ready", kind=StepRefusalKind.NOT_READY)
    )

    assert connections.replies[0].step_result.failure_code == (step_pb2.STEP_FAILURE_CODE_NOT_READY)


def test_a_contract_refusal_names_the_entry_and_the_field() -> None:
    outcome = CommandOutcome.reject("prompt", "must not be empty")

    assert refused("state.prompt", outcome) == (
        "state.prompt is refused: prompt: must not be empty"
    )
