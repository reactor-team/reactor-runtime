"""The model server's core, end to end over in-process links: sessions, batches and faults."""

from __future__ import annotations

import asyncio
import collections
import queue
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from reactor_runtime.core import Health, HealthStatus
from reactor_runtime.model_server import (
    BatchFailed,
    ModelServer,
    PayloadTooLarge,
    ResetRequired,
    ServerRequest,
    ServerSettings,
    StepExpired,
    StreamLink,
    WorkerRejected,
    WorkerStopping,
)
from reactor_runtime.model_server import worker as worker_module
from reactor_runtime.model_server.batching import Waiting
from reactor_runtime.model_server.connection import (
    Connection,
    ConnectionMoveError,
    ConnectionState,
)
from reactor_runtime.model_server.protocol import (
    PROTOCOL,
    Cancel,
    Closed,
    Failed,
    Hello,
    Refused,
    Reset,
    ResetDone,
    Result,
    Signal,
    Step,
    Welcome,
    error_of,
)
from reactor_runtime.model_server.worker import ServerMoveError, ServerState, WorkerServer


@dataclass(frozen=True)
class Tick:
    value: int


@dataclass(frozen=True)
class Echo:
    value: int
    slot: int
    generation: int
    session_id: str | None


@dataclass(frozen=True)
class Stop:
    reason: str


@dataclass(frozen=True)
class Other:
    reason: str


class EchoModel(ModelServer[Tick, Echo]):
    max_batch = 4

    def __init__(self) -> None:
        self.batches: list[list[int]] = []
        self.events: list[tuple[str, int, int]] = []
        self.signals: list[tuple[int, Any]] = []
        self.hold = threading.Event()
        self.hold.set()
        self.stepping = threading.Event()
        self.raise_batch = False
        self.refuse_open = False
        self.health_error = False

    def session_opened(self, slot: int, generation: int) -> None:
        if self.refuse_open:
            raise ValueError("no room")
        self.events.append(("opened", slot, generation))

    def session_closed(self, slot: int, generation: int) -> None:
        self.events.append(("closed", slot, generation))

    def signal(self, slot: int, signal: Any) -> None:
        self.signals.append((slot, signal))

    def health(self) -> Health:
        if self.health_error:
            raise RuntimeError("probe failed")
        return Health.healthy()

    def step(self, requests: list[ServerRequest[Tick]]) -> list[Echo | Exception]:
        self.batches.append([r.payload.value for r in requests])
        self.stepping.set()
        self.hold.wait(10)
        if self.raise_batch:
            raise ValueError("boom")
        return [
            ValueError("negative")
            if r.payload.value < 0
            else Echo(r.payload.value, r.slot, r.generation, r.session_id)
            for r in requests
        ]


class _Listener:
    def __init__(self) -> None:
        self.pending: queue.Queue[StreamLink | None] = queue.Queue()

    def accept(self) -> StreamLink:
        link = self.pending.get()
        if link is None:
            raise OSError("closed")
        return link

    def close(self) -> None:
        self.pending.put(None)


class SmallLink(StreamLink):
    """A model server end that cannot carry a result for a tick of 1000 or more."""

    def send(self, message: Any) -> None:
        if isinstance(message, Result) and message.output.value >= 1000:
            raise PayloadTooLarge(f"a result for tick {message.output.value} does not fit")
        super().send(message)


class PairTransport:
    """Links over in-process socket pairs, every array inline."""

    def __init__(self, server_link: type[StreamLink] = StreamLink) -> None:
        self.listener = _Listener()
        self.server_link = server_link

    def listen(self, address: str) -> _Listener:
        return self.listener

    def connect(self, address: str, slot: int, *, timeout: float | None = None) -> StreamLink:
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.pending.put(self.server_link(b))
        return StreamLink(a)


def _wait(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("the condition did not hold in time")
        time.sleep(0.005)


@dataclass
class Rig:
    model: EchoModel
    server: WorkerServer
    transport: PairTransport

    def session(self, slot: int = 0) -> StreamLink:
        link = self.transport.connect("unused", slot)
        link.send(Hello(slot=slot))
        assert isinstance(link.receive(), Welcome)
        return link


def _rig(
    *,
    max_batch: int = 4,
    batch_window: float = 0.0,
    stuck_after: float = 60.0,
    transport: PairTransport | None = None,
    model: EchoModel | None = None,
) -> Rig:
    model = model or EchoModel()
    transport = transport or PairTransport()
    settings = ServerSettings(max_batch, batch_window, 1 << 20, 1 << 20)
    server = WorkerServer(
        model, transport, "unused", settings=settings, max_sessions=4, stuck_after=stuck_after
    )
    asyncio.run(server.start())
    return Rig(model, server, transport)


@pytest.fixture
def rig() -> Iterator[Rig]:
    rig = _rig()
    yield rig
    rig.model.hold.set()
    asyncio.run(rig.server.drain())
    asyncio.run(rig.server.stop())


# -- sessions -------------------------------------------------------------------------------


def test_a_session_is_welcomed_with_its_slot_and_first_generation(rig: Rig) -> None:
    link = rig.transport.connect("unused", 2)
    link.send(Hello(slot=2))

    assert link.receive() == Welcome(slot=2, generation=1, max_batch=4)
    assert rig.model.events == [("opened", 2, 1)]


@pytest.mark.parametrize(
    ("hello", "match"),
    [
        (Hello(slot=4), "slot 4 is out of range"),
        (Hello(slot=0, protocol=PROTOCOL + 1), "the worker speaks"),
    ],
)
def test_a_hello_the_worker_cannot_serve_is_refused(rig: Rig, hello: Hello, match: str) -> None:
    link = rig.transport.connect("unused", 0)
    link.send(hello)

    answer = link.receive()

    assert isinstance(answer, Refused)
    assert match in answer.reason
    with pytest.raises(Closed):
        link.receive()


def test_a_session_that_session_opened_refuses_is_told_why(rig: Rig) -> None:
    rig.model.refuse_open = True
    link = rig.transport.connect("unused", 0)
    link.send(Hello(slot=0))

    answer = link.receive()

    assert isinstance(answer, Refused)
    assert "no room" in answer.reason


def test_a_session_that_leaves_has_its_state_closed(rig: Rig) -> None:
    link = rig.session(1)
    link.close()

    _wait(lambda: ("closed", 1, 1) in rig.model.events)


def test_a_new_hello_for_a_held_slot_replaces_the_old_connection(rig: Rig) -> None:
    old = rig.session(0)
    new = rig.transport.connect("unused", 0)
    new.send(Hello(slot=0))

    assert new.receive() == Welcome(slot=0, generation=2, max_batch=4)
    assert rig.model.events == [("opened", 0, 1), ("closed", 0, 1), ("opened", 0, 2)]
    with pytest.raises(Closed):
        old.receive()


def test_a_reset_closes_and_reopens_the_slot_with_the_session_id(rig: Rig) -> None:
    link = rig.session(0)
    link.send(Reset(request_id=1, session_id="s-abc"))
    assert link.receive() == ResetDone(request_id=1, generation=2)

    link.send(Step(request_id=2, timeout=None, payload=Tick(5)))

    assert link.receive() == Result(2, Echo(5, slot=0, generation=2, session_id="s-abc"))
    assert rig.model.events == [("opened", 0, 1), ("closed", 0, 1), ("opened", 0, 2)]


# -- steps ----------------------------------------------------------------------------------


def test_a_request_is_stepped_and_answered(rig: Rig) -> None:
    link = rig.session(1)
    link.send(Step(request_id=7, timeout=None, payload=Tick(3)))

    assert link.receive() == Result(7, Echo(3, slot=1, generation=1, session_id=None))


def test_an_output_that_is_an_exception_fails_that_request_alone(rig: Rig) -> None:
    link = rig.session(0)
    link.send(Step(request_id=1, timeout=None, payload=Tick(-1)))

    answer = link.receive()

    assert isinstance(answer, Failed)
    error = error_of(answer)
    assert type(error) is ValueError
    assert str(error) == "negative"


def test_a_result_too_large_for_the_link_fails_that_request_alone() -> None:
    rig = _rig(transport=PairTransport(server_link=SmallLink))
    link = rig.session(0)
    try:
        link.send(Step(request_id=1, timeout=None, payload=Tick(1000)))
        error = error_of(link.receive())
        assert isinstance(error, PayloadTooLarge)
        assert str(error) == "a result for tick 1000 does not fit"

        link.send(Step(request_id=2, timeout=None, payload=Tick(1)))
        assert link.receive() == Result(2, Echo(1, 0, 1, None))
    finally:
        asyncio.run(rig.server.drain())
        asyncio.run(rig.server.stop())


def test_a_raising_step_fails_every_request_of_its_batch_once() -> None:
    rig = _rig(batch_window=1.0)
    rig.model.raise_batch = True
    a, b = rig.session(0), rig.session(1)
    try:
        a.send(Step(request_id=1, timeout=None, payload=Tick(1)))
        b.send(Step(request_id=1, timeout=None, payload=Tick(2)))

        errors = [error_of(a.receive()), error_of(b.receive())]

        assert all(isinstance(e, BatchFailed) for e in errors)
        assert all(type(e.cause) is ValueError for e in errors if isinstance(e, BatchFailed))
        assert [sorted(batch) for batch in rig.model.batches] == [[1, 2]]  # one batch, once
    finally:
        asyncio.run(rig.server.drain())
        asyncio.run(rig.server.stop())


def test_a_second_request_while_one_runs_is_refused(rig: Rig) -> None:
    rig.model.hold.clear()
    link = rig.session(0)
    link.send(Step(request_id=1, timeout=None, payload=Tick(1)))
    assert rig.model.stepping.wait(5)

    link.send(Step(request_id=2, timeout=None, payload=Tick(2)))
    refused = link.receive()
    rig.model.hold.set()

    assert isinstance(error_of(refused), WorkerRejected)
    assert refused.request_id == 2
    assert link.receive() == Result(1, Echo(1, 0, 1, None))


def test_a_request_whose_timeout_passes_in_the_queue_expires(rig: Rig) -> None:
    rig.model.hold.clear()
    busy, late = rig.session(0), rig.session(1)
    busy.send(Step(request_id=1, timeout=None, payload=Tick(1)))
    assert rig.model.stepping.wait(5)
    late.send(Step(request_id=1, timeout=0.01, payload=Tick(2)))
    time.sleep(0.05)
    rig.model.hold.set()

    assert isinstance(error_of(late.receive()), StepExpired)
    assert rig.model.batches == [[1]]


def test_a_cancelled_request_is_never_stepped(rig: Rig) -> None:
    rig.model.hold.clear()
    busy, cancelled = rig.session(0), rig.session(1)
    busy.send(Step(request_id=1, timeout=None, payload=Tick(1)))
    assert rig.model.stepping.wait(5)
    cancelled.send(Step(request_id=1, timeout=None, payload=Tick(2)))
    cancelled.send(Cancel(request_id=1))
    time.sleep(0.05)
    rig.model.hold.set()
    busy.receive()

    cancelled.send(Step(request_id=2, timeout=None, payload=Tick(3)))
    assert cancelled.receive() == Result(2, Echo(3, 1, 1, None))
    assert rig.model.batches == [[1], [3]]


# -- signals --------------------------------------------------------------------------------


def test_every_signal_reaches_the_model_as_sent(rig: Rig) -> None:
    link = rig.session(3)
    link.send(Signal(Stop("barge-in")))
    link.send(Signal(Other("anything")))

    _wait(lambda: rig.model.signals == [(3, Stop("barge-in")), (3, Other("anything"))])


# -- the component --------------------------------------------------------------------------


def test_drain_answers_what_still_waits_as_stopping() -> None:
    rig = _rig()
    rig.model.hold.clear()
    busy, waiting = rig.session(0), rig.session(1)
    busy.send(Step(request_id=1, timeout=None, payload=Tick(1)))
    assert rig.model.stepping.wait(5)
    waiting.send(Step(request_id=1, timeout=None, payload=Tick(2)))
    time.sleep(0.05)

    drained = threading.Thread(target=lambda: asyncio.run(rig.server.drain()))
    drained.start()
    _wait(lambda: rig.server.state is ServerState.DRAINING)  # no new batch is taken
    rig.model.hold.set()
    drained.join(timeout=5)
    asyncio.run(rig.server.stop())

    assert busy.receive() == Result(1, Echo(1, 0, 1, None))
    assert isinstance(error_of(waiting.receive()), WorkerStopping)
    assert ("closed", 0, 1) in rig.model.events


def test_health_reports_a_stuck_step_and_a_raising_probe() -> None:
    rig = _rig(stuck_after=0.05)
    try:
        assert rig.server.health().status is HealthStatus.HEALTHY
        rig.model.hold.clear()
        link = rig.session(0)
        link.send(Step(request_id=1, timeout=None, payload=Tick(1)))
        assert rig.model.stepping.wait(5)
        _wait(lambda: rig.server.health().status is HealthStatus.UNHEALTHY)
        assert "stuck" in (rig.server.health().detail or "")
        rig.model.hold.set()
        link.receive()

        rig.model.health_error = True
        report = rig.server.health()
        assert report.status is HealthStatus.UNHEALTHY
        assert "probe failed" in (report.detail or "")
    finally:
        rig.model.hold.set()
        asyncio.run(rig.server.drain())
        asyncio.run(rig.server.stop())


# -- the connection -------------------------------------------------------------------------


def test_a_connection_refuses_a_move_its_sequence_does_not_allow() -> None:
    a, b = socket.socketpair()
    connection = Connection(StreamLink(a), 1)
    try:
        with pytest.raises(ConnectionMoveError, match="from connecting to open"):
            connection.move(ConnectionState.OPEN)
        connection.move(ConnectionState.OPENING)
        connection.move(ConnectionState.OPEN)
        connection.move(ConnectionState.CLOSING)
        connection.move(ConnectionState.CLOSED)
        with pytest.raises(ConnectionMoveError):
            connection.move(ConnectionState.OPEN)
    finally:
        connection.close()
        b.close()


def test_a_full_outbox_is_reported_instead_of_blocking() -> None:
    a, b = socket.socketpair()
    hold = threading.Event()

    class Stuck(StreamLink):
        def send(self, message: Any) -> None:
            hold.wait(5)

    connection = Connection(Stuck(a), 1, outbox=2)
    try:
        posted = [connection.post(Result(n, None)) for n in range(4)]
        assert posted[-1] is False
    finally:
        hold.set()
        connection.close()
        b.close()


# -- the second review round: lifecycles, ordering, and what is in flight --------------------


def test_a_reset_whose_reopen_raises_refuses_and_closes_the_session(rig: Rig) -> None:
    link = rig.session(0)
    rig.model.refuse_open = True
    link.send(Reset(request_id=1, session_id="s-next"))

    refused = link.receive()

    assert isinstance(error_of(refused), WorkerRejected)
    with pytest.raises(Closed):
        link.receive()
    assert rig.model.events == [("opened", 0, 1), ("closed", 0, 1)]


def test_a_step_and_a_reset_after_drain_are_answered_as_stopping() -> None:
    rig = _rig()
    link = rig.session(0)
    asyncio.run(rig.server.drain())
    try:
        link.send(Step(request_id=1, timeout=None, payload=Tick(1)))
        assert isinstance(error_of(link.receive()), WorkerStopping)
        link.send(Reset(request_id=2, session_id=None))
        assert isinstance(error_of(link.receive()), WorkerStopping)
    finally:
        asyncio.run(rig.server.stop())


def test_the_server_moves_only_forward() -> None:
    rig = _rig()
    asyncio.run(rig.server.drain())
    assert rig.server.state is ServerState.DRAINING
    asyncio.run(rig.server.stop())
    assert rig.server.state is ServerState.STOPPED
    with pytest.raises(ServerMoveError, match="from stopped to serving"):
        rig.server._move(ServerState.SERVING)


def test_a_step_sent_while_its_reset_is_pending_is_refused_in_order(rig: Rig) -> None:
    rig.model.hold.clear()
    busy, session = rig.session(0), rig.session(1)
    busy.send(Step(request_id=1, timeout=None, payload=Tick(1)))
    assert rig.model.stepping.wait(5)  # the step thread is busy, so the reset waits

    session.send(Reset(request_id=1, session_id="s-new"))
    session.send(Step(request_id=2, timeout=None, payload=Tick(2)))
    early = session.receive()
    rig.model.hold.set()

    assert isinstance(error_of(early), ResetRequired)
    assert session.receive() == ResetDone(request_id=1, generation=2)
    session.send(Step(request_id=3, timeout=None, payload=Tick(3)))
    assert session.receive() == Result(3, Echo(3, slot=1, generation=2, session_id="s-new"))


def test_a_connection_that_sends_no_hello_is_closed_after_the_timeout(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker_module, "HELLO_TIMEOUT", 0.2)
    silent = rig.transport.connect("unused", 0)

    started = time.monotonic()
    with pytest.raises(Closed):
        silent.receive()
    assert time.monotonic() - started < 5


def test_stop_closes_a_connection_that_never_sent_its_hello() -> None:
    rig = _rig()
    silent = rig.transport.connect("unused", 0)
    time.sleep(0.05)  # accepted, waiting for its hello

    asyncio.run(rig.server.drain())
    asyncio.run(rig.server.stop())

    with pytest.raises(Closed):
        silent.receive()


def test_at_most_max_sessions_connections_open_at_once(rig: Rig) -> None:
    silent = [rig.transport.connect("unused", slot) for slot in range(4)]
    time.sleep(0.1)  # all four accepted and waiting for their hellos

    extra = rig.transport.connect("unused", 0)
    with pytest.raises(Closed):
        extra.receive()
    for link in silent:
        link.close()


def test_drain_closes_a_pending_open_and_answers_a_pending_reset_as_stopping() -> None:
    rig = _rig()
    rig.model.hold.clear()
    busy, resetting = rig.session(0), rig.session(1)
    busy.send(Step(request_id=1, timeout=None, payload=Tick(1)))
    assert rig.model.stepping.wait(5)
    opening = rig.transport.connect("unused", 2)
    opening.send(Hello(slot=2))
    resetting.send(Reset(request_id=1, session_id=None))
    time.sleep(0.05)  # both queued for the busy step thread

    drained = threading.Thread(target=lambda: asyncio.run(rig.server.drain()))
    drained.start()
    _wait(lambda: rig.server.state is ServerState.DRAINING)
    rig.model.hold.set()
    drained.join(timeout=5)
    try:
        assert isinstance(error_of(resetting.receive()), WorkerStopping)
        with pytest.raises(Closed):  # closed, not refused: the session waits for the next one
            opening.receive()
    finally:
        asyncio.run(rig.server.stop())


def _opening(server: WorkerServer) -> int:
    with server._cond:
        return len(server._opening)


class SlowOpenModel(EchoModel):
    """A model whose session_opened() waits until the test lets it go on."""

    def __init__(self) -> None:
        super().__init__()
        self.opening = threading.Event()
        self.open_hold = threading.Event()

    def session_opened(self, slot: int, generation: int) -> None:
        self.opening.set()
        self.open_hold.wait(10)
        super().session_opened(slot, generation)


def test_a_slow_session_opened_still_counts_toward_the_opening_bound() -> None:
    model = SlowOpenModel()
    rig = _rig(model=model)
    try:
        slow = rig.transport.connect("unused", 0)
        slow.send(Hello(slot=0))
        assert model.opening.wait(5)
        silent = [rig.transport.connect("unused", slot) for slot in range(1, 4)]
        _wait(lambda: _opening(rig.server) == 4)  # the slow open and three silent ones

        extra = rig.transport.connect("unused", 0)
        started = time.monotonic()
        with pytest.raises(Closed):
            extra.receive()
        assert time.monotonic() - started < 2  # refused at once, not by its hello timeout
        for link in silent:
            link.close()
    finally:
        model.open_hold.set()
        asyncio.run(rig.server.drain())
        asyncio.run(rig.server.stop())


def test_a_session_opened_while_the_server_drains_is_closed_not_welcomed() -> None:
    model = SlowOpenModel()
    rig = _rig(model=model)
    link = rig.transport.connect("unused", 0)
    link.send(Hello(slot=0))
    assert model.opening.wait(5)

    drained = threading.Thread(target=lambda: asyncio.run(rig.server.drain()))
    drained.start()
    _wait(lambda: rig.server.state is ServerState.DRAINING)
    model.open_hold.set()
    drained.join(timeout=5)
    assert not drained.is_alive()
    try:
        with pytest.raises(Closed):  # closed, not welcomed: the session waits for the next one
            link.receive()
        assert model.events == [("opened", 0, 1), ("closed", 0, 1)]
    finally:
        asyncio.run(rig.server.stop())


class SlowRefusingModel(SlowOpenModel):
    """A model whose slow session_opened() then refuses the session."""

    def session_opened(self, slot: int, generation: int) -> None:
        self.opening.set()
        self.open_hold.wait(10)
        raise ValueError("no room")


def test_a_session_opened_refused_while_the_server_drains_is_closed_not_refused() -> None:
    model = SlowRefusingModel()
    rig = _rig(model=model)
    link = rig.transport.connect("unused", 0)
    link.send(Hello(slot=0))
    assert model.opening.wait(5)

    drained = threading.Thread(target=lambda: asyncio.run(rig.server.drain()))
    drained.start()
    _wait(lambda: rig.server.state is ServerState.DRAINING)
    model.open_hold.set()
    drained.join(timeout=5)
    assert not drained.is_alive()
    try:
        with pytest.raises(Closed):  # no Refused: the session waits for the next one
            link.receive()
    finally:
        asyncio.run(rig.server.stop())


def test_a_batch_whose_sessions_all_left_never_reaches_step(rig: Rig) -> None:
    gone = Waiting(connection=999, request_id=1, payload=Tick(1), arrived=0.0, deadline=None)

    rig.server._run([gone])

    assert rig.model.batches == []


class SequenceModel(EchoModel):
    """Answers with a sequence that is neither a list nor a tuple, or with a string."""

    def step(self, requests: list[ServerRequest[Tick]]) -> Any:
        outputs = super().step(requests)
        if requests[0].payload.value == 7:
            return "not outputs"
        return collections.UserList(outputs)


def test_any_sequence_of_outputs_is_answered_and_a_string_fails_its_batch() -> None:
    rig = _rig(model=SequenceModel())
    link = rig.session(0)
    try:
        link.send(Step(request_id=1, timeout=None, payload=Tick(1)))
        assert link.receive() == Result(1, Echo(1, slot=0, generation=1, session_id=None))
        link.send(Step(request_id=2, timeout=None, payload=Tick(7)))
        error = error_of(link.receive())
        assert isinstance(error, BatchFailed)
        assert "returned str outputs" in str(error)
    finally:
        asyncio.run(rig.server.drain())
        asyncio.run(rig.server.stop())


class UnpicklableModel(EchoModel):
    """Answers tick 666 with a result that does not pickle."""

    def step(self, requests: list[ServerRequest[Tick]]) -> Any:
        if requests[0].payload.value == 666:
            return [Echo(lambda: None, 0, 0, None)]  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # a result that does not pickle
        return super().step(requests)


def test_a_result_that_does_not_pickle_fails_that_request_alone() -> None:
    rig = _rig(model=UnpicklableModel())
    link = rig.session(0)
    try:
        link.send(Step(request_id=1, timeout=None, payload=Tick(666)))
        assert isinstance(link.receive(), Failed)
        link.send(Step(request_id=2, timeout=None, payload=Tick(2)))
        assert link.receive() == Result(2, Echo(2, slot=0, generation=1, session_id=None))
    finally:
        asyncio.run(rig.server.drain())
        asyncio.run(rig.server.stop())


def test_a_step_loop_that_stops_reports_the_worker_unhealthy(rig: Rig) -> None:
    link = rig.session(0)

    def broken(now: float, open_sessions: int) -> Any:
        raise RuntimeError("the policy broke")

    rig.server._policy.take = broken  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]  # inject a fault into the loop
    with rig.server._cond:
        rig.server._cond.notify_all()

    _wait(lambda: rig.server.health().status is HealthStatus.UNHEALTHY)
    assert "the step loop stopped: RuntimeError: the policy broke" in str(
        rig.server.health().detail
    )
    link.send(Step(request_id=1, timeout=None, payload=Tick(1)))
    assert isinstance(error_of(link.receive()), WorkerStopping)


def test_a_full_outbox_closes_the_link_at_close() -> None:
    a, b = socket.socketpair()
    hold = threading.Event()

    class Stuck(StreamLink):
        def send(self, message: Any) -> None:
            hold.wait(5)

    connection = Connection(Stuck(a), 1, outbox=1)
    try:
        connection.post(Result(1, None))  # taken by the writer, which is stuck sending it
        time.sleep(0.05)
        connection.post(Result(2, None))  # fills the outbox
        connection.close()
        b.settimeout(2)
        assert b.recv(1) == b""  # closed at once, with no room to ask for a flush
    finally:
        hold.set()
        b.close()


def test_close_with_room_delivers_every_queued_answer_first() -> None:
    a, b = socket.socketpair()
    connection = Connection(StreamLink(a), 1)
    peer = StreamLink(b)
    try:
        connection.post(Result(1, "first"))
        connection.post(Result(2, "second"))
        connection.close()
        assert [peer.receive().request_id, peer.receive().request_id] == [1, 2]
        with pytest.raises(Closed):
            peer.receive()
    finally:
        peer.close()
