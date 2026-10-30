"""The client against a real worker: results, failures, and a lost link that stays lost."""

from __future__ import annotations

import asyncio
import logging
import pickle
import queue
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from reactor_runtime import log
from reactor_runtime.distributed.errors import WorkerTimeout
from reactor_runtime.ipc import SharedSlotAllocationFailed
from reactor_runtime.model_server import (
    BatchFailed,
    ClientClosed,
    LinkConfig,
    LinkState,
    ModelClient,
    ModelServer,
    PayloadTooLarge,
    ResetRequired,
    ServerRequest,
    ServerSettings,
    StepStillRunning,
    StepTimedOut,
    StreamLink,
    WorkerLost,
    WorkerRejected,
    WorkerServer,
    WorkerUnreachable,
    link_config,
)
from reactor_runtime.model_server.link import NoListener
from reactor_runtime.model_server.protocol import Cancel, Result, Step


@dataclass(frozen=True)
class Tick:
    value: int


@dataclass(frozen=True)
class Frame:
    value: int
    generation: int
    session_id: str | None


@dataclass(frozen=True)
class Stop:
    reason: str


class NoFaceError(ValueError):
    pass


class Model(ModelServer[Tick, Frame]):
    def __init__(self) -> None:
        self.hold = threading.Event()
        self.hold.set()
        self.entered = threading.Event()
        self.signals: list[Any] = []

    def signal(self, slot: int, signal: Any) -> None:
        self.signals.append(signal)

    def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
        self.entered.set()
        self.hold.wait(10)
        out: list[Any] = []
        for r in requests:
            if r.payload.value < 0:
                out.append(NoFaceError("no face"))
            elif r.payload.value == 13:
                raise RuntimeError("the batch broke")
            else:
                out.append(Frame(r.payload.value, r.generation, r.session_id))
        return out


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
    """An end that cannot carry a request for 1000 or a result for 2000."""

    def send(self, message: Any) -> None:
        if isinstance(message, Step) and message.payload.value == 1000:
            raise PayloadTooLarge("the request does not fit")
        if isinstance(message, Result) and message.output.value == 2000:
            raise PayloadTooLarge("the result does not fit")
        super().send(message)


class PairTransport:
    """Links over in-process socket pairs; refuses to connect until a server listens."""

    def __init__(self, link: type[StreamLink] = StreamLink) -> None:
        self.listener: _Listener | None = None
        self.link = link

    def listen(self, address: str) -> _Listener:
        self.listener = _Listener()
        return self.listener

    def connect(self, address: str, slot: int, *, timeout: float | None = None) -> StreamLink:
        if self.listener is None:
            raise NoListener("nothing listens yet")
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.pending.put(self.link(b))
        return self.link(a)


@dataclass
class Rig:
    model: Model
    server: WorkerServer
    transport: PairTransport

    def client(self, slot: int = 0, *, step_timeout: float | None = 5.0) -> ModelClient:
        config = LinkConfig(
            socket="unused", slot=slot, step_timeout=step_timeout, connect_timeout=5
        )
        return ModelClient.connect(config, transport=self.transport)


def _server(model: Model, transport: PairTransport) -> WorkerServer:
    settings = ServerSettings(max_batch=4, batch_window=0.0, shm_input_bytes=1, shm_output_bytes=1)
    server = WorkerServer(model, transport, "unused", settings=settings, max_sessions=2)
    asyncio.run(server.start())
    return server


@pytest.fixture
def rig() -> Iterator[Rig]:
    model, transport = Model(), PairTransport()
    rig = Rig(model, _server(model, transport), transport)
    yield rig
    model.hold.set()
    asyncio.run(rig.server.drain())
    asyncio.run(rig.server.stop())


def test_a_step_returns_the_result_after_the_application_resets(rig: Rig) -> None:
    client = rig.client()
    with pytest.raises(ResetRequired):
        client.step(Tick(1))

    generation = client.reset(session_id="s-1")

    assert generation == client.generation == 2
    assert client.step(Tick(4)) == Frame(4, generation=2, session_id="s-1")


def test_a_connected_session_runtime_stamps_its_role_and_slot(rig: Rig) -> None:
    client = rig.client(slot=1)
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "line", None, None)
    log.SessionContextFilter().filter(record)

    fields = getattr(record, "reactor_fields", {})
    assert (fields.get("runtime_role"), fields.get("slot")) == ("session", 1)
    client.close()


def test_a_reset_names_the_live_session_by_default(rig: Rig) -> None:
    client = rig.client()
    log.set_session_id("s-live")
    try:
        client.reset()
    finally:
        log.clear_session_id()
    assert client.step(Tick(1)).session_id == "s-live"


def test_the_models_own_exception_and_a_failed_batch_reach_the_caller(rig: Rig) -> None:
    client = rig.client()
    client.reset()

    with pytest.raises(NoFaceError, match="no face"):
        client.step(Tick(-1))
    with pytest.raises(BatchFailed, match="the batch broke"):
        client.step(Tick(13))
    assert client.step(Tick(2)).value == 2  # the client goes on


def test_a_signal_reaches_the_model_as_sent(rig: Rig) -> None:
    client = rig.client()
    client.reset()

    client.signal(Stop("barge-in"))
    deadline = time.monotonic() + 5
    while not rig.model.signals and time.monotonic() < deadline:
        time.sleep(0.01)
    assert rig.model.signals == [Stop("barge-in")]


def test_a_step_with_no_answer_in_time_times_out_and_the_client_goes_on(rig: Rig) -> None:
    client = rig.client()
    client.reset()
    rig.model.hold.clear()

    with pytest.raises(StepTimedOut):
        client.step(Tick(1), timeout=0.05)
    rig.model.hold.set()

    # The abandoned step still runs on the model server, and a session has one
    # request at a time, so the next one is refused until it ends.
    deadline = time.monotonic() + 5
    while True:
        try:
            assert client.step(Tick(2)).value == 2
            break
        except StepStillRunning:
            assert time.monotonic() < deadline
            time.sleep(0.01)


def test_a_lost_link_stays_lost_for_every_call(rig: Rig) -> None:
    client = rig.client()
    client.reset()
    asyncio.run(rig.server.drain())
    asyncio.run(rig.server.stop())

    deadline = time.monotonic() + 5
    while client.state is LinkState.OPEN and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.state is LinkState.LOST
    for call in (lambda: client.step(Tick(1)), client.reset, lambda: client.signal(Stop("x"))):
        with pytest.raises(WorkerLost, match="worker_died"):
            call()


def _connect_watching(rig: Rig, on_lost: Callable[[WorkerLost], None]) -> ModelClient:
    config = LinkConfig(socket="unused", slot=0, step_timeout=5.0, connect_timeout=5)
    return ModelClient.connect(config, on_lost=on_lost, transport=rig.transport)


def test_a_link_lost_between_steps_reaches_on_lost_once(rig: Rig) -> None:
    lost: list[WorkerLost] = []
    heard = threading.Event()

    def on_lost(error: WorkerLost) -> None:
        lost.append(error)
        heard.set()

    client = _connect_watching(rig, on_lost)
    asyncio.run(rig.server.drain())
    asyncio.run(rig.server.stop())

    assert heard.wait(5)
    with pytest.raises(WorkerLost) as raised:
        client.reset()
    assert [(type(e), str(e)) for e in lost] == [(type(raised.value), str(raised.value))]
    client.close()
    assert len(lost) == 1


def test_a_payload_too_large_for_the_link_fails_that_step_and_the_client_goes_on() -> None:
    model, transport = Model(), PairTransport(link=SmallLink)
    server = _server(model, transport)
    lost: list[WorkerLost] = []
    config = LinkConfig(socket="unused", slot=0, step_timeout=5.0, connect_timeout=5)
    client = ModelClient.connect(config, on_lost=lost.append, transport=transport)
    try:
        client.reset()
        with pytest.raises(PayloadTooLarge, match="the request does not fit"):
            client.step(Tick(1000))
        with pytest.raises(PayloadTooLarge, match="the result does not fit"):
            client.step(Tick(2000))

        assert client.step(Tick(3)).value == 3
        assert client.state is LinkState.OPEN
        assert lost == []
    finally:
        client.close()
        asyncio.run(server.drain())
        asyncio.run(server.stop())


def test_close_never_reaches_on_lost(rig: Rig) -> None:
    lost: list[WorkerLost] = []
    client = _connect_watching(rig, lost.append)

    client.close()
    time.sleep(0.2)  # the reader sees its link closed

    assert client.state is LinkState.CLOSED
    assert lost == []


def test_an_on_lost_that_raises_leaves_the_client_lost(rig: Rig) -> None:
    def on_lost(error: WorkerLost) -> None:
        raise RuntimeError("the callback broke")

    client = _connect_watching(rig, on_lost)
    asyncio.run(rig.server.drain())
    asyncio.run(rig.server.stop())

    deadline = time.monotonic() + 5
    while client.state is LinkState.OPEN and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.state is LinkState.LOST
    with pytest.raises(WorkerLost):
        client.reset()


def test_close_makes_every_call_raise_client_closed(rig: Rig) -> None:
    client = rig.client()
    client.close()
    client.close()  # idempotent

    assert client.state is LinkState.CLOSED
    with pytest.raises(ClientClosed):
        client.reset()


def test_connect_waits_for_a_model_server_that_starts_later() -> None:
    model, transport = Model(), PairTransport()
    started: list[WorkerServer] = []
    threading.Timer(0.3, lambda: started.append(_server(model, transport))).start()

    config = LinkConfig(socket="unused", slot=1, connect_timeout=10)
    client = ModelClient.connect(config, transport=transport)
    try:
        assert client.slot == 1
    finally:
        client.close()
        asyncio.run(started[0].drain())
        asyncio.run(started[0].stop())


def test_connect_gives_up_after_its_timeout_naming_why() -> None:
    config = LinkConfig(socket="unused", slot=0, connect_timeout=0.3)
    with pytest.raises(WorkerTimeout, match="nothing listens yet"):
        ModelClient.connect(config, transport=PairTransport())


def test_connect_gives_up_on_a_model_server_that_never_opens_the_session() -> None:
    class Silent(PairTransport):
        def connect(self, address: str, slot: int, *, timeout: float | None = None) -> StreamLink:
            a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            self.kept = b  # accepted, never answered
            return StreamLink(a)

    config = LinkConfig(socket="unused", slot=0, connect_timeout=0.3)
    started = time.monotonic()
    with pytest.raises(WorkerTimeout, match="did not open the session"):
        ModelClient.connect(config, transport=Silent())
    assert time.monotonic() - started < 5


def test_a_refused_session_and_an_unreachable_deployment_raise_at_once(rig: Rig) -> None:
    with pytest.raises(WorkerRejected, match="out of range"):
        rig.client(slot=5)

    class Unreachable(PairTransport):
        def connect(self, address: str, slot: int, *, timeout: float | None = None) -> StreamLink:
            raise WorkerUnreachable("no socket directory")

    config = LinkConfig(socket="unused", slot=0, connect_timeout=30)
    started = time.monotonic()
    with pytest.raises(WorkerUnreachable):
        ModelClient.connect(config, transport=Unreachable())
    assert time.monotonic() - started < 1


# -- the configuration ----------------------------------------------------------------------


def test_link_config_reads_the_environment_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACTOR_MODEL_WORKER_SOCKET", "/run/reactor-worker/sock")
    monkeypatch.setenv("REACTOR_SESSION_SLOT", "2")
    monkeypatch.delenv("REACTOR_SHM_DIR", raising=False)
    monkeypatch.setenv("REACTOR_WORKER_STEP_TIMEOUT", "0")

    config = link_config()

    assert config == LinkConfig(
        socket="/run/reactor-worker/sock", slot=2, shm_dir=None, step_timeout=None
    )
    monkeypatch.setenv("REACTOR_SHM_DIR", "/dev/shm/s2")
    assert link_config().shm_dir == Path("/dev/shm/s2")


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"REACTOR_SESSION_SLOT": "0"}, "REACTOR_MODEL_WORKER_SOCKET is not set"),
        ({"REACTOR_MODEL_WORKER_SOCKET": "/s"}, "REACTOR_SESSION_SLOT is not set"),
        ({"REACTOR_MODEL_WORKER_SOCKET": "/s", "REACTOR_SESSION_SLOT": "-1"}, "non-negative"),
        (
            {
                "REACTOR_MODEL_WORKER_SOCKET": "/s",
                "REACTOR_SESSION_SLOT": "0",
                "REACTOR_WORKER_CONNECT_TIMEOUT": "soon",
            },
            "number of seconds",
        ),
    ],
)
def test_link_config_refuses_a_missing_or_malformed_variable(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], match: str
) -> None:
    for name in (
        "REACTOR_MODEL_WORKER_SOCKET",
        "REACTOR_SESSION_SLOT",
        "REACTOR_WORKER_CONNECT_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=match):
        link_config()


# -- the second review round ----------------------------------------------------------------


def test_a_request_that_does_not_pickle_is_raised_and_the_client_stays_open(rig: Rig) -> None:
    lost: list[WorkerLost] = []
    client = _connect_watching(rig, lost.append)
    client.reset()

    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        client.step(Tick(lambda: None))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # a request that does not pickle

    assert client.state is LinkState.OPEN
    assert lost == []
    assert client.step(Tick(4)).value == 4
    client.close()


def test_every_call_after_a_loss_raises_a_fresh_exception(rig: Rig) -> None:
    client = rig.client()
    asyncio.run(rig.server.drain())
    asyncio.run(rig.server.stop())
    deadline = time.monotonic() + 5
    while client.state is LinkState.OPEN and time.monotonic() < deadline:
        time.sleep(0.01)

    raised = []
    for _ in range(1000):
        with pytest.raises(WorkerLost) as error:
            client.reset()
        raised.append(error.value)

    assert len({id(e) for e in raised}) == 1000
    assert {str(e) for e in raised} == {str(raised[0])}
    assert all(e.__context__ is None and e.__cause__ is None for e in raised)


def test_connect_timeout_bounds_a_transport_whose_setup_stalls() -> None:
    class Stalling(PairTransport):
        def connect(self, address: str, slot: int, *, timeout: float | None = None) -> StreamLink:
            assert timeout is not None
            time.sleep(timeout)  # a setup that never completes within the time it was given
            raise NoListener("the model server did not finish setting up the link in time")

    config = LinkConfig(socket="unused", slot=0, connect_timeout=0.5)
    started = time.monotonic()
    with pytest.raises(WorkerTimeout, match="did not finish setting up"):
        ModelClient.connect(config, transport=Stalling())
    assert time.monotonic() - started < 3


def test_a_connect_with_no_time_left_times_out_naming_the_last_reason() -> None:
    calls: list[float | None] = []

    class Absent(PairTransport):
        def connect(self, address: str, slot: int, *, timeout: float | None = None) -> StreamLink:
            calls.append(timeout)
            raise NoListener("nothing accepts there")

    config = LinkConfig(socket="unused", slot=0, connect_timeout=0.2)
    with pytest.raises(WorkerTimeout, match="nothing accepts there"):
        ModelClient.connect(config, transport=Absent())
    assert calls
    assert all(t is not None and t > 0 for t in calls)  # never a non-blocking 0


def test_a_step_abandoned_while_running_makes_the_next_one_still_running(rig: Rig) -> None:
    client = rig.client()
    client.reset()
    rig.model.hold.clear()
    outcome: list[BaseException] = []

    def abandoned() -> None:
        try:
            client.step(Tick(1), timeout=1.0)
        except BaseException as exc:
            outcome.append(exc)

    first = threading.Thread(target=abandoned)
    first.start()
    assert rig.model.entered.wait(5)  # running before the client gives up on it
    first.join(timeout=5)
    assert len(outcome) == 1
    assert isinstance(outcome[0], StepTimedOut)

    with pytest.raises(StepStillRunning, match="retry once it ends"):
        client.step(Tick(2))
    rig.model.hold.set()


def test_a_cancel_that_cannot_be_sent_still_raises_step_timed_out(rig: Rig) -> None:
    client = rig.client()
    client.reset()
    rig.model.hold.clear()
    real_send = client._link.send

    def no_room_for_cancels(message: Any) -> None:
        if isinstance(message, Cancel):
            raise SharedSlotAllocationFailed("no room for the replacement block")
        real_send(message)

    client._link.send = no_room_for_cancels  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]  # a cancel that fails
    with pytest.raises(StepTimedOut):
        client.step(Tick(1), timeout=0.05)
    assert client.state is LinkState.OPEN
    rig.model.hold.set()


def test_a_timeout_of_zero_waits_without_a_bound_and_a_negative_one_is_refused(
    rig: Rig,
) -> None:
    client = rig.client(step_timeout=0.05)
    client.reset()
    rig.model.hold.clear()
    threading.Timer(0.3, rig.model.hold.set).start()

    assert client.step(Tick(5), timeout=0).value == 5  # longer than the configured 0.05 s
    with pytest.raises(ValueError, match="at least 0"):
        client.step(Tick(6), timeout=-1)
    with pytest.raises(ValueError, match="at least 0"):
        client.reset(timeout=-1)
