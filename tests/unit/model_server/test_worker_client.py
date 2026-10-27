"""The application half's ``ModelClient``: what each call raises, and what it never does alone.

The client reports and never decides: it does not reset a slot, retry a reset,
or reconnect on its own. These tests run a real worker on a thread of the test
process and drive the client the way an application does.
"""

from __future__ import annotations

import io
import json
import logging
import os
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime import log
from reactor_runtime.model_server import (
    ClientClosed,
    ModelClient,
    ModelServer,
    ResetRequired,
    ServerRequest,
    StepExpired,
    StepTimedOut,
    StepUnanswered,
    WorkerCrashed,
    WorkerLost,
    WorkerRejected,
    WorkerTimeout,
    WorkerUnreachable,
    codec,
    wire,
)
from reactor_runtime.model_server import client as client_module
from reactor_runtime.model_server import server as server_module
from reactor_runtime.model_server.transport import UnixShmTransport
from reactor_runtime.model_server.wire import Kind


@dataclass(frozen=True)
class Preempt:
    reason: str


@dataclass(frozen=True)
class In:
    frame: np.ndarray
    chunk: int


@dataclass(frozen=True)
class Out:
    frame: np.ndarray
    chunk: int


class Gated(ModelServer):
    """Echo plain payloads, holding each step until the test opens the gate."""

    max_batch = 4

    def load(self, config_path: Path | None) -> None:
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.steps: list[list[Any]] = []
        self.events: list[tuple[str, int, int]] = []

    def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
        self.entered.set()
        self.gate.wait(10)
        self.steps.append([r.payload for r in requests])
        return [{"slot": r.slot, "gen": r.generation, "x": r.payload} for r in requests]

    def session_opened(self, slot: int, generation: int) -> None:
        self.events.append(("opened", slot, generation))

    def session_closed(self, slot: int, generation: int) -> None:
        self.events.append(("closed", slot, generation))


class TypedWorker(ModelServer):
    request_type = In
    result_type = Out
    signals = (Preempt,)

    def load(self, config_path: Path | None) -> None:
        self.received: list[tuple[int, Any]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def step(self, requests: list[ServerRequest[In]]) -> list[Out]:
        self.entered.set()
        self.gate.wait(10)
        return [Out(frame=r.payload.frame * 2, chunk=r.payload.chunk) for r in requests]

    def signal(self, slot: int, signal: Any) -> None:
        self.received.append((slot, signal))


def _typed(served: Any, slot: int, **kwargs: Any) -> ModelClient:
    return served.client(slot, request_type=In, result_type=Out, signals=(Preempt,), **kwargs)


def _until(check: Any, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


def _short_dir() -> Path:
    # A Unix socket path is limited to about 100 bytes, which pytest's tmp_path can pass.
    return Path(tempfile.mkdtemp(prefix="rrc"))


# -- reset is the application's -------------------------------------------------------------


def test_a_step_before_the_application_resets_its_slot_is_refused(serve: Any) -> None:
    worker = Gated()
    served: Any = serve(worker)
    client = served.client(0, reset=False)
    with pytest.raises(ResetRequired, match="reset"):
        client.step(1)
    assert worker.steps == []
    assert client.reset() == 2
    assert client.step(1)["gen"] == 2


def test_a_reset_that_did_not_complete_keeps_the_step_refused(serve: Any) -> None:
    worker = Gated()
    served: Any = serve(worker)
    client = served.client(0)
    holder = served.client(1)
    worker.gate.clear()
    hold = threading.Thread(target=lambda: holder.step(1))
    hold.start()
    worker.entered.wait(5)
    with pytest.raises(StepTimedOut, match="did not reset"):
        client.reset(timeout=0.1)
    with pytest.raises(ResetRequired):
        client.step(1)
    worker.gate.set()
    hold.join(5)
    # Nothing retried the reset: the application does, and the step then runs.
    generation = client.reset()
    assert client.step(1)["gen"] == generation


def test_a_refused_reset_keeps_the_step_refused(serve: Any) -> None:
    class RefusesReopen(Gated):
        def session_opened(self, slot: int, generation: int) -> None:
            super().session_opened(slot, generation)
            if generation > 1:
                raise ValueError("no room")

    worker = RefusesReopen()
    served: Any = serve(worker)
    client = served.client(0, reset=False)
    with pytest.raises(WorkerRejected, match="no room"):
        client.reset()
    with pytest.raises(ResetRequired):
        client.step(1)
    # The step thread is idle: another session is served at once.
    assert served.client(1, reset=False).generation == 1


# -- a lost worker --------------------------------------------------------------------------


def test_wait_lost_returns_why_once_the_worker_goes(serve: Any) -> None:
    served: Any = serve(Gated())
    client = served.client(0)
    with pytest.raises(TimeoutError, match=r"is up after 0.1 s"):
        client.wait_lost(timeout=0.1)
    got: list[str] = []
    waiter = threading.Thread(target=lambda: got.append(client.wait_lost(timeout=10)))
    waiter.start()
    served.halt()
    waiter.join(10)
    assert got
    assert got[0].startswith("worker_died: ")
    # Lost already: it returns at once, with the same reason.
    assert client.wait_lost(timeout=0) == got[0]
    assert client.lost_reason == got[0]
    assert not client.available


def test_wait_lost_raises_client_closed_when_the_client_closes(serve: Any) -> None:
    served: Any = serve(Gated())
    client = served.client(0)
    raised: list[BaseException] = []

    def wait() -> None:
        try:
            client.wait_lost()
        except BaseException as exc:
            raised.append(exc)

    waiter = threading.Thread(target=wait)
    waiter.start()
    time.sleep(0.1)
    client.close()
    waiter.join(5)
    assert len(raised) == 1
    assert isinstance(raised[0], ClientClosed)
    with pytest.raises(ClientClosed):
        client.wait_lost(timeout=0)


def test_every_call_raises_worker_lost_once_the_connection_is_lost(serve: Any) -> None:
    served: Any = serve(TypedWorker())
    clients = [_typed(served, slot) for slot in range(2)]
    served.halt()
    for client in clients:
        client.wait_lost(timeout=5)
        for call in (
            lambda c=client: c.step(In(frame=np.ones(1), chunk=0)),
            client.reset,
            lambda c=client: c.signal(Preempt("barge")),
        ):
            with pytest.raises(WorkerLost, match=r"^worker_died: "):
                call()
        assert client.generation == 0
        assert issubclass(WorkerLost, WorkerCrashed)


def test_every_call_raises_client_closed_once_the_client_is_closed(serve: Any) -> None:
    served: Any = serve(TypedWorker())
    client = _typed(served, 0)
    client.close()
    client.close()
    for call in (
        lambda: client.step(In(frame=np.ones(1), chunk=0)),
        client.reset,
        lambda: client.signal(Preempt("barge")),
        lambda: client.wait_lost(timeout=0),
        lambda: client.reconnect(timeout=0.1),
    ):
        with pytest.raises(ClientClosed):
            call()


def test_nothing_reconnects_until_the_application_does(serve: Any) -> None:
    worker = Gated()
    served: Any = serve(worker)
    client = served.client(4)
    served.halt()
    client.wait_lost(timeout=5)
    again: Any = serve(Gated(), path=served.path)
    time.sleep(1.0)
    assert not client.available
    with pytest.raises(WorkerLost):
        client.step(1)
    assert again.worker.events == []
    client.reconnect(timeout=5)
    assert client.available
    assert client.slot == 4
    with pytest.raises(ResetRequired):
        client.step(1)
    assert client.reset() == 2
    assert client.step(1)["slot"] == 4
    assert again.worker.events == [("opened", 4, 1), ("closed", 4, 1), ("opened", 4, 2)]


def test_reconnect_drops_a_live_connection_and_opens_a_new_session(serve: Any) -> None:
    worker = Gated()
    served: Any = serve(worker)
    client = served.client(0)
    client.reconnect(timeout=5)
    assert client.generation == 3
    with pytest.raises(ResetRequired):
        client.step(1)
    _until(lambda: ("closed", 0, 2) in worker.events)


def test_reconnect_is_bounded_by_its_timeout(serve: Any) -> None:
    served: Any = serve(Gated())
    client = served.client(0)
    served.halt()
    client.wait_lost(timeout=5)
    started = time.monotonic()
    with pytest.raises(WorkerTimeout, match=r"within 0.3 s"):
        client.reconnect(timeout=0.3)
    assert time.monotonic() - started < 2


def test_a_reconnect_blocked_on_a_missing_worker_returns_when_the_client_closes(
    serve: Any,
) -> None:
    served: Any = serve(Gated())
    client = served.client(0)
    served.halt()
    client.wait_lost(timeout=5)
    raised: list[BaseException] = []

    def reconnect() -> None:
        try:
            client.reconnect(timeout=30)
        except BaseException as exc:
            raised.append(exc)

    thread = threading.Thread(target=reconnect)
    thread.start()
    time.sleep(0.3)
    started = time.monotonic()
    client.close()
    thread.join(5)
    assert time.monotonic() - started < 3
    assert len(raised) == 1
    assert isinstance(raised[0], ClientClosed)


# -- connecting -----------------------------------------------------------------------------


def test_connect_waits_for_a_worker_that_starts_late(serve: Any) -> None:
    path = _short_dir() / "late.sock"
    connected: list[ModelClient] = []
    thread = threading.Thread(
        target=lambda: connected.append(
            ModelClient.connect(socket_path=path, slot=2, connect_timeout=10)
        )
    )
    thread.start()
    time.sleep(0.5)
    assert not connected
    served: Any = serve(Gated(), path=path)
    thread.join(10)
    client = connected[0]
    served.clients.append(client)
    client.reset()
    assert client.step(1)["slot"] == 2


def test_connect_gives_up_after_its_deadline(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(WorkerTimeout, match=r"no model worker welcomed this session within 0.3"):
        ModelClient.connect(socket_path=tmp_path / "none.sock", slot=0, connect_timeout=0.3)
    assert time.monotonic() - started < 2


def test_a_connect_waiting_for_a_worker_returns_when_the_client_closes() -> None:
    path = _short_dir() / "never.sock"
    raised: list[BaseException] = []

    def connect() -> None:
        try:
            ModelClient.connect(socket_path=path, slot=0, connect_timeout=30)
        except BaseException as exc:
            raised.append(exc)

    thread = threading.Thread(target=connect)
    thread.start()
    _until(lambda: [c for c in list(client_module._OPEN) if c._path == str(path)])
    waiting = next(c for c in list(client_module._OPEN) if c._path == str(path))
    started = time.monotonic()
    # What the interpreter's exit does on SIGTERM, before it joins pool threads.
    waiting.close()
    thread.join(5)
    assert time.monotonic() - started < 3
    assert len(raised) == 1
    assert isinstance(raised[0], ClientClosed)


def test_a_missing_socket_directory_fails_at_once(tmp_path: Path) -> None:
    missing = tmp_path / "not-mounted" / "sock"
    started = time.monotonic()
    with pytest.raises(WorkerUnreachable, match=r"socket directory .* does not exist"):
        ModelClient.connect(socket_path=missing, slot=0, connect_timeout=30)
    assert time.monotonic() - started < 2


def test_no_socket_named_fails_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REACTOR_MODEL_WORKER_SOCKET", raising=False)
    with pytest.raises(WorkerUnreachable, match="REACTOR_MODEL_WORKER_SOCKET is not set"):
        ModelClient.connect()


def test_shared_memory_the_session_cannot_see_fails_the_connect(serve: Any) -> None:
    served: Any = serve(Gated())
    transport = UnixShmTransport()
    broken = {"request_prefix": "rrx_", "probe": "rrnosuchprobe", "probe_nonce": "00"}
    with pytest.raises(WorkerUnreachable, match="must share /dev/shm"):
        transport.client_carrier(broken)
    wrong = {**broken, "probe": f"{served.prefix}probe", "probe_nonce": "ffffffffffffffff"}
    with pytest.raises(WorkerUnreachable, match="do not share one /dev/shm"):
        transport.client_carrier(wrong)

    class Blind(UnixShmTransport):
        def client_carrier(self, welcome: Any) -> Any:
            raise WorkerUnreachable("the model worker's shared memory is not visible here")

    started = time.monotonic()
    with pytest.raises(WorkerUnreachable, match="not visible here"):
        ModelClient.connect(socket_path=served.path, slot=0, transport=Blind(), connect_timeout=30)
    assert time.monotonic() - started < 3


def test_a_session_whose_types_differ_from_the_workers_is_refused(serve: Any) -> None:
    served: Any = serve(TypedWorker())
    with pytest.raises(WorkerRejected) as raised:
        served.client(0, result_type=Out, reset=False)
    message = str(raised.value)
    assert served.server.schema in message
    assert "differ from the worker's" in message
    assert "this session declares: request: plain" in message
    # The signal order is part of the contract too.
    with pytest.raises(WorkerRejected, match="differ"):
        served.client(0, request_type=In, result_type=Out, reset=False)
    assert _typed(served, 0).available


def test_a_session_of_another_protocol_is_refused_with_both_versions(serve: Any) -> None:
    served: Any = serve(Gated())
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    try:
        hello = {"slot": 0, "pid": os.getpid(), "token": "abcd", "protocol": 1, "schema": "x"}
        wire.send(sock, Kind.HELLO, 0, json.dumps(hello).encode())
        kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
    finally:
        sock.close()
    assert kind is Kind.REFUSED
    reason = wire.decode_message(payload, wire.Refusal).reason
    assert "protocol 1" in reason
    assert f"protocol {wire.PROTOCOL}" in reason


def test_a_declared_type_that_is_not_data_only_fails_the_connect(serve: Any) -> None:
    @dataclass
    class Loose:
        x: int

    served: Any = serve(Gated())
    with pytest.raises(TypeError, match=r"result_type: .*is not frozen"):
        served.client(0, result_type=Loose)
    with pytest.raises(TypeError, match="signals must be a tuple"):
        served.client(0, signals=[Preempt])


# -- answers that did not come --------------------------------------------------------------


def test_a_late_answer_times_the_step_out_and_the_client_goes_on(serve: Any) -> None:
    worker = Gated()
    served: Any = serve(worker)
    client = served.client(0)
    worker.gate.clear()
    with pytest.raises(StepTimedOut, match=r"no answer within 0.2") as raised:
        client.step(np.ones(2), timeout=0.2)
    assert isinstance(raised.value, StepUnanswered)
    assert isinstance(raised.value, WorkerTimeout)
    worker.gate.set()
    np.testing.assert_array_equal(client.step(np.full(2, 5.0))["x"], np.full(2, 5.0))


def test_a_request_that_expired_in_the_queue_is_a_step_expired(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = server_module.WorkerServer._take_request

    def short_deadline(self: Any, conn: Any, request_id: int, payload: bytes) -> None:
        # The worker sees a deadline far shorter than the client waits for.
        original(self, conn, request_id, wire.TIMEOUT.pack(0.05) + payload[wire.TIMEOUT.size :])

    monkeypatch.setattr(server_module.WorkerServer, "_take_request", short_deadline)
    worker = Gated()
    served: Any = serve(worker)
    holder, late = served.client(0), served.client(1)
    worker.gate.clear()
    hold = threading.Thread(target=lambda: holder.step(1, timeout=10))
    hold.start()
    worker.entered.wait(5)
    raised: list[BaseException] = []

    def step() -> None:
        try:
            late.step(2, timeout=10)
        except BaseException as exc:
            raised.append(exc)

    waiting = threading.Thread(target=step)
    waiting.start()
    time.sleep(0.2)
    worker.gate.set()
    hold.join(5)
    waiting.join(5)
    assert len(raised) == 1
    assert isinstance(raised[0], StepExpired), raised
    assert isinstance(raised[0], StepUnanswered)
    # Its payload never reached step().
    assert worker.steps == [[1]]
    assert late.step(3)["x"] == 3


def test_a_request_superseded_by_the_same_session_is_answered_as_expired(serve: Any) -> None:
    worker = Gated()
    served: Any = serve(worker)
    holder = served.client(1)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    hello = {
        "slot": 0,
        "pid": os.getpid(),
        "token": "abcd1234",
        "protocol": wire.PROTOCOL,
        "schema": served.server.schema,
    }
    wire.send(sock, Kind.HELLO, 0, json.dumps(hello).encode())
    assert wire.recv(sock, wire.CONTROL_LIMIT)[0] is Kind.WELCOME
    worker.gate.clear()
    hold = threading.Thread(target=lambda: holder.step(0))
    hold.start()
    worker.entered.wait(5)
    try:
        for request_id in (1, 2):
            header = json.dumps({"v": 1, "block": "x", "spans": [], "tree": request_id}).encode()
            wire.send(sock, Kind.STEP, request_id, wire.TIMEOUT.pack(10.0) + header)
        assert wire.recv(sock, 1 << 20)[:2] == (Kind.EXPIRED, 1)
        worker.gate.set()
        assert wire.recv(sock, 1 << 20)[:2] == (Kind.RESULT, 2)
    finally:
        worker.gate.set()
        hold.join(5)
        sock.close()


def test_after_a_timeout_the_input_block_is_used_again(serve: Any) -> None:
    worker = Gated()
    served: Any = serve(worker)
    client = served.client(0)
    worker.gate.clear()
    with pytest.raises(StepTimedOut):
        client.step(np.ones(1024), timeout=0.1)
    worker.gate.set()
    client.step(np.ones(1024))
    blocks = {p.name for p in Path("/dev/shm").glob(f"{served.prefix}*")}
    for _ in range(5):
        client.step(np.ones(1024))
    after = {p.name for p in Path("/dev/shm").glob(f"{served.prefix}*")}
    assert after <= blocks
    assert not [n for n in after if "_g" in n]


def test_a_rejected_request_does_not_free_an_abandoned_ones_block(serve: Any) -> None:
    class Slow(ModelServer):
        max_request_bytes = 1 << 16

        def load(self, config_path: Any) -> None:
            self.seen: list[int] = []

        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
            request = requests[0]
            if request.payload.get("slow"):
                time.sleep(1.0)
            self.seen.append(int(request.payload["a"][0]))
            return [0]

    served: Any = serve(Slow())
    client = served.client(0)
    with pytest.raises(StepTimedOut):
        client.step({"a": np.full(1000, 1, np.uint8), "slow": True}, timeout=0.2)
    # Its buffers fit the limit; with its header it does not.
    with pytest.raises(WorkerRejected):
        client.step({"a": np.full((1 << 16) - 4, 9, np.uint8)}, timeout=5)
    client.step({"a": np.full(1000, 2, np.uint8)}, timeout=5)
    assert served.worker.seen[0] == 1


def test_an_oversize_request_leaves_no_block_behind(serve: Any) -> None:
    class Small(Gated):
        max_request_bytes = 1 << 20

    served: Any = serve(Small())
    client = served.client(0)
    before = {p.name for p in Path("/dev/shm").glob(f"{served.prefix}*")}
    with pytest.raises(WorkerRejected, match="over the worker's"):
        client.step(np.zeros(8 << 20, dtype=np.uint8))
    assert {p.name for p in Path("/dev/shm").glob(f"{served.prefix}*")} == before


def test_a_request_whose_header_passes_the_cap_is_refused_before_it_is_sent(serve: Any) -> None:
    worker = Gated()
    served: Any = serve(worker)
    client = served.client(0)
    with pytest.raises(WorkerRejected, match="carry bulk data as numpy arrays"):
        client.step(list(range(codec.MAX_HEADER_BYTES // 4)))
    assert worker.steps == []
    assert client.step(np.arange(codec.MAX_HEADER_BYTES // 4))["slot"] == 0


# -- typed requests, results, and signals ---------------------------------------------------


def test_a_typed_step_sends_and_returns_the_declared_types(serve: Any) -> None:
    served: Any = serve(TypedWorker())
    client = _typed(served, 0)
    out = client.step(In(frame=np.arange(3.0), chunk=7))
    assert type(out) is Out
    assert out.chunk == 7
    np.testing.assert_array_equal(out.frame, np.arange(3.0) * 2)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"frame": np.ones(1), "chunk": 1}, "expected a .*In, got a dict"),
        (Out(frame=np.ones(1), chunk=1), "expected a .*In, got a Out"),
        (In(frame=np.ones(1), chunk=True), "chunk must be int"),
        # A list where an array is declared: the wrong type is what is tested.
        (In(frame=[1.0], chunk=1), "frame must be numpy.ndarray"),  # ty: ignore[invalid-argument-type]
    ],
)
def test_a_payload_that_is_not_the_request_type_is_refused_before_it_is_sent(
    serve: Any, payload: Any, reason: str
) -> None:
    served: Any = serve(TypedWorker())
    client = _typed(served, 0)
    with pytest.raises(TypeError, match=reason):
        client.step(payload)
    assert client.step(In(frame=np.ones(1), chunk=1)).chunk == 1


def test_a_plain_payload_that_does_not_travel_is_refused_before_it_is_sent(serve: Any) -> None:
    served: Any = serve(Gated())
    client = served.client(0)
    with pytest.raises(TypeError, match="does not travel"):
        client.step(object())
    with pytest.raises(TypeError, match="does not travel"):
        client.step({"on_done": print})
    with pytest.raises(TypeError, match="dtype object"):
        client.step(np.array([object()], dtype=object))
    assert client.step(1)["slot"] == 0


def test_a_signal_reaches_the_worker_as_its_type_while_a_step_runs(serve: Any) -> None:
    worker = TypedWorker()
    served: Any = serve(worker)
    client = _typed(served, 3)
    worker.gate.clear()
    running = threading.Thread(target=lambda: client.step(In(frame=np.ones(1), chunk=0)))
    running.start()
    worker.entered.wait(5)
    client.signal(Preempt(reason="barge"))
    _until(lambda: worker.received == [(3, Preempt(reason="barge"))])
    assert type(worker.received[0][1]) is Preempt
    worker.gate.set()
    running.join(5)


def test_a_signal_of_an_undeclared_type_is_refused_by_the_client(serve: Any) -> None:
    @dataclass(frozen=True)
    class Shutdown:
        now: bool

    worker = TypedWorker()
    served: Any = serve(worker)
    client = _typed(served, 0)
    with pytest.raises(ValueError, match="Shutdown is not a declared signal"):
        client.signal(Shutdown(now=True))
    with pytest.raises(ValueError, match="is not a declared signal"):
        client.signal("preempt")
    with pytest.raises(TypeError, match="reason must be str"):
        client.signal(Preempt(reason=1))  # ty: ignore[invalid-argument-type]  # the wrong type is tested
    assert client.available
    assert worker.received == []


# -- the environment ------------------------------------------------------------------------


def test_connect_reads_the_socket_and_slot_from_the_environment(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: Any = serve(Gated())
    monkeypatch.setenv("REACTOR_MODEL_WORKER_SOCKET", str(served.path))
    monkeypatch.setenv("REACTOR_SESSION_SLOT", "7")
    client = ModelClient.connect()
    served.clients.append(client)
    client.reset()
    assert client.step(1)["slot"] == 7


def _links(caplog: pytest.LogCaptureFixture) -> list[tuple[int, int]]:
    """Return the (slot, generation) of every ``model server link`` line so far."""
    links = []
    for record in caplog.records:
        if record.getMessage() == "model server link":
            fields = record.__dict__["reactor_fields"]
            links.append((fields["slot"], fields["generation"]))
    return links


def test_a_link_line_follows_every_connect_reconnect_and_reset(
    serve: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="reactor_runtime.model_server.client")
    served: Any = serve(Gated())
    client = served.client(5, reset=False)
    assert _links(caplog) == [(5, 1)]
    client.reset()
    assert _links(caplog) == [(5, 1), (5, 2)]
    client.reconnect(timeout=5)
    assert _links(caplog) == [(5, 1), (5, 2), (5, 3)]
    client.reset()
    assert _links(caplog) == [(5, 1), (5, 2), (5, 3), (5, 4)]


def test_no_link_line_after_a_reset_that_failed(
    serve: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="reactor_runtime.model_server.client")
    served: Any = serve(Gated())
    client = served.client(0, reset=False)
    served.halt()
    client.wait_lost(timeout=5)
    with pytest.raises(WorkerLost):
        client.reset()
    assert _links(caplog) == [(0, 1)]


def test_a_link_line_during_a_session_carries_its_id(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REACTOR_LOG_FORMAT", "json")
    buffer = io.StringIO()
    log.configure(level=logging.INFO, stream=buffer)
    served: Any = serve(Gated())
    client = served.client(2, reset=False)
    log.set_session_id("s-live")
    client.reset()
    records = [json.loads(line) for line in buffer.getvalue().splitlines()]
    links = [record for record in records if record["msg"] == "model server link"]
    assert [(r["slot"], r["generation"], r.get("session_id")) for r in links] == [
        (2, 1, None),
        (2, 2, "s-live"),
    ]


def test_connect_reads_its_deadline_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REACTOR_WORKER_CONNECT_TIMEOUT", "0.3")
    started = time.monotonic()
    with pytest.raises(WorkerTimeout, match=r"within 0.3 s"):
        ModelClient.connect(socket_path=tmp_path / "none.sock", slot=0)
    assert time.monotonic() - started < 2
    monkeypatch.setenv("REACTOR_WORKER_CONNECT_TIMEOUT", "soon")
    with pytest.raises(ValueError, match="REACTOR_WORKER_CONNECT_TIMEOUT 'soon'"):
        ModelClient.connect(socket_path=tmp_path / "none.sock", slot=0)


def test_a_step_reads_its_timeout_from_the_environment(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = Gated()
    served: Any = serve(worker)
    monkeypatch.setenv("REACTOR_WORKER_STEP_TIMEOUT", "0.2")
    client = served.client(0)
    worker.gate.clear()
    with pytest.raises(StepTimedOut, match=r"within 0.2 s"):
        client.step(1)
    worker.gate.set()


def test_the_session_process_caps_malloc_arenas(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    class Libc:
        def mallopt(self, option: int, value: int) -> int:
            calls.append((option, value))
            return 1

    monkeypatch.setattr(client_module.ctypes, "CDLL", lambda name: Libc())
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    monkeypatch.setattr(client_module, "_arenas_capped", False)
    client_module._cap_malloc_arenas()
    assert calls == [(-8, 2)]
    monkeypatch.setattr(client_module, "_arenas_capped", False)
    monkeypatch.setenv("MALLOC_ARENA_MAX", "4")
    client_module._cap_malloc_arenas()
    assert calls == [(-8, 2)]
    monkeypatch.delenv("MALLOC_ARENA_MAX")
    monkeypatch.setenv("REACTOR_SESSION_MALLOC_ARENAS", "0")
    monkeypatch.setattr(client_module, "_arenas_capped", False)
    client_module._cap_malloc_arenas()
    assert calls == [(-8, 2)]
