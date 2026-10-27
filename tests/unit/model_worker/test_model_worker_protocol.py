"""The model worker answers every session with its own output, by the documented policy.

The worker runs on a thread of the test process and each client is a real
``WorkerClient`` on a real Unix socket, so every request crosses the wire and
shared memory as it does between processes.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime.distributed.ipc import SharedSlot
from reactor_runtime.model_worker import (
    ModelWorker,
    WorkerClient,
    WorkerError,
    WorkerLost,
    WorkerRejected,
    WorkerRequest,
    WorkerTimeout,
    codec,
    wire,
)
from reactor_runtime.model_worker.transport import UnixShmTransport, WorkerUnreachable
from reactor_runtime.model_worker.wire import Kind


class BoomError(RuntimeError):
    """The model's own error, raised for one request."""


class Recorder(ModelWorker):
    """Double each payload's array and record every call the server makes."""

    max_batch = 4
    signals = frozenset({"preempt"})

    def load(self, config_path: Path | None) -> None:
        self.batches: list[list[int]] = []
        self.events: list[tuple[str, int, int]] = []
        self.received: list[tuple[int, str, Any]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.lock = threading.Lock()

    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        self.entered.set()
        self.gate.wait(10)
        with self.lock:
            self.batches.append([r.slot for r in requests])
        outputs: list[Any] = []
        for request in requests:
            payload = request.payload
            if isinstance(payload, dict) and payload.get("fail"):
                outputs.append(BoomError(f"slot {request.slot} failed"))
            elif isinstance(payload, dict) and payload.get("raise"):
                raise BoomError("the whole step failed")
            else:
                outputs.append({"slot": request.slot, "gen": request.generation, "x": payload * 2})
        return outputs

    def session_opened(self, slot: int, generation: int) -> None:
        self.events.append(("opened", slot, generation))

    def session_closed(self, slot: int, generation: int) -> None:
        self.events.append(("closed", slot, generation))

    def signal(self, slot: int, name: str, payload: Any) -> None:
        with self.lock:
            self.received.append((slot, name, payload))


def _until(check: Any, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


def test_a_step_round_trips_arrays_through_shared_memory(serve: Any) -> None:
    served: Any = serve(Recorder())
    client = served.client(0)
    frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    out = client.step(frame)
    assert out["slot"] == 0
    assert out["gen"] == 1
    np.testing.assert_array_equal(out["x"], frame * 2)
    # The session owns its copy: the next step cannot change it.
    out["x"][0, 0, 0] = 7
    again = client.step(frame)
    np.testing.assert_array_equal(again["x"], frame * 2)


def test_the_worker_sees_request_arrays_read_only(serve: Any) -> None:
    class Writes(ModelWorker):
        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            requests[0].payload[0] = 1
            return [None]

    served: Any = serve(Writes())
    with pytest.raises(WorkerError, match="read-only"):
        served.client(0).step(np.zeros(1 << 16, dtype=np.uint8))


def test_requests_waiting_together_are_stepped_together_in_arrival_order(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    clients = [served.client(slot) for slot in range(3)]
    worker.gate.clear()
    # Slot 0's step holds the worker while the other two queue behind it.
    first = threading.Thread(target=clients[0].step, args=(np.ones(4),))
    first.start()
    worker.entered.wait(5)
    others = []
    for client in (clients[2], clients[1]):
        thread = threading.Thread(target=client.step, args=(np.ones(4),))
        thread.start()
        others.append(thread)
        time.sleep(0.05)
    worker.gate.set()
    for thread in [first, *others]:
        thread.join(5)
    assert worker.batches == [[0], [2, 1]]


def test_max_batch_one_steps_sessions_one_at_a_time(serve: Any) -> None:
    class One(Recorder):
        max_batch = 1

    worker = One()
    served: Any = serve(worker)
    clients = [served.client(slot) for slot in range(3)]
    worker.gate.clear()
    threads = [threading.Thread(target=c.step, args=(np.ones(2),)) for c in clients]
    threads[0].start()
    worker.entered.wait(5)
    for thread in threads[1:]:
        thread.start()
        time.sleep(0.05)
    worker.gate.set()
    for thread in threads:
        thread.join(5)
    assert worker.batches == [[0], [1], [2]]


def test_the_batch_window_waits_for_sessions_arriving_together(serve: Any) -> None:
    class Windowed(Recorder):
        batch_window = 0.5

    worker = Windowed()
    served: Any = serve(worker)
    clients = [served.client(slot) for slot in range(3)]
    threads = [threading.Thread(target=c.step, args=(np.ones(2),)) for c in clients]
    for thread in threads:
        thread.start()
        time.sleep(0.02)
    for thread in threads:
        thread.join(5)
    assert worker.batches == [[0, 1, 2]]


def test_the_batch_window_ends_early_when_every_open_session_waits(serve: Any) -> None:
    class Windowed(Recorder):
        batch_window = 30.0

    worker = Windowed()
    served: Any = serve(worker)
    clients = [served.client(slot) for slot in range(2)]
    started = time.monotonic()
    threads = [threading.Thread(target=c.step, args=(np.ones(2),)) for c in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert time.monotonic() - started < 5
    assert [sorted(batch) for batch in worker.batches] == [[0, 1]]


def test_a_session_that_joins_mid_step_waits_for_the_next_step(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    first = served.client(0)
    worker.gate.clear()
    running = threading.Thread(target=first.step, args=(np.ones(2),))
    running.start()
    worker.entered.wait(5)
    joined: list[WorkerClient] = []
    joiner = threading.Thread(target=lambda: joined.append(served.client(1)))
    joiner.start()
    time.sleep(0.2)
    # The join is answered between steps, not during one.
    assert not joined
    worker.gate.set()
    running.join(5)
    joiner.join(5)
    assert joined[0].step(np.ones(2))["slot"] == 1
    assert worker.batches == [[0], [1]]


def test_a_session_that_leaves_mid_step_is_closed_after_it(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    leaving = served.client(0)
    staying = served.client(1)
    worker.gate.clear()
    threading.Thread(target=_swallow, args=(leaving.step, np.ones(2)), daemon=True).start()
    worker.entered.wait(5)
    leaving.close()
    time.sleep(0.1)
    assert ("closed", 0, 1) not in worker.events
    worker.gate.set()
    _until(lambda: ("closed", 0, 1) in worker.events)
    assert staying.step(np.ones(2))["slot"] == 1


def test_an_output_that_is_an_exception_fails_that_request_alone(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    failing, fine = served.client(0), served.client(1)
    worker.gate.clear()
    results: dict[str, Any] = {}

    def run(name: str, client: WorkerClient, payload: Any) -> None:
        try:
            results[name] = client.step(payload)
        except Exception as exc:
            results[name] = exc

    holder = served.client(2)
    hold = threading.Thread(target=holder.step, args=(np.ones(1),))
    hold.start()
    worker.entered.wait(5)
    threads = [
        threading.Thread(target=run, args=("failing", failing, {"fail": True})),
        threading.Thread(target=run, args=("fine", fine, np.ones(3))),
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.1)
    worker.gate.set()
    for thread in [hold, *threads]:
        thread.join(5)
    assert [0, 1] in worker.batches
    failure = results["failing"]
    assert isinstance(failure, WorkerError)
    assert failure.type_name == "test_model_worker_protocol.BoomError"
    assert failure.message == "slot 0 failed"
    assert "BoomError" in failure.worker_traceback
    np.testing.assert_array_equal(results["fine"]["x"], np.ones(3) * 2)
    # The failure ended nothing: the same client steps again.
    np.testing.assert_array_equal(failing.step(np.ones(2))["x"], np.ones(2) * 2)


def test_a_step_that_raises_fails_its_batch_and_the_worker_goes_on(serve: Any) -> None:
    served: Any = serve(Recorder())
    client = served.client(0)
    with pytest.raises(WorkerError, match="BoomError: the whole step failed"):
        client.step({"raise": True})
    np.testing.assert_array_equal(client.step(np.ones(2))["x"], np.ones(2) * 2)


def test_a_step_returning_the_wrong_count_fails_every_request(serve: Any) -> None:
    class Short(ModelWorker):
        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            return []

    served: Any = serve(Short())
    with pytest.raises(RuntimeError, match="0 outputs for 1 requests"):
        served.client(0).step(1)


def test_an_output_that_does_not_travel_fails_that_request(serve: Any) -> None:
    class Unpicklable(ModelWorker):
        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            return [threading.Lock()]

    served: Any = serve(Unpicklable())
    with pytest.raises(WorkerError, match="TypeError: a lock does not travel"):
        served.client(0).step(1)
    with pytest.raises(WorkerError, match="TypeError"):
        served.client(1).step(1)


def test_a_payload_that_does_not_travel_is_refused_before_it_is_sent(serve: Any) -> None:
    served: Any = serve(Recorder())
    client = served.client(0)
    with pytest.raises(TypeError, match="does not travel"):
        client.step(object())
    with pytest.raises(TypeError, match="dtype object"):
        client.step(np.array([object()], dtype=object))
    assert client.step(np.ones(2))["slot"] == 0


def test_an_exception_whatever_it_holds_arrives_by_name_and_message(serve: Any) -> None:
    class OddError(Exception):
        def __init__(self, lock: Any) -> None:
            super().__init__("odd")
            self.lock = lock

    class Raises(ModelWorker):
        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            return [OddError(threading.Lock())]

    served: Any = serve(Raises())
    with pytest.raises(WorkerError, match="OddError: odd"):
        served.client(0).step(1)


def test_an_oversize_request_is_refused_for_that_session_alone(serve: Any) -> None:
    class Small(Recorder):
        max_request_bytes = 1 << 16

    served: Any = serve(Small())
    big, small = served.client(0), served.client(1)
    with pytest.raises(WorkerRejected, match="over the worker's 65536 limit"):
        big.step(np.zeros(1 << 17, dtype=np.uint8))
    assert small.step(np.ones(4))["slot"] == 1
    assert big.step(np.ones(4))["slot"] == 0


def _raw(served: Any, slot: int) -> tuple[socket.socket, str]:
    """Connect by hand; return the socket and the prefix the worker maps requests under."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    hello = {"slot": slot, "pid": os.getpid(), "token": "abcd1234"}
    wire.send(sock, Kind.HELLO, 0, wire.to_json(hello))
    kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
    assert kind is Kind.WELCOME
    return sock, wire.from_json(payload)["request_prefix"]


def _header(tree: Any, *, block: str = "OWN", spans: list[list[int]] | None = None) -> bytes:
    """A request header written by hand, as a hostile or broken session might.

    ``OWN`` stands for the session's own input block, filled in once connected.
    """
    return json.dumps({"v": 1, "block": block, "spans": spans or [], "tree": tree}).encode()


def _deep(depth: int) -> Any:
    tree: Any = 1
    for _ in range(depth):
        tree = {"l": [tree]}
    return tree


def _step_raw(sock: socket.socket, request_id: int, header: bytes) -> tuple[Kind, bytes]:
    wire.send(sock, Kind.STEP, request_id, wire.TIMEOUT.pack(5.0) + header)
    kind, got, payload = wire.recv(sock, 1 << 20)
    assert got == request_id
    return kind, payload


def test_the_worker_refuses_an_oversize_request_the_client_did_not_check(serve: Any) -> None:
    class Small(Recorder):
        max_request_bytes = 1 << 12

    served: Any = serve(Small())
    sock, prefix = _raw(served, 0)
    slot = SharedSlot(prefix=f"{prefix}raw_")
    try:
        header, _ = codec.encode(np.zeros(1 << 13, dtype=np.uint8), slot)
        kind, reason = _step_raw(sock, 1, header)
        assert kind is Kind.REJECTED
        assert b"over the 4096 limit" in reason
    finally:
        slot.close()
        sock.close()


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (b"not json", b"not JSON"),
        # A pickle that would run code if anything unpickled it.
        (pickle.dumps(shutil.rmtree), b"not JSON"),
        (json.dumps({"v": 2, "block": "x", "spans": [], "tree": 1}).encode(), b"version 1"),
        (_header({"o": 1}), b"unknown or malformed tag"),
        (_header({"a": ["|O", [1], 0]}, spans=[[0, 8]]), b"does not travel"),
        (_header({"a": ["<f8", [3], 0]}, spans=[[0, 8]]), b"needs 24 bytes"),
        (_header({"a": ["V8", [1], 0]}, spans=[[0, 8]]), b"does not travel"),
        (_header({"b": 5}, spans=[[0, 1]]), b"span 5"),
        (_header(_deep(80)), b"deeper than"),
        (_header(1, block="rrother_0", spans=[[0, 4]]), b"not one of this client's"),
        (_header(1, spans=[[-1, 4]]), b"two non-negative integers"),
    ],
)
def test_a_malformed_request_is_rejected_and_the_session_goes_on(
    serve: Any, header: bytes, reason: bytes
) -> None:
    served: Any = serve(Recorder())
    other = served.client(1)
    sock, prefix = _raw(served, 0)
    ok = SharedSlot(prefix=f"{prefix}ok_")
    header = header.replace(b'"OWN"', json.dumps(f"{prefix}in").encode())
    try:
        kind, payload = _step_raw(sock, 1, header)
        assert kind is Kind.REJECTED
        assert reason in payload
        kind, payload = _step_raw(sock, 2, codec.encode(3, ok)[0])
        assert kind is Kind.RESULT
    finally:
        ok.close()
        sock.close()
    assert other.step(np.ones(2))["slot"] == 1


def test_a_span_past_the_block_is_rejected(serve: Any) -> None:
    served: Any = serve(Recorder())
    sock, prefix = _raw(served, 0)
    slot = SharedSlot(prefix=f"{prefix}span_", initial_bytes=4096)
    try:
        header = json.loads(codec.encode(np.ones(8), slot)[0])
        header["spans"] = [[0, 1 << 20]]
        kind, payload = _step_raw(sock, 1, json.dumps(header).encode())
        assert kind is Kind.REJECTED
        assert b"past the block" in payload
    finally:
        slot.close()
        sock.close()


def test_a_frame_that_breaks_the_protocol_closes_that_connection_alone(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    other = served.client(1)
    sock, _ = _raw(served, 0)
    sock.sendall(b"\xff" * 13)
    sock.settimeout(5)
    assert sock.recv(1) == b""
    sock.close()
    _until(lambda: ("closed", 0, 1) in worker.events)
    assert other.step(np.ones(2))["slot"] == 1


def test_a_step_that_times_out_is_cancelled_and_the_client_goes_on(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    client = served.client(0)
    worker.gate.clear()
    with pytest.raises(WorkerTimeout, match=r"no answer within 0.2"):
        client.step(np.ones(2), timeout=0.2)
    worker.gate.set()
    out = client.step(np.full(2, 5.0))
    np.testing.assert_array_equal(out["x"], np.full(2, 10.0))


def test_a_request_that_expired_while_queued_never_reaches_step(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    holder, late = served.client(0), served.client(1)
    worker.gate.clear()
    hold = threading.Thread(target=holder.step, args=(np.ones(1),))
    hold.start()
    worker.entered.wait(5)
    with pytest.raises(WorkerTimeout):
        late.step(np.ones(1), timeout=0.2)
    # Past its deadline on the worker too, whether or not its cancel landed.
    time.sleep(0.1)
    worker.gate.set()
    hold.join(5)
    time.sleep(0.2)
    assert worker.batches == [[0]]


def test_signals_reach_the_worker_while_a_step_runs(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    client = served.client(3)
    worker.gate.clear()
    running = threading.Thread(target=client.step, args=(np.ones(1),))
    running.start()
    worker.entered.wait(5)
    client.signal("preempt", {"why": "barge"})
    _until(lambda: worker.received == [(3, "preempt", {"why": "barge"})])
    worker.gate.set()
    running.join(5)


def test_an_unknown_signal_is_refused_by_the_client_and_dropped_by_the_worker(
    serve: Any,
) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    client = served.client(0)
    with pytest.raises(ValueError, match="declares no signal 'shutdown'"):
        client.signal("shutdown")
    sock, _ = _raw(served, 1)
    try:
        # Unknown names, a pickle, and garbage reach no signal() and close nothing.
        wire.send(sock, Kind.SIGNAL, 0, wire.to_json({"name": "shutdown", "payload": 1}))
        wire.send(sock, Kind.SIGNAL, 0, pickle.dumps(("preempt", shutil.rmtree)))
        wire.send(sock, Kind.SIGNAL, 0, b"\xff\xfe")
        wire.send(sock, Kind.SIGNAL, 0, wire.to_json({"name": "preempt", "payload": [1, 2]}))
        _until(lambda: worker.received == [(1, "preempt", [1, 2])])
        assert client.step(1)["slot"] == 0
    finally:
        sock.close()


def test_reset_closes_and_reopens_the_slot_with_a_new_generation(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    client = served.client(0)
    assert client.generation == 1
    assert client.reset() == 2
    assert client.step(1)["gen"] == 2
    assert worker.events == [("opened", 0, 1), ("closed", 0, 1), ("opened", 0, 2)]


def test_a_refusing_session_opened_rejects_the_connect(serve: Any) -> None:
    class Refuses(Recorder):
        def session_opened(self, slot: int, generation: int) -> None:
            if slot == 9:
                raise ValueError("no room for slot 9")

    served: Any = serve(Refuses())
    with pytest.raises(WorkerRejected, match="no room for slot 9"):
        served.client(9)
    assert served.client(1).step(1)["slot"] == 1


def test_a_slot_held_by_a_live_process_is_refused(serve: Any) -> None:
    served: Any = serve(Recorder())
    served.client(0)
    with pytest.raises(WorkerRejected, match="slot 0 is held by process"):
        served.client(0)


def test_a_slot_is_closed_before_it_is_opened_again(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    first = served.client(0)
    first.close()
    second = served.client(0)
    assert second.step(1)["gen"] == 2
    assert worker.events == [("opened", 0, 1), ("closed", 0, 1), ("opened", 0, 2)]


def test_a_dead_worker_ends_every_session_with_its_reason(serve: Any) -> None:
    served: Any = serve(Recorder())
    clients = [served.client(slot) for slot in range(2)]
    served.server.stop()
    served.thread.join(5)
    for client in clients:
        with pytest.raises(WorkerLost, match=r"^worker_died: "):
            client.step(1)
        # Every later call raises the same while no worker is there.
        with pytest.raises(WorkerLost, match=r"^worker_died: "):
            client.step(1)
        assert not client.available
        client.signal("ignored")


def test_a_client_reconnects_to_a_worker_that_comes_back(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    client = served.client(4)
    served.server.stop()
    served.thread.join(5)
    with pytest.raises(WorkerLost):
        client.step(1)
    again: Any = serve(Recorder(), path=served.path)
    assert client.wait_available(10)
    assert client.slot == 4
    assert client.step(1)["slot"] == 4
    assert again.worker.events == [("opened", 4, 1)]


class FakeApp:
    """Stands in for a session's app: records what the client reports to it."""

    def __init__(self) -> None:
        self.reports: list[str | None] = []

    def report_unavailable(self, reason: str) -> None:
        self.reports.append(reason)

    def report_available(self) -> None:
        self.reports.append(None)


def test_a_client_with_an_app_reports_the_worker_going_and_coming_back(serve: Any) -> None:
    served: Any = serve(Recorder())
    app = FakeApp()
    client = WorkerClient.connect(app, socket_path=served.path, slot=0)
    served.clients.append(client)
    assert app.reports == [None]
    served.server.stop()
    served.thread.join(5)
    _until(lambda: len(app.reports) == 2)
    assert app.reports[1] is not None
    assert app.reports[1].startswith("worker_died: ")
    serve(Recorder(), path=served.path)
    _until(lambda: app.reports[-1] is None and len(app.reports) == 3)


def test_a_client_with_an_app_starts_before_its_worker(serve: Any, tmp_path: Path) -> None:
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="rrq")) / "late.sock"
    app = FakeApp()
    client = WorkerClient.connect(app, socket_path=path, slot=2)
    try:
        assert app.reports
        assert app.reports[0] is not None
        assert app.reports[0].startswith("worker_unavailable: ")
        with pytest.raises(WorkerLost, match=r"^worker_unavailable: "):
            client.step(1)
        serve(Recorder(), path=path)
        assert client.wait_available(10)
        assert app.reports[-1] is None
        assert client.step(1)["slot"] == 2
    finally:
        client.close()


def test_the_worker_assigns_the_lowest_free_slot(serve: Any) -> None:
    served: Any = serve(Recorder())
    first = served.client(0)
    auto = WorkerClient.connect(socket_path=served.path)
    served.clients.append(auto)
    assert first.slot == 0
    assert auto.slot == 1
    assert auto.step(1)["slot"] == 1


def test_closing_releases_every_shared_block(serve: Any) -> None:
    served: Any = serve(Recorder())
    client = served.client(0)
    client.step(np.ones(1 << 16, dtype=np.uint8))
    shm = Path("/dev/shm")
    if not shm.is_dir():
        pytest.skip("shared blocks are not files on this platform")

    def blocks() -> list[Path]:
        # The worker's probe block lives as long as the worker.
        return [p for p in shm.glob(f"{served.prefix}*") if not p.name.endswith("probe")]

    assert blocks()
    client.close()
    _until(lambda: not blocks())


def test_a_session_that_vanishes_has_its_blocks_unlinked_by_the_worker(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    sock, prefix = _raw(served, 0)
    slot = SharedSlot(prefix=f"{prefix}gone_")
    kind, _ = _step_raw(sock, 1, codec.encode(np.ones(1 << 12), slot)[0])
    assert kind is Kind.RESULT
    # The session dies without closing what it made.
    sock.close()
    _until(lambda: ("closed", 0, 1) in worker.events)
    _until(lambda: not list(Path("/dev/shm").glob(f"{prefix}*")))
    slot._shm.close()


def test_connect_reads_the_socket_and_slot_from_the_environment(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: Any = serve(Recorder())
    monkeypatch.setenv("REACTOR_MODEL_WORKER_SOCKET", str(served.path))
    monkeypatch.setenv("REACTOR_SESSION_SLOT", "7")
    client = WorkerClient.connect()
    served.clients.append(client)
    assert client.step(1)["slot"] == 7


def test_connect_gives_up_when_no_worker_listens(tmp_path: Path) -> None:
    with pytest.raises(WorkerTimeout, match="no model worker accepted"):
        WorkerClient.connect(socket_path=tmp_path / "none.sock", slot=0, connect_timeout=0.3)


def test_a_hello_without_a_token_is_refused(serve: Any) -> None:
    served: Any = serve(Recorder())
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    wire.send(sock, Kind.HELLO, 0, wire.to_json({"slot": 0}))
    kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
    sock.close()
    assert kind is Kind.REFUSED
    assert b"token" in payload


@pytest.mark.parametrize(
    ("attr", "value"), [("max_batch", 0), ("batch_window", -1.0), ("max_request_bytes", 0)]
)
def test_a_bad_policy_is_refused_at_start(tmp_path: Path, attr: str, value: Any) -> None:
    from reactor_runtime.model_worker.server import WorkerServer

    bad = type("Bad", (ModelWorker,), {attr: value})
    with pytest.raises(ValueError, match=attr):
        WorkerServer(bad(), tmp_path / "x.sock", shm_prefix="rrx_")


def _swallow(call: Any, *args: Any) -> None:
    with contextlib.suppress(Exception):
        call(*args)


def test_a_missing_socket_directory_fails_at_once_even_with_an_app(tmp_path: Path) -> None:
    missing = tmp_path / "not-mounted" / "sock"
    with pytest.raises(WorkerUnreachable, match=r"socket directory .* does not exist"):
        WorkerClient.connect(socket_path=missing, slot=0, connect_timeout=30)
    with pytest.raises(WorkerUnreachable, match="does not exist"):
        WorkerClient.connect(FakeApp(), socket_path=missing, slot=0)


def test_shared_memory_the_session_cannot_see_fails_the_connect(serve: Any) -> None:
    served: Any = serve(Recorder())
    transport = UnixShmTransport()
    broken = {"request_prefix": "rrx_", "probe": "rrnosuchprobe", "probe_nonce": "00"}
    with pytest.raises(WorkerUnreachable, match="must share /dev/shm"):
        transport.client_carrier(broken)
    wrong = {**broken, "probe": f"{served.prefix}probe", "probe_nonce": "ffffffffffffffff"}
    with pytest.raises(WorkerUnreachable, match="do not share one /dev/shm"):
        transport.client_carrier(wrong)


def test_a_worker_refuses_to_start_without_its_socket_directory(tmp_path: Path) -> None:
    from reactor_runtime.model_worker.server import WorkerServer

    server = WorkerServer(Recorder(), tmp_path / "gone" / "x.sock", shm_prefix="rrx_")
    with pytest.raises(WorkerUnreachable, match="does not exist"):
        server.bind()


def _shm_sizes(prefix: str) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in Path("/dev/shm").glob(f"{prefix}*")}


def test_a_sessions_blocks_are_made_at_join_and_gone_after_it_leaves(serve: Any) -> None:
    class Sized(Recorder):
        shm_input_bytes = 3 << 20
        shm_output_bytes = 5 << 20

    if not Path("/dev/shm").is_dir():
        pytest.skip("shared blocks are not files on this platform")
    served: Any = serve(Sized())
    assert Sized.shm_slot_size() == 8 << 20
    client = served.client(6)
    # Before any step: the input block, and the output block, at their sizes.
    blocks = _shm_sizes(f"{served.prefix}")
    inputs = [n for n in blocks if n.startswith(f"{served.prefix}s6_") and n.endswith("_in")]
    outputs = [n for n in blocks if n.startswith(f"{served.prefix}w6c")]
    assert [blocks[n] for n in inputs] == [3 << 20]
    assert [blocks[n] for n in outputs] == [5 << 20]
    client.close()
    _until(lambda: not [n for n in _shm_sizes(served.prefix) if "s6_" in n or "w6c" in n])


def test_a_request_that_outgrows_its_block_still_travels(serve: Any) -> None:
    class Small(Recorder):
        shm_input_bytes = 4096

    served: Any = serve(Small())
    client = served.client(0)
    big = np.arange(1 << 16, dtype=np.float64)
    np.testing.assert_array_equal(client.step(big)["x"], big * 2)
    np.testing.assert_array_equal(client.step(np.ones(4))["x"], np.ones(4) * 2)


def test_a_session_the_worker_has_no_memory_for_is_refused(serve: Any) -> None:
    from reactor_runtime.distributed import SharedSlotAllocationFailed
    from reactor_runtime.model_worker.server import WorkerServer

    class Full(UnixShmTransport):
        def worker_carrier(self, *args: Any, **kwargs: Any) -> Any:
            raise SharedSlotAllocationFailed("/dev/shm has 0 bytes free")

    import tempfile

    worker = Recorder()
    worker.device = "cpu"
    worker.load(None)
    path = Path(tempfile.mkdtemp(prefix="rrf")) / "w.sock"
    server = WorkerServer(worker, path, shm_prefix="rrfull_", transport=Full())
    server.bind()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        with pytest.raises(WorkerRejected, match="no shared memory for one more session"):
            WorkerClient.connect(socket_path=path, slot=0)
    finally:
        server.stop()
        thread.join(5)


def test_the_session_process_caps_malloc_arenas(monkeypatch: pytest.MonkeyPatch) -> None:
    from reactor_runtime.model_worker import client as client_module

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


def test_a_worker_that_drops_a_session_right_after_welcome_is_reconnected_to(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reactor_runtime.model_worker import client as client_module
    from reactor_runtime.model_worker.server import WorkerServer

    monkeypatch.setattr(client_module, "_RECONNECT_FIRST", 0.05)
    monkeypatch.setattr(client_module, "_RECONNECT_MAX", 0.1)
    drops = {"left": 3}
    original = WorkerServer._open_conn

    def flaky(self: WorkerServer, conn: Any) -> None:
        original(self, conn)
        if drops["left"] > 0:
            drops["left"] -= 1
            conn.sock.shutdown(socket.SHUT_RDWR)

    monkeypatch.setattr(WorkerServer, "_open_conn", flaky)
    served: Any = serve(Recorder())
    app = FakeApp()
    client = WorkerClient.connect(app, socket_path=served.path, slot=0)
    served.clients.append(client)
    _until(lambda: drops["left"] == 0 and client.available, within=10)
    time.sleep(0.3)
    assert client.available
    assert app.reports[-1] is None
    assert client.step(1)["slot"] == 0


def test_a_session_whose_reset_was_refused_is_rejected_at_once(serve: Any) -> None:
    class RefusesReopen(Recorder):
        def session_opened(self, slot: int, generation: int) -> None:
            super().session_opened(slot, generation)
            if generation > 1:
                raise ValueError("no room")

    served: Any = serve(RefusesReopen())
    client = served.client(0)
    with pytest.raises(WorkerRejected, match="no room"):
        client.reset()
    started = time.monotonic()
    # The step first retries the reset the worker refused, and is refused again.
    with pytest.raises(WorkerRejected, match="no room"):
        client.step(1)
    assert time.monotonic() - started < 1.0
    # The step thread is idle, not spinning on the request it cannot run.
    other = served.client(1)
    assert other.step(1)["slot"] == 1


def test_a_held_slot_is_refused_even_when_the_peer_pid_is_unknown(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reactor_runtime.model_worker import server as server_module

    # Containers of a pod do not share a PID namespace: the peer reads as pid 0.
    monkeypatch.setattr(server_module, "_peer_pid", lambda sock: 0)
    worker = Recorder()
    served: Any = serve(worker)
    first = served.client(0)
    with pytest.raises(WorkerRejected, match="slot 0 is held"):
        served.client(0)
    assert first.step(1)["slot"] == 0
    # A holder whose socket is gone is closed, and the slot opens again.
    sock, _ = _raw(served, 3)
    sock.close()
    again = served.client(3)
    assert again.step(1)["slot"] == 3


def test_an_oversize_request_leaves_no_block_behind(serve: Any) -> None:
    class Small(Recorder):
        max_request_bytes = 1 << 20

    served: Any = serve(Small())
    client = served.client(0)
    before = set(_shm_sizes(served.prefix))
    with pytest.raises(WorkerRejected, match="over the worker's"):
        client.step(np.zeros(8 << 20, dtype=np.uint8))
    assert set(_shm_sizes(served.prefix)) == before


def test_an_output_that_outgrows_its_block_grows_it_in_place(serve: Any) -> None:
    class Sized(ModelWorker):
        shm_output_bytes = 4096

        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            return [np.zeros(int(r.payload), dtype=np.uint8) for r in requests]

    served: Any = serve(Sized())
    client = served.client(0)
    before = _shm_sizes(served.prefix)
    assert client.step(1 << 20).nbytes == 1 << 20
    grown = _shm_sizes(served.prefix)
    # One block replaced by one its size, not one more block.
    assert len(grown) == len(before)
    assert max(grown.values()) >= 1 << 20
    assert client.step(16).nbytes == 16
    assert _shm_sizes(served.prefix) == grown


def test_a_step_that_times_out_is_a_model_unavailable() -> None:
    from reactor_runtime import ModelUnavailable
    from reactor_runtime.model_worker import StepTimedOut, WorkerLost

    assert issubclass(StepTimedOut, WorkerTimeout)
    assert issubclass(StepTimedOut, ModelUnavailable)
    assert issubclass(WorkerLost, ModelUnavailable)


def test_an_oversize_control_frame_is_refused_before_it_is_read(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    sock, _ = _raw(served, 0)
    # Announce a 100 MB signal and send none of it: the worker must not wait for it.
    sock.sendall(wire.HEADER.pack(100 << 20, Kind.SIGNAL, 0))
    sock.settimeout(5)
    assert sock.recv(1) == b""
    sock.close()
    _until(lambda: ("closed", 0, 1) in worker.events)


def test_a_batch_that_raises_is_stepped_again_one_request_at_a_time(serve: Any) -> None:
    class Poisoned(Recorder):
        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            if any(isinstance(r.payload, dict) and r.payload.get("poison") for r in requests):
                with self.lock:
                    self.batches.append([r.slot for r in requests])
                raise BoomError("out of memory")
            return super().step(requests)

    worker = Poisoned()
    served: Any = serve(worker)
    clients = [served.client(slot) for slot in range(3)]
    holder = served.client(3)
    worker.gate.clear()
    hold = threading.Thread(target=holder.step, args=(np.ones(1),))
    hold.start()
    worker.entered.wait(5)
    results: dict[int, Any] = {}

    def run(slot: int, payload: Any) -> None:
        try:
            results[slot] = clients[slot].step(payload)
        except Exception as exc:
            results[slot] = exc

    threads = [
        threading.Thread(target=run, args=(0, np.ones(2))),
        threading.Thread(target=run, args=(1, {"poison": True})),
        threading.Thread(target=run, args=(2, np.ones(3))),
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    worker.gate.set()
    for thread in [hold, *threads]:
        thread.join(10)
    assert [0, 1, 2] in [sorted(batch) for batch in worker.batches]
    assert isinstance(results[1], WorkerError)
    np.testing.assert_array_equal(results[0]["x"], np.ones(2) * 2)
    np.testing.assert_array_equal(results[2]["x"], np.ones(3) * 2)


def test_after_a_timeout_the_input_block_is_used_again(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    client = served.client(0)
    worker.gate.clear()
    with pytest.raises(WorkerTimeout):
        client.step(np.ones(1024), timeout=0.1)
    worker.gate.set()
    client.step(np.ones(1024))
    blocks = set(_shm_sizes(served.prefix))
    for _ in range(5):
        client.step(np.ones(1024))
    assert set(_shm_sizes(served.prefix)) <= blocks
    assert not [n for n in _shm_sizes(served.prefix) if "_g" in n]


def test_an_incomplete_reset_is_retried_before_the_next_step(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    client = served.client(0)
    worker.gate.clear()
    hold = threading.Thread(target=_swallow, args=(served.client(1).step, np.ones(1)))
    hold.start()
    worker.entered.wait(5)
    with pytest.raises(WorkerTimeout):
        client.reset(timeout=0.1)
    worker.gate.set()
    hold.join(5)
    out = client.step(1)
    assert out["gen"] > 1
    assert ("closed", 0, 1) in worker.events


@pytest.mark.parametrize(
    "value",
    [
        np.array(7),
        np.array(3.5, dtype=np.float32),
        np.zeros((0, 4), dtype=np.int16),
        np.arange(6, dtype=">i4").reshape(2, 3),
        np.array(["2026-10-26"], dtype="datetime64[D]"),
        np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2],
        np.array([1 + 2j], dtype=np.complex64),
        {"t": (1, "a", None, True, b"xy"), 3: [1.5, float("inf"), float("-inf")]},
    ],
)
def test_the_codec_round_trips_what_it_carries(value: Any) -> None:
    slot = SharedSlot(initial_bytes=4096)
    try:
        header, _ = codec.encode(value, slot)
        _, spans, tree = codec.parse(header)
        buf = slot._shm.buf
        assert buf is not None
        copies = [memoryview(bytearray(buf[o : o + n])) for o, n in spans]
        del buf
    finally:
        slot.close()
    out = codec.decode(tree, copies.__getitem__)
    if isinstance(value, np.ndarray):
        assert out.shape == value.shape
        assert out.dtype == value.dtype
        np.testing.assert_array_equal(out, value)
    else:
        assert out == value


def test_an_array_that_does_not_rebuild_is_rejected_not_a_closed_connection(serve: Any) -> None:
    served: Any = serve(Recorder())
    sock, prefix = _raw(served, 0)
    try:
        header = _header({"a": ["<u1", [2**62, 2**62, 0], 0]}, spans=[[0, 0]])
        header = header.replace(b'"OWN"', json.dumps(f"{prefix}in").encode())
        kind, _ = _step_raw(sock, 1, header)
        assert kind is Kind.REJECTED
        bad_name = _header(1, block=f"{prefix}in\\u0000x", spans=[[0, 1]])
        kind, _ = _step_raw(sock, 2, bad_name)
        assert kind is Kind.REJECTED
    finally:
        sock.close()


def test_the_availability_replay_at_bind_comes_before_any_later_report() -> None:
    from reactor_runtime import ReactorApp

    class App(ReactorApp):
        def generate(self, input: Any) -> Any:
            return None

    app = App()
    app.report_unavailable("worker_unavailable: not up")
    heard: list[str | None] = []
    reporter: list[threading.Thread] = []

    def sink(reason: str | None) -> None:
        if not reporter:
            reporter.append(threading.Thread(target=app.report_available))
            reporter[0].start()
            time.sleep(0.1)
        heard.append(reason)

    app._bind_availability(sink)
    reporter[0].join(5)
    assert heard == ["worker_unavailable: not up", None]


def test_a_reconnect_that_cannot_reach_the_worker_reports_why(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reactor_runtime.model_worker import client as client_module

    monkeypatch.setattr(client_module, "_RECONNECT_FIRST", 0.05)
    monkeypatch.setattr(client_module, "_RECONNECT_MAX", 0.1)
    served: Any = serve(Recorder())
    app = FakeApp()
    client = WorkerClient.connect(app, socket_path=served.path, slot=0)
    served.clients.append(client)
    holder_path = served.path
    served.server.stop()
    served.thread.join(5)

    class Blind(UnixShmTransport):
        def client_carrier(self, welcome: dict[str, Any]) -> Any:
            raise WorkerUnreachable("the model worker's shared memory is not visible here")

    client._transport = Blind()
    serve(Recorder(), path=holder_path)
    _until(lambda: app.reports and "not visible here" in str(app.reports[-1]), within=5)
