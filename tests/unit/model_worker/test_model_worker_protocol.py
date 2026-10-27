"""The model worker answers every session with its own output, by the documented policy.

The worker runs on a thread of the test process and each client is a real
``WorkerClient`` on a real Unix socket, so every request crosses the wire and
shared memory as it does between processes.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import socket
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime.distributed.ipc import SharedSlot, pack
from reactor_runtime.model_worker import (
    ModelWorker,
    WorkerClient,
    WorkerLost,
    WorkerRejected,
    WorkerRequest,
    WorkerTimeout,
    wire,
)
from reactor_runtime.model_worker.transport import UnixShmTransport, WorkerUnreachable
from reactor_runtime.model_worker.wire import Kind


class BoomError(RuntimeError):
    """The model's own error, raised for one request."""


class Recorder(ModelWorker):
    """Double each payload's array and record every call the server makes."""

    max_batch = 4

    def load(self, config_path: Path | None) -> None:
        self.batches: list[list[int]] = []
        self.events: list[tuple[str, int, int]] = []
        self.signals: list[tuple[int, str, Any]] = []
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
            self.signals.append((slot, name, payload))


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
    with pytest.raises(ValueError, match="read-only"):
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
    assert isinstance(results["failing"], BoomError)
    assert "slot 0 failed" in str(results["failing"])
    assert any("Raised in the model worker" in note for note in results["failing"].__notes__)
    np.testing.assert_array_equal(results["fine"]["x"], np.ones(3) * 2)
    # The failure ended nothing: the same client steps again.
    np.testing.assert_array_equal(failing.step(np.ones(2))["x"], np.ones(2) * 2)


def test_a_step_that_raises_fails_its_batch_and_the_worker_goes_on(serve: Any) -> None:
    served: Any = serve(Recorder())
    client = served.client(0)
    with pytest.raises(BoomError, match="whole step failed"):
        client.step({"raise": True})
    np.testing.assert_array_equal(client.step(np.ones(2))["x"], np.ones(2) * 2)


def test_a_step_returning_the_wrong_count_fails_every_request(serve: Any) -> None:
    class Short(ModelWorker):
        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            return []

    served: Any = serve(Short())
    with pytest.raises(RuntimeError, match="0 outputs for 1 requests"):
        served.client(0).step(1)


def test_an_output_that_does_not_pickle_fails_that_request(serve: Any) -> None:
    class Unpicklable(ModelWorker):
        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            return [threading.Lock()]

    served: Any = serve(Unpicklable())
    with pytest.raises(TypeError):
        served.client(0).step(1)
    with pytest.raises(TypeError):
        served.client(0 + 1).step(1)


def test_an_exception_that_does_not_pickle_arrives_as_a_runtime_error(serve: Any) -> None:
    class OddError(Exception):
        def __init__(self, lock: Any) -> None:
            super().__init__("odd")
            self.lock = lock

    class Raises(ModelWorker):
        def step(self, requests: list[WorkerRequest]) -> list[Any]:
            return [OddError(threading.Lock())]

    served: Any = serve(Raises())
    with pytest.raises(RuntimeError, match="OddError: odd"):
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
        kind, reason = _step_raw(sock, 1, pack(np.zeros(1 << 13, dtype=np.uint8), slot))
        assert kind is Kind.REJECTED
        assert b"over the 4096 limit" in reason
    finally:
        slot.close()
        sock.close()


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (b"not a pickle", b"does not decode"),
        (pickle.dumps(("blob", 1, [])), b"wrong shape"),
        (pickle.dumps((b"\x80\x05junk", "x", [])), b"does not unpickle"),
        (pickle.dumps((pickle.dumps(1), "rrother_0", [(0, 4)])), b"not one of this client's"),
    ],
)
def test_a_malformed_request_is_rejected_and_the_session_goes_on(
    serve: Any, header: bytes, reason: bytes
) -> None:
    served: Any = serve(Recorder())
    other = served.client(1)
    sock, prefix = _raw(served, 0)
    ok = SharedSlot(prefix=f"{prefix}ok_")
    try:
        kind, payload = _step_raw(sock, 1, header)
        assert kind is Kind.REJECTED
        assert reason in payload
        kind, payload = _step_raw(sock, 2, pack(3, ok))
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
        blob, name, _ = pickle.loads(pack(np.ones(8), slot))
        kind, payload = _step_raw(sock, 1, pickle.dumps((blob, name, [(0, 1 << 20)])))
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
    _until(lambda: worker.signals == [(3, "preempt", {"why": "barge"})])
    worker.gate.set()
    running.join(5)


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
    assert second.step(1)["gen"] == 1
    assert worker.events == [("opened", 0, 1), ("closed", 0, 1), ("opened", 0, 1)]


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

    def set_unavailable(self, reason: str) -> None:
        self.reports.append(reason)

    def set_available(self) -> None:
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
    kind, _ = _step_raw(sock, 1, pack(np.ones(1 << 12), slot))
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
