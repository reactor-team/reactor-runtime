"""The worker's server: accepting sessions, reading their frames, batching, and stepping.

The worker runs on a thread of the test process. Sessions are real
``ModelClient`` instances on a real Unix socket, or raw sockets that write the
frames a broken or hostile session would.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import shutil
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime.distributed.ipc import SharedSlot
from reactor_runtime.model_server import (
    BatchFailed,
    ModelClient,
    ModelServer,
    ServerRequest,
    ServerSettings,
    WorkerError,
    WorkerLost,
    WorkerRejected,
    WorkerUnreachable,
    codec,
    wire,
)
from reactor_runtime.model_server.server import WorkerServer
from reactor_runtime.model_server.transport import UnixShmTransport
from reactor_runtime.model_server.wire import Kind


class BoomError(RuntimeError):
    """The model's own error, raised for one request."""


class Recorder(ModelServer):
    """Double each payload's array and record every call the server makes."""

    max_batch = 4

    def load(self, config_path: Path | None) -> None:
        self.batches: list[list[int]] = []
        self.events: list[tuple[str, int, int]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.lock = threading.Lock()

    def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
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


def _until(check: Any, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


def _swallow(call: Any, *args: Any) -> None:
    with contextlib.suppress(Exception):
        call(*args)


def _run_into(results: dict[Any, Any], key: Any, call: Any, *args: Any) -> threading.Thread:
    def run() -> None:
        try:
            results[key] = call(*args)
        except Exception as exc:
            results[key] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def _hello(served: Any, at: int | None, **fields: Any) -> bytes:
    hello: dict[str, Any] = {
        "slot": at,
        "pid": os.getpid(),
        "token": "abcd1234",
        "protocol": wire.PROTOCOL,
        "schema": served.server.schema,
    }
    hello.update(fields)
    return json.dumps({k: v for k, v in hello.items() if v is not _DROP}).encode()


_DROP = object()


def _raw(served: Any, slot: int) -> tuple[socket.socket, str]:
    """Connect by hand; return the socket and the prefix the worker maps requests under."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    wire.send(sock, Kind.HELLO, 0, _hello(served, slot))
    kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
    assert kind is Kind.WELCOME, payload
    return sock, wire.decode_message(payload, wire.Welcome).request_prefix


def _refused(served: Any, hello: bytes) -> str:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    try:
        wire.send(sock, Kind.HELLO, 0, hello)
        kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
    finally:
        sock.close()
    assert kind is Kind.REFUSED
    return wire.decode_message(payload, wire.Refusal).reason


def _header(tree: Any, *, block: str = "OWN", spans: list[list[int]] | None = None) -> bytes:
    """A request header written by hand; ``OWN`` stands for the session's own input block."""
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


def _closed_by_worker(sock: socket.socket) -> bool:
    sock.settimeout(5)
    try:
        while True:
            if not sock.recv(1 << 16):
                return True
    except OSError:
        return True


def _shm_sizes(prefix: str) -> dict[str, int]:
    sizes = {}
    for path in Path("/dev/shm").glob(f"{prefix}*"):
        # A block can be unlinked between the listing and the stat.
        with contextlib.suppress(FileNotFoundError):
            sizes[path.name] = path.stat().st_size
    return sizes


needs_shm_files = pytest.mark.skipif(
    not Path("/dev/shm").is_dir(), reason="shared blocks are not files on this platform"
)


# -- stepping -------------------------------------------------------------------------------


def test_a_step_round_trips_arrays_through_shared_memory(serve: Any) -> None:
    served: Any = serve(Recorder())
    client = served.client(0)
    frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    out = client.step(frame)
    assert out["slot"] == 0
    assert out["gen"] == 2
    np.testing.assert_array_equal(out["x"], frame * 2)
    # The session owns its copy: the next step cannot change it.
    out["x"][0, 0, 0] = 7
    np.testing.assert_array_equal(client.step(frame)["x"], frame * 2)


def test_the_worker_sees_request_arrays_read_only(serve: Any) -> None:
    class Writes(ModelServer):
        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
            requests[0].payload[0] = 1
            return [None]

    served: Any = serve(Writes())
    with pytest.raises(WorkerError, match="read-only"):
        served.client(0).step(np.zeros(1 << 16, dtype=np.uint8))


@dataclass(frozen=True)
class In:
    frame: np.ndarray
    chunk: int


@dataclass(frozen=True)
class Out:
    frame: np.ndarray
    slot: int


class Typed(ModelServer):
    request_type = In
    result_type = Out
    max_batch = 2

    def step(self, requests: list[ServerRequest[In]]) -> list[Any]:
        out: list[Any] = []
        for r in requests:
            assert type(r.payload) is In
            if r.payload.chunk < 0:
                out.append({"not": "an Out"})
            else:
                out.append(Out(frame=r.payload.frame + r.payload.chunk, slot=r.slot))
        return out


def test_a_typed_request_and_result_cross_as_the_declared_types(serve: Any) -> None:
    served: Any = serve(Typed())
    client = served.client(0, request_type=In, result_type=Out)
    out = client.step(In(frame=np.ones(3), chunk=2))
    assert type(out) is Out
    assert out.slot == 0
    np.testing.assert_array_equal(out.frame, np.full(3, 3.0))


def test_an_output_that_is_not_the_result_type_fails_that_request(serve: Any) -> None:
    served: Any = serve(Typed())
    client = served.client(0, request_type=In, result_type=Out)
    with pytest.raises(WorkerError, match=r"SchemaError: expected a .*Out, got a dict"):
        client.step(In(frame=np.ones(1), chunk=-1))
    assert client.step(In(frame=np.ones(1), chunk=1)).slot == 0


def test_a_request_that_is_not_the_request_type_is_rejected_for_that_session(
    serve: Any,
) -> None:
    served: Any = serve(Typed())
    other = served.client(1, request_type=In, result_type=Out)
    sock, prefix = _raw(served, 0)
    ok = SharedSlot(prefix=f"{prefix}ok_")
    try:
        for bad in ({"frame": 1, "chunk": 1}, {"chunk": 1}, {"frame": np.ones(1), "chunk": True}):
            kind, payload = _step_raw(sock, 1, codec.encode(bad, ok)[0])
            assert kind is Kind.REJECTED
            assert "does not decode" in wire.decode_message(payload, wire.Refusal).reason
        good = {"frame": np.ones(2), "chunk": 1}
        assert _step_raw(sock, 2, codec.encode(good, ok)[0])[0] is Kind.RESULT
    finally:
        ok.close()
        sock.close()
    assert other.step(In(frame=np.ones(1), chunk=0)).slot == 1


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


def test_the_settings_the_server_is_given_are_what_it_batches_by_and_welcomes_with(
    serve: Any,
) -> None:
    worker = Recorder()
    settings = ServerSettings.resolve(Recorder, {"max_batch": 1, "shm_input_bytes": 2 << 20})
    served: Any = serve(worker, settings=settings)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    try:
        wire.send(sock, Kind.HELLO, 0, _hello(served, 5))
        kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
    finally:
        sock.close()
    assert kind is Kind.WELCOME
    welcome = wire.decode_message(payload, wire.Welcome)
    assert welcome.max_batch == 1
    assert welcome.max_request_bytes == Recorder.max_request_bytes
    assert welcome.shm_slot_bytes == (2 << 20) + Recorder.shm_output_bytes
    assert served.server.settings == settings
    clients = [served.client(slot) for slot in range(3)]
    worker.gate.clear()
    threads = [threading.Thread(target=c.step, args=(np.ones(2),)) for c in clients]
    threads[0].start()
    worker.entered.wait(5)
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.1)
    worker.gate.set()
    for thread in threads:
        thread.join(5)
    assert all(len(batch) == 1 for batch in worker.batches)


def test_a_session_that_joins_mid_step_waits_for_the_next_step(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    first = served.client(0)
    worker.gate.clear()
    running = threading.Thread(target=first.step, args=(np.ones(2),))
    running.start()
    worker.entered.wait(5)
    joined: list[ModelClient] = []
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
    assert ("closed", 0, 2) not in worker.events
    worker.gate.set()
    _until(lambda: ("closed", 0, 2) in worker.events)
    assert staying.step(np.ones(2))["slot"] == 1


def test_an_output_that_is_an_exception_fails_that_request_alone(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    failing, fine, holder = served.client(0), served.client(1), served.client(2)
    worker.gate.clear()
    results: dict[str, Any] = {}
    hold = threading.Thread(target=holder.step, args=(np.ones(1),))
    hold.start()
    worker.entered.wait(5)
    threads = [
        _run_into(results, "failing", failing.step, {"fail": True}),
        _run_into(results, "fine", fine.step, np.ones(3)),
    ]
    time.sleep(0.1)
    worker.gate.set()
    for thread in [hold, *threads]:
        thread.join(5)
    assert [0, 1] in worker.batches
    failure = results["failing"]
    assert type(failure) is WorkerError
    assert failure.type_name == "test_worker_server.BoomError"
    assert failure.message == "slot 0 failed"
    assert "BoomError" in failure.worker_traceback
    np.testing.assert_array_equal(results["fine"]["x"], np.ones(3) * 2)
    # The failure ended nothing: the same client steps again.
    np.testing.assert_array_equal(failing.step(np.ones(2))["x"], np.ones(2) * 2)


def test_a_step_that_raises_fails_every_request_of_its_batch_and_runs_once(serve: Any) -> None:
    class Poisoned(Recorder):
        def load(self, config_path: Path | None) -> None:
            super().load(config_path)
            self.calls = 0

        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
            self.calls += 1
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
    threads = [
        _run_into(results, 0, clients[0].step, np.ones(2)),
        _run_into(results, 1, clients[1].step, {"raise": True}),
        _run_into(results, 2, clients[2].step, np.ones(3)),
    ]
    time.sleep(0.2)
    worker.gate.set()
    for thread in [hold, *threads]:
        thread.join(10)
    # One step for the holder and one for the batch: nothing is stepped again.
    assert worker.calls == 2
    for slot in range(3):
        failure = results[slot]
        assert isinstance(failure, BatchFailed), failure
        assert failure.type_name == "test_worker_server.BoomError"
        assert failure.message == "the whole step failed"
        assert "BoomError" in failure.worker_traceback
    # The worker goes on, batching as before.
    assert clients[0].step(np.ones(2))["slot"] == 0


def test_a_step_returning_the_wrong_count_fails_every_request(serve: Any) -> None:
    class Short(ModelServer):
        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
            return []

    served: Any = serve(Short())
    with pytest.raises(BatchFailed, match="0 outputs for 1 requests"):
        served.client(0).step(1)


def test_an_output_that_does_not_travel_fails_that_request(serve: Any) -> None:
    class Unsendable(ModelServer):
        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
            return [threading.Lock()]

    served: Any = serve(Unsendable())
    with pytest.raises(WorkerError, match="TypeError: a lock does not travel") as raised:
        served.client(0).step(1)
    assert type(raised.value) is WorkerError
    with pytest.raises(WorkerError, match="TypeError"):
        served.client(1).step(1)


def test_an_exception_whatever_it_holds_arrives_by_name_and_message(serve: Any) -> None:
    class OddError(Exception):
        def __init__(self, lock: Any) -> None:
            super().__init__("odd")
            self.lock = lock

    class Raises(ModelServer):
        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
            return [OddError(threading.Lock())]

    served: Any = serve(Raises())
    with pytest.raises(WorkerError, match="OddError: odd"):
        served.client(0).step(1)


def test_an_exception_that_cannot_be_printed_fails_its_request_alone(serve: Any) -> None:
    class UnprintableError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no str")

    class Unprintable(ModelServer):
        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
            return [UnprintableError() if r.slot == 0 else 1 for r in requests]

    served: Any = serve(Unprintable())
    first, second = served.client(0), served.client(1)
    with pytest.raises(WorkerError, match="Unprintable"):
        first.step({"x": 1}, timeout=5)
    assert served.thread.is_alive()
    assert second.step({"x": 1}, timeout=5) == 1


# -- what a session sends -------------------------------------------------------------------


def test_an_oversize_request_is_refused_for_that_session_alone(serve: Any) -> None:
    class Small(Recorder):
        max_request_bytes = 1 << 16

    served: Any = serve(Small())
    big, small = served.client(0), served.client(1)
    with pytest.raises(WorkerRejected, match="over the worker's 65536 limit"):
        big.step(np.zeros(1 << 17, dtype=np.uint8))
    assert small.step(np.ones(4))["slot"] == 1
    assert big.step(np.ones(4))["slot"] == 0


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
        assert "over the 4096 limit" in wire.decode_message(reason, wire.Refusal).reason
    finally:
        slot.close()
        sock.close()


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (b"not json", "not JSON"),
        # A pickle that would run code if anything unpickled it.
        (pickle.dumps(shutil.rmtree), "not JSON"),
        (json.dumps({"v": 2, "block": "x", "spans": [], "tree": 1}).encode(), "version 1"),
        (_header({"o": 1}), "unknown or malformed tag"),
        (_header({"a": ["|O", [1], 0]}, spans=[[0, 8]]), "does not travel"),
        (_header({"a": ["<f8", [3], 0]}, spans=[[0, 8]]), "needs 24 bytes"),
        (_header({"a": ["V8", [1], 0]}, spans=[[0, 8]]), "does not travel"),
        (_header({"b": 5}, spans=[[0, 1]]), "span 5"),
        (_header(_deep(80)), "deeper than"),
        (_header(1, block="rrother_0", spans=[[0, 4]]), "not one of this client's"),
        (_header(1, spans=[[-1, 4]]), "two non-negative integers"),
        (_header({"a": ["<u1", [2**62, 2**62, 0], 0]}, spans=[[0, 0]]), "needs"),
    ],
)
def test_a_malformed_request_is_rejected_and_the_session_goes_on(
    serve: Any, header: bytes, reason: str
) -> None:
    served: Any = serve(Recorder())
    other = served.client(1)
    sock, prefix = _raw(served, 0)
    ok = SharedSlot(prefix=f"{prefix}ok_")
    header = header.replace(b'"OWN"', json.dumps(f"{prefix}in").encode())
    try:
        kind, payload = _step_raw(sock, 1, header)
        assert kind is Kind.REJECTED
        assert reason in wire.decode_message(payload, wire.Refusal).reason
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
        assert "past the block" in wire.decode_message(payload, wire.Refusal).reason
        bad_name = _header(1, block=f"{prefix}in\\u0000x", spans=[[0, 1]])
        assert _step_raw(sock, 2, bad_name)[0] is Kind.REJECTED
    finally:
        slot.close()
        sock.close()


def test_a_step_frame_past_the_header_cap_closes_that_connection_unread(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    other = served.client(1)
    sock, _ = _raw(served, 0)
    # One byte past what a STEP frame may hold, announced and never sent.
    length = wire.TIMEOUT.size + codec.MAX_HEADER_BYTES + 1
    sock.sendall(wire.HEADER.pack(length, Kind.STEP, 1))
    assert _closed_by_worker(sock)
    sock.close()
    _until(lambda: ("closed", 0, 1) in worker.events)
    assert other.step(np.ones(2))["slot"] == 1


def test_an_oversize_control_frame_is_refused_before_it_is_read(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    sock, _ = _raw(served, 0)
    # Announce a 100 MB signal and send none of it: the worker must not wait for it.
    sock.sendall(wire.HEADER.pack(100 << 20, Kind.SIGNAL, 0))
    assert _closed_by_worker(sock)
    sock.close()
    _until(lambda: ("closed", 0, 1) in worker.events)


@pytest.mark.parametrize(
    "frame",
    [b"\xff" * 13, wire.encode(Kind.RESULT, 1, b""), wire.encode(Kind.HELLO, 1, b"{}")],
    ids=["garbage", "result", "second-hello"],
)
def test_a_frame_that_breaks_the_protocol_closes_that_connection_alone(
    serve: Any, frame: bytes
) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    other = served.client(1)
    sock, _ = _raw(served, 0)
    sock.sendall(frame)
    assert _closed_by_worker(sock)
    sock.close()
    _until(lambda: ("closed", 0, 1) in worker.events)
    assert other.step(np.ones(2))["slot"] == 1


@dataclass(frozen=True)
class Preempt:
    reason: str


class Signalled(Recorder):
    signals = (Preempt,)

    def load(self, config_path: Path | None) -> None:
        super().load(config_path)
        self.received: list[tuple[int, Any]] = []

    def signal(self, slot: int, signal: Any) -> None:
        with self.lock:
            self.received.append((slot, signal))


def _signal_frame(tag: int, body: bytes) -> bytes:
    return wire.encode(Kind.SIGNAL, 0, wire.SIGNAL_TAG.pack(tag) + body)


def test_a_declared_signal_reaches_the_worker_as_its_type(serve: Any) -> None:
    worker = Signalled()
    served: Any = serve(worker)
    sock, _ = _raw(served, 2)
    try:
        sock.sendall(_signal_frame(0, json.dumps(codec.to_tree({"reason": "barge"})).encode()))
        _until(lambda: worker.received == [(2, Preempt("barge"))])
        assert type(worker.received[0][1]) is Preempt
    finally:
        sock.close()


def test_a_signal_of_a_connection_that_left_does_not_reach_its_slot(serve: Any) -> None:
    from reactor_runtime.model_server.server import _Conn

    worker = Signalled()
    served: Any = serve(worker)
    gone = _Conn(
        slot=3, peer_pid=0, sock=None, carrier=None, request_prefix="", transport={}, hello_id=0
    )  # type: ignore[arg-type]
    gone.opened, gone.closed = True, True
    body = json.dumps(codec.to_tree({"reason": "barge"})).encode()
    served.server._signal(gone, wire.SIGNAL_TAG.pack(0) + body)
    assert (
        worker.received == []
    )  # a buffered signal of a closed connection is not the next session's
    gone.closed = False
    served.server._signal(gone, wire.SIGNAL_TAG.pack(0) + body)
    assert worker.received == [(3, Preempt("barge"))]


@pytest.mark.parametrize(
    ("frame", "reason"),
    [
        (_signal_frame(99, b"{}"), "tag 99, which the worker does not declare"),
        (_signal_frame(1, b"{}"), "tag 1, which the worker does not declare"),
        (_signal_frame(0, b"not json"), "Preempt signal does not decode"),
        (_signal_frame(0, json.dumps({"d": [["why", "x"]]}).encode()), "does not decode"),
        (_signal_frame(0, json.dumps({"d": [["reason", 1]]}).encode()), "must be str"),
        (_signal_frame(0, pickle.dumps(shutil.rmtree)), "does not decode"),
        (wire.encode(Kind.SIGNAL, 0, b"\x00"), "has no tag"),
    ],
    ids=["tag-99", "tag-1", "not-json", "wrong-field", "wrong-type", "pickle", "no-tag"],
)
def test_an_undeclared_or_malformed_signal_refuses_that_session_alone(
    serve: Any, frame: bytes, reason: str
) -> None:
    worker = Signalled()
    served: Any = serve(worker)
    other = served.client(1, signals=(Preempt,))
    sock, _ = _raw(served, 0)
    try:
        sock.sendall(frame)
        kind, _, payload = wire.recv(sock, wire.CONTROL_LIMIT)
        assert kind is Kind.REFUSED
        assert reason in wire.decode_message(payload, wire.Refusal).reason
        assert _closed_by_worker(sock)
    finally:
        sock.close()
    assert worker.received == []
    _until(lambda: ("closed", 0, 1) in worker.events)
    assert other.step(np.ones(2))["slot"] == 1


# -- accepting sessions ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"token": _DROP}, "missing token"),
        ({"token": "not hex!"}, "valid token"),
        ({"extra": 1}, "unexpected extra"),
        ({"pid": "1"}, "pid must be int"),
        ({"slot": -1}, "non-negative"),
        ({"slot": True}, "slot must be int"),
        ({"schema": "0000"}, "differ from the worker's"),
        ({"protocol": 1}, "protocol 1 and this worker speaks protocol"),
        ({"protocol": _DROP}, "protocol None"),
    ],
)
def test_a_malformed_or_mismatched_hello_is_refused(
    serve: Any, fields: dict[str, Any], reason: str
) -> None:
    served: Any = serve(Recorder())
    assert reason in _refused(served, _hello(served, 0, **fields))
    assert served.client(0).step(1)["slot"] == 0


def test_a_hello_that_is_not_a_json_object_closes_the_connection(serve: Any) -> None:
    served: Any = serve(Recorder())
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    try:
        wire.send(sock, Kind.HELLO, 0, b"[1, 2]")
        assert _closed_by_worker(sock)
    finally:
        sock.close()


def test_a_refusing_session_opened_refuses_the_connect(serve: Any) -> None:
    class Refuses(Recorder):
        def session_opened(self, slot: int, generation: int) -> None:
            if slot == 9:
                raise ValueError("no room for slot 9")

    served: Any = serve(Refuses())
    with pytest.raises(WorkerRejected, match="no room for slot 9"):
        served.client(9)
    assert served.client(1).step(1)["slot"] == 1


def test_the_worker_assigns_the_lowest_free_slot(serve: Any) -> None:
    served: Any = serve(Recorder())
    first = served.client(0)
    auto = served.client(None)
    assert first.slot == 0
    assert auto.slot == 1
    assert auto.step(1)["slot"] == 1


def test_a_worker_with_a_session_count_serves_only_its_slots(serve: Any) -> None:
    served: Any = serve(Recorder(), max_sessions=2)
    served.client(1)
    # A slot no session of the deployment has, asked for or assigned, is refused.
    with pytest.raises(
        WorkerRejected, match="slot 2 is out of range: this worker serves slots 0 to 1"
    ):
        served.client(2)
    served.client(None)  # slot 0, the one left
    with pytest.raises(WorkerRejected, match="slot 2 is out of range"):
        served.client(None)


def test_a_worker_without_a_session_count_serves_any_slot(serve: Any) -> None:
    served: Any = serve(Recorder())
    assert served.client(7).slot == 7


def test_a_slot_held_by_a_live_process_is_refused(serve: Any) -> None:
    served: Any = serve(Recorder())
    served.client(0)
    with pytest.raises(WorkerRejected, match="slot 0 is held by process"):
        served.client(0)


def test_a_held_slot_is_refused_even_when_the_peer_pid_is_unknown(
    serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reactor_runtime.model_server import server as server_module

    # Containers of a pod do not share a PID namespace: the peer reads as pid 0.
    monkeypatch.setattr(server_module, "_peer_pid", lambda sock: 0)
    served: Any = serve(Recorder())
    first = served.client(0)
    with pytest.raises(WorkerRejected, match="slot 0 is held"):
        served.client(0)
    assert first.step(1)["slot"] == 0
    # A holder whose socket is gone is closed, and the slot opens again.
    sock, _ = _raw(served, 3)
    sock.close()
    assert served.client(3).step(1)["slot"] == 3


def test_a_slot_is_closed_before_it_is_opened_again(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    first = served.client(0, reset=False)
    first.close()
    second = served.client(0, reset=False)
    assert second.generation == 2
    _until(lambda: len(worker.events) == 3)
    assert worker.events == [("opened", 0, 1), ("closed", 0, 1), ("opened", 0, 2)]


def test_reset_closes_and_reopens_the_slot_with_a_new_generation(serve: Any) -> None:
    worker = Recorder()
    served: Any = serve(worker)
    client = served.client(0, reset=False)
    assert client.generation == 1
    assert client.reset() == 2
    assert client.step(1)["gen"] == 2
    assert worker.events == [("opened", 0, 1), ("closed", 0, 1), ("opened", 0, 2)]


def test_a_session_the_worker_has_no_memory_for_is_refused(tmp_path: Path) -> None:
    from reactor_runtime.distributed import SharedSlotAllocationFailed

    class Full(UnixShmTransport):
        def worker_carrier(self, *args: Any, **kwargs: Any) -> Any:
            raise SharedSlotAllocationFailed("/dev/shm has 0 bytes free")

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
            ModelClient.connect(socket_path=path, slot=0, connect_timeout=5)
    finally:
        server.stop()
        thread.join(5)


def test_a_stopping_worker_closes_new_connections_without_a_refusal(tmp_path: Path) -> None:
    worker = Recorder()
    worker.load(None)
    server = WorkerServer(worker, tmp_path / "x.sock", shm_prefix="rrx_")
    server.stop()
    session, end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with session, end:
        hello = wire.Hello(
            slot=0, pid=1, token="abcd1234", protocol=wire.PROTOCOL, schema=server.schema
        )
        wire.send(session, Kind.HELLO, 0, wire.encode_message(hello))
        assert server._hello(end) is None
        # Nothing was sent: a session waiting in connect() goes on waiting.
        session.setblocking(False)
        with pytest.raises(BlockingIOError):
            session.recv(1)


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("max_batch", 0),
        ("batch_window", -1.0),
        ("max_request_bytes", 0),
        ("shm_input_bytes", 0),
    ],
)
def test_a_bad_policy_is_refused_at_start(tmp_path: Path, attr: str, value: Any) -> None:
    good = ServerSettings.resolve(Recorder, {})
    bad = ServerSettings(**{**good.__dict__, attr: value})
    with pytest.raises(ValueError, match=attr):
        WorkerServer(Recorder(), tmp_path / "x.sock", shm_prefix="rrx_", settings=bad)


def test_a_worker_refuses_to_start_without_its_socket_directory(tmp_path: Path) -> None:
    server = WorkerServer(Recorder(), tmp_path / "gone" / "x.sock", shm_prefix="rrx_")
    with pytest.raises(WorkerUnreachable, match="does not exist"):
        server.bind()


# -- shared memory --------------------------------------------------------------------------


@needs_shm_files
def test_a_sessions_blocks_are_made_at_join_and_gone_after_it_leaves(serve: Any) -> None:
    class Sized(Recorder):
        shm_input_bytes = 3 << 20
        shm_output_bytes = 5 << 20

    served: Any = serve(Sized())
    assert Sized.shm_slot_size() == 8 << 20
    client = served.client(6, reset=False)
    # Before any step: the input block, and the output block, at their sizes.
    blocks = _shm_sizes(served.prefix)
    inputs = [n for n in blocks if n.startswith(f"{served.prefix}s6_") and n.endswith("_in")]
    outputs = [n for n in blocks if n.startswith(f"{served.prefix}w6c")]
    assert [blocks[n] for n in inputs] == [3 << 20]
    assert [blocks[n] for n in outputs] == [5 << 20]
    client.close()
    _until(lambda: not [n for n in _shm_sizes(served.prefix) if "s6_" in n or "w6c" in n])


@needs_shm_files
def test_closing_releases_every_shared_block(serve: Any) -> None:
    served: Any = serve(Recorder())
    client = served.client(0)
    client.step(np.ones(1 << 16, dtype=np.uint8))

    def blocks() -> list[str]:
        # The worker's probe block lives as long as the worker.
        return [n for n in _shm_sizes(served.prefix) if not n.endswith("probe")]

    assert blocks()
    client.close()
    _until(lambda: not blocks())


@needs_shm_files
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


def test_a_request_that_outgrows_its_block_still_travels(serve: Any) -> None:
    class Small(Recorder):
        shm_input_bytes = 4096

    served: Any = serve(Small())
    client = served.client(0)
    big = np.arange(1 << 16, dtype=np.float64)
    np.testing.assert_array_equal(client.step(big)["x"], big * 2)
    np.testing.assert_array_equal(client.step(np.ones(4))["x"], np.ones(4) * 2)


@needs_shm_files
def test_an_output_that_outgrows_its_block_grows_it_in_place(serve: Any) -> None:
    class Sized(ModelServer):
        shm_output_bytes = 4096

        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
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


# -- a session that stops reading -----------------------------------------------------------


class Echo(ModelServer):
    def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
        return [r.payload for r in requests]


def test_a_session_that_stops_reading_is_cut_off_and_stalls_nobody(serve: Any) -> None:
    served: Any = serve(Echo())
    good = served.client(1)
    sock, _ = _raw(served, 0)
    body = wire.TIMEOUT.pack(30.0) + b'{"v":1,"block":"x","spans":[],"tree":1}'
    sock.setblocking(False)
    sent = 0
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            sock.send(wire.encode(Kind.STEP, sent + 1, body))
            sent += 1
        except BlockingIOError:
            time.sleep(0.01)
        except OSError:
            break
    try:
        # The flooding session's answers back up until it is cut off; the other
        # session is served throughout.
        for _ in range(3):
            assert good.step({"x": 1}, timeout=10) == {"x": 1}
    finally:
        sock.close()


def test_a_session_that_does_not_take_its_answer_holds_the_worker_five_seconds_at_most(
    serve: Any,
) -> None:
    class BigAnswer(ModelServer):
        def step(self, requests: list[ServerRequest[Any]]) -> list[Any]:
            return [b"x" * (64 << 20) if r.payload == "big" else r.payload for r in requests]

    served: Any = serve(BigAnswer())
    good = served.client(1)
    sock, _ = _raw(served, 0)
    # Asks for a large answer, then reads none of it. Its bytes are in shared
    # memory; its header is small, so the cut-off is the send timeout's.
    body = wire.TIMEOUT.pack(30.0) + b'{"v":1,"block":"x","spans":[],"tree":"big"}'
    wire.send(sock, Kind.STEP, 1, body)
    time.sleep(0.2)
    started = time.monotonic()
    try:
        assert good.step("small", timeout=20) == "small"
        assert time.monotonic() - started < 8
    finally:
        sock.close()


def test_a_dead_worker_ends_every_session_with_its_reason(serve: Any) -> None:
    served: Any = serve(Recorder())
    clients = [served.client(slot) for slot in range(2)]
    served.halt()
    for client in clients:
        with pytest.raises(WorkerLost, match=r"^worker_died: "):
            client.step(1)
