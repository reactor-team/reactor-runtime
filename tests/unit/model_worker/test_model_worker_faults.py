"""Faults one session causes stay with it: slow readers, odd errors, failing input."""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from typing import Any

import numpy as np
import pytest

from reactor_runtime.model_worker import (
    ModelWorker,
    StepTimedOut,
    WorkerError,
    WorkerRejected,
    WorkerRequest,
    wire,
)
from reactor_runtime.model_worker.wire import Kind


class Returner(ModelWorker):
    max_batch = 4
    batch_window = 0.3

    def load(self, config_path: Any) -> None:
        self.calls: list[list[int]] = []

    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        self.calls.append([r.slot for r in requests])
        return [ValueError(f"no face {r.slot}") for r in requests]


def test_errors_a_step_returns_are_not_stepped_again(serve: Any) -> None:
    served: Any = serve(Returner())
    clients = [served.client(0), served.client(1)]
    errors: list[Exception] = []

    def go(client: Any) -> None:
        try:
            client.step({"x": 1})
        except WorkerError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(c,)) for c in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert [sorted(call) for call in served.worker.calls] == [[0, 1]]
    assert len(errors) == 2


class Slow(ModelWorker):
    max_request_bytes = 1 << 16

    def load(self, config_path: Any) -> None:
        self.seen: list[int] = []

    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        request = requests[0]
        if request.payload.get("slow"):
            time.sleep(1.0)
        self.seen.append(int(request.payload["a"][0]))
        return [0]


def test_a_rejected_request_does_not_free_an_abandoned_ones_block(serve: Any) -> None:
    served: Any = serve(Slow())
    client = served.client(0)
    with pytest.raises(StepTimedOut):
        client.step({"a": np.full(1000, 1, np.uint8), "slow": True}, timeout=0.2)
    # Its buffers fit the limit; with its header it does not.
    with pytest.raises(WorkerRejected):
        client.step({"a": np.full((1 << 16) - 4, 9, np.uint8)}, timeout=5)
    client.step({"a": np.full(1000, 2, np.uint8)}, timeout=5)
    assert served.worker.seen[0] == 1


class Echo(ModelWorker):
    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        return [r.payload for r in requests]


def test_a_session_that_stops_reading_is_cut_off_and_stalls_nobody(serve: Any) -> None:
    served: Any = serve(Echo())
    good = served.client(1)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    wire.send(sock, Kind.HELLO, 0, wire.to_json({"slot": 0, "token": "abcd"}))
    kind, _, _ = wire.recv(sock, wire.CONTROL_LIMIT)
    assert kind is Kind.WELCOME
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


class UnprintableError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no str")


class UnprintableWorker(ModelWorker):
    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        return [UnprintableError() if r.slot == 0 else 1 for r in requests]


def test_an_exception_that_cannot_be_printed_fails_its_request_alone(serve: Any) -> None:
    served: Any = serve(UnprintableWorker())
    first, second = served.client(0), served.client(1)
    with pytest.raises(WorkerError, match="Unprintable"):
        first.step({"x": 1}, timeout=5)
    assert served.thread.is_alive()
    assert second.step({"x": 1}, timeout=5) == 1


class Raises(ModelWorker):
    def load(self, config_path: Any) -> None:
        self.broken = False

    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        if self.broken or requests[0].slot == 0:
            raise RuntimeError("bad input")
        return [1 for _ in requests]


def test_one_session_whose_steps_raise_does_not_make_the_worker_failing(serve: Any) -> None:
    worker = Raises()
    served: Any = serve(worker)
    bad, other, idle = served.client(0), served.client(1), served.client(2)
    for _ in range(4):
        with pytest.raises(WorkerError):
            bad.step(1)
        assert other.step(1) == 1
    assert not served.server.failing(3)
    # A broken GPU: every session that steps raises, while another sits idle.
    worker.broken = True
    for client in (bad, other, bad, other):
        with pytest.raises(WorkerError):
            client.step(1)
    assert idle.available
    assert served.server.failing(3)


class SlotZeroRaises(ModelWorker):
    max_batch = 4
    batch_window = 0.05

    def load(self, config_path: Any) -> None:
        self.batches: list[list[int]] = []

    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        self.batches.append([r.slot for r in requests])
        if any(r.slot == 0 for r in requests):
            raise RuntimeError("bad input")
        return [1 for _ in requests]


def test_a_slot_that_raises_alone_is_stepped_apart_from_its_batch_mates(serve: Any) -> None:
    worker = SlotZeroRaises()
    served: Any = serve(worker)
    clients = [served.client(slot) for slot in range(3)]

    def run(client: Any, times: int) -> None:
        for _ in range(times):
            with contextlib.suppress(WorkerError):
                client.step(1)

    threads = [threading.Thread(target=run, args=(c, 6)) for c in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    # Once slot 0 has raised alone, it is never batched with the others again.
    mixed = [i for i, b in enumerate(worker.batches) if 0 in b and len(b) > 1]
    assert len(mixed) <= 1, worker.batches
    assert any(b == [0] for b in worker.batches)


class BigAnswer(ModelWorker):
    def step(self, requests: list[WorkerRequest]) -> list[Any]:
        return ["x" * (4 << 20) if r.payload == "big" else r.payload for r in requests]


def test_a_session_that_does_not_take_a_large_answer_holds_the_worker_five_seconds_at_most(
    serve: Any,
) -> None:
    served: Any = serve(BigAnswer())
    good = served.client(1)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(served.path))
    wire.send(sock, Kind.HELLO, 0, wire.to_json({"slot": 0, "token": "abcd"}))
    kind, _, _ = wire.recv(sock, wire.CONTROL_LIMIT)
    assert kind is Kind.WELCOME
    # Asks for a 4 MB inline answer, then reads none of it.
    body = wire.TIMEOUT.pack(30.0) + b'{"v":1,"block":"x","spans":[],"tree":"big"}'
    wire.send(sock, Kind.STEP, 1, body)
    time.sleep(0.2)
    started = time.monotonic()
    try:
        assert good.step("small", timeout=20) == "small"
        assert time.monotonic() - started < 8
    finally:
        sock.close()
