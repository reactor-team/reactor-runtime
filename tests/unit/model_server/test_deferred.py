"""Exercise deferred work over threaded links, with bounded waits for every answer."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from reactor_runtime.core import HealthStatus
from reactor_runtime.model_server import (
    DEFERRED,
    BatchFailed,
    ModelServer,
    ModelServerStopping,
    ServerRequest,
    ServerSettings,
    StepExpired,
    StepStillRunning,
)
from reactor_runtime.model_server.protocol import (
    Closed,
    Hello,
    Reset,
    ResetDone,
    Result,
    Step,
    Welcome,
    error_of,
)
from reactor_runtime.model_server.server_core import ModelServerCore, ServerState


class _Link:
    def __init__(self, incoming: queue.Queue[Any], outgoing: queue.Queue[Any]) -> None:
        self.incoming = incoming
        self.outgoing = outgoing

    def send(self, message: Any) -> None:
        self.outgoing.put(message)

    def receive(self) -> Any:
        message = self.incoming.get(timeout=5)
        if message is Closed:
            raise Closed()
        return message

    def close(self) -> None:
        self.incoming.put(Closed)
        self.outgoing.put(Closed)


class _Transport:
    def __init__(self) -> None:
        self.pending: queue.Queue[_Link | None] = queue.Queue()

    def listen(self, address: str) -> _Transport:
        return self

    def accept(self) -> _Link:
        link = self.pending.get()
        if link is None:
            raise OSError("closed")
        return link

    def close(self) -> None:
        self.pending.put(None)

    def connect(self, address: str, slot: int, *, timeout: float | None = None) -> _Link:
        incoming: queue.Queue[Any] = queue.Queue()
        outgoing: queue.Queue[Any] = queue.Queue()
        self.pending.put(_Link(outgoing, incoming))
        return _Link(incoming, outgoing)


class _Model(ModelServer[int, int]):
    max_batch = 4

    def __init__(self) -> None:
        self.requests: queue.Queue[list[ServerRequest[int]]] = queue.Queue()
        self.closed: queue.Queue[tuple[int, int]] = queue.Queue()
        self.release = threading.Event()
        self.release.set()
        self.entered = threading.Event()
        self.crash = False
        self.early = False

    def step(self, requests: list[ServerRequest[int]]) -> list[Any]:
        self.requests.put(requests)
        self.entered.set()
        assert self.release.wait(5)
        if self.crash:
            raise RuntimeError("step failed")
        if self.early:
            with ThreadPoolExecutor() as executor:
                executor.submit(self.answer, requests[0], 123).result(timeout=5)
        return [DEFERRED if r.payload >= 0 else r.payload for r in requests]

    def session_closed(self, slot: int, generation: int) -> None:
        self.closed.put((slot, generation))


class _Rig:
    def __init__(self) -> None:
        self.model = _Model()
        self.transport = _Transport()
        self.server = ModelServerCore(
            self.model,
            self.transport,
            "unused",
            settings=ServerSettings(4, 0.0, 4096, 4096),
            max_sessions=4,
        )
        asyncio.run(self.server.start())

    def session(self, slot: int = 0) -> _Link:
        link = self.transport.connect("unused", slot)
        link.send(Hello(slot=slot))
        assert isinstance(link.receive(), Welcome)
        return link

    def step(self, link: _Link, request_id: int = 1, payload: int = 1) -> ServerRequest[int]:
        link.send(Step(request_id=request_id, timeout=None, payload=payload))
        requests = self.model.requests.get(timeout=5)
        assert len(requests) == 1
        return requests[0]

    def wait_deferred(self, count: int = 1) -> None:
        with self.server._cond:
            assert self.server._cond.wait_for(
                lambda: sum(p.deferred for p in self.server._pending.values()) == count, timeout=5
            )


@pytest.fixture
def rig() -> Iterator[_Rig]:
    rig = _Rig()
    yield rig
    rig.model.release.set()
    asyncio.run(rig.server.stop())
    assert rig.server._step_thread is not None
    assert not rig.server._step_thread.is_alive()
    assert not rig.server._pending


def test_deferred_answers_allow_other_sessions_but_keep_the_session_busy(rig: _Rig) -> None:
    first, second = rig.session(0), rig.session(1)
    one = rig.step(first)
    rig.wait_deferred()
    first.send(Step(request_id=2, timeout=None, payload=2))
    assert isinstance(error_of(first.receive()), StepStillRunning)
    two = rig.step(second)
    rig.wait_deferred(2)
    with ThreadPoolExecutor() as executor:
        executor.submit(rig.model.answer, two, 20).result(timeout=5)
        executor.submit(rig.model.answer, one, 10).result(timeout=5)
    assert second.receive() == Result(1, 20)
    assert first.receive() == Result(1, 10)
    three = rig.step(first, 3)
    rig.model.answer(three, 30)
    assert first.receive() == Result(3, 30)


def test_exception_output_fails_only_its_deferred_request(rig: _Rig) -> None:
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    rig.model.answer(request, ValueError("bad input"))
    assert str(error_of(link.receive())) == "bad input"


def test_reset_frees_the_slot_and_late_answers_do_not_release_new_work(rig: _Rig) -> None:
    link = rig.session()
    old = rig.step(link)
    rig.wait_deferred()
    link.send(Reset(request_id=2, session_id="new"))
    assert isinstance(error_of(link.receive()), StepExpired)
    assert link.receive() == ResetDone(2, 2)
    assert rig.model.closed.get(timeout=5) == (0, 1)
    new = rig.step(link, 3)
    rig.wait_deferred()
    rig.model.answer(old, 99)
    link.send(Step(request_id=4, timeout=None, payload=4))
    assert isinstance(error_of(link.receive()), StepStillRunning)
    rig.model.answer(new, 30)
    assert link.receive() == Result(3, 30)


def test_leave_drops_deferred_work_and_slot_reuse_ignores_late_answers(rig: _Rig) -> None:
    link = rig.session()
    old = rig.step(link)
    rig.wait_deferred()
    link.close()
    assert rig.model.closed.get(timeout=5) == (0, 1)
    rig.wait_deferred(0)
    replacement = rig.session()
    new = rig.step(replacement)
    rig.wait_deferred()
    rig.model.answer(old, 99)
    rig.model.answer(new, 10)
    assert replacement.receive() == Result(1, 10)


def test_duplicate_and_non_deferred_answers_are_logged_and_ignored(
    rig: _Rig, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    rig.model.answer(request, 10)
    rig.model.answer(request, 99)
    assert link.receive() == Result(1, 10)
    immediate = rig.step(link, 2, -2)
    assert link.receive() == Result(2, -2)
    rig.model.answer(immediate, 99)
    rig.model.answer(ServerRequest(0, 1, 0), 99)
    assert caplog.text.count("ignoring an invalid, duplicate, or late deferred answer") == 3
    request = rig.step(link, 3)
    rig.model.answer(request, 30)
    assert link.receive() == Result(3, 30)


def test_answer_without_a_model_server_raises() -> None:
    with pytest.raises(RuntimeError, match="needs the model server that runs this model"):
        _Model().answer(ServerRequest(0, 1, 1), 1)


def test_completion_thread_can_answer_before_step_returns(rig: _Rig) -> None:
    rig.model.early = True
    link = rig.session()
    rig.step(link)
    assert link.receive() == Result(1, 123)


def test_immediate_output_takes_precedence_over_early_answer(
    rig: _Rig, caplog: pytest.LogCaptureFixture
) -> None:
    rig.model.early = True
    link = rig.session()
    rig.step(link, payload=-1)
    assert link.receive() == Result(1, -1)
    assert "ignoring answer for a request with an immediate output" in caplog.text


def test_mixed_batch_releases_only_immediate_sessions(rig: _Rig) -> None:
    first, second, third = rig.session(0), rig.session(1), rig.session(2)
    rig.model.release.clear()
    rig.step(first)
    second.send(Step(request_id=1, timeout=None, payload=-2))
    third.send(Step(request_id=1, timeout=None, payload=3))
    with rig.server._cond:
        assert rig.server._cond.wait_for(lambda: rig.server._policy.waiting == 2, timeout=5)
    rig.model.release.set()
    batch = rig.model.requests.get(timeout=5)
    assert sorted(r.payload for r in batch) == [-2, 3]
    assert second.receive() == Result(1, -2)
    rig.wait_deferred(2)
    rig.model.answer(next(r for r in batch if r.payload == 3), 30)
    assert third.receive() == Result(1, 30)


def test_drain_waits_for_deferred_answer(rig: _Rig) -> None:
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    with ThreadPoolExecutor() as executor:
        drained = executor.submit(asyncio.run, rig.server.drain())
        with rig.server._cond:
            assert rig.server._cond.wait_for(
                lambda: rig.server.state is ServerState.DRAINING, timeout=5
            )
        assert not drained.done()
        rig.model.answer(request, 10)
        assert link.receive() == Result(1, 10)
        drained.result(timeout=5)


async def test_cancelled_drain_then_stop_fails_leftovers(rig: _Rig) -> None:
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(rig.server.drain(), timeout=0.05)
    await rig.server.stop()
    assert isinstance(error_of(link.receive()), ModelServerStopping)
    rig.model.answer(request, 99)


def test_step_failure_fails_only_its_batch_and_keeps_earlier_deferred_requests(rig: _Rig) -> None:
    first, second = rig.session(0), rig.session(1)
    old = rig.step(first)
    rig.wait_deferred()
    rig.model.crash = True
    rig.step(second)
    assert isinstance(error_of(second.receive()), BatchFailed)
    rig.model.answer(old, 99)
    assert first.receive() == Result(1, 99)
    rig.model.crash = False
    new = rig.step(first, 2)
    rig.model.answer(new, 20)
    assert first.receive() == Result(2, 20)


def test_step_loop_crash_fails_deferred_requests(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()

    def crash(now: float, open_sessions: int) -> Any:
        raise RuntimeError("policy failed")

    with rig.server._cond:
        monkeypatch.setattr(rig.server._policy, "take", crash)
        rig.server._cond.notify_all()
    assert isinstance(error_of(link.receive()), ModelServerStopping)
    assert rig.server.health().status is HealthStatus.UNHEALTHY
    rig.model.answer(request, 99)


def test_endurance_probe_counts_deferred_requests_and_their_release(rig: _Rig) -> None:
    from endurance.model_server_launcher import pending_requests

    baseline = pending_requests()
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    assert pending_requests() == baseline + 1
    rig.model.answer(request, 10)
    assert link.receive() == Result(1, 10)
    assert pending_requests() == baseline


def test_concurrent_answers_deliver_exactly_one_result(rig: _Rig) -> None:
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    barrier = threading.Barrier(2)

    def answer(output: int) -> None:
        barrier.wait(timeout=5)
        rig.model.answer(request, output)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(answer, value) for value in (10, 20)]
        for future in futures:
            future.result(timeout=5)
    assert link.receive() in (Result(1, 10), Result(1, 20))
    new = rig.step(link, 2)
    rig.model.answer(new, 30)
    assert link.receive() == Result(2, 30)


def test_sentinel_as_answer_is_ignored_without_releasing_the_request(rig: _Rig) -> None:
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    assert rig.model._answerer is not None
    rig.model._answerer(request, DEFERRED)  # ty: ignore[invalid-argument-type]  # misuse at runtime
    link.send(Step(request_id=2, timeout=None, payload=2))
    assert isinstance(error_of(link.receive()), StepStillRunning)
    rig.model.answer(request, 10)
    assert link.receive() == Result(1, 10)


def test_deferred_answer_uses_the_outbox_failure_path(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    connection = next(iter(rig.server._connections.values()))
    monkeypatch.setattr(connection, "post", lambda message: False)
    rig.model.answer(request, 10)
    with pytest.raises(Closed):
        link.receive()
    assert rig.model.closed.get(timeout=5) == (0, 1)


def test_drain_fails_queued_work_after_deferred_work_finishes(rig: _Rig) -> None:
    first, second = rig.session(0), rig.session(1)
    request = rig.step(first)
    rig.wait_deferred()
    with rig.server._cond:
        rig.server._policy._window = 60
    second.send(Step(request_id=1, timeout=None, payload=2))
    with rig.server._cond:
        assert rig.server._cond.wait_for(lambda: rig.server._policy.waiting == 1, timeout=5)
    with ThreadPoolExecutor() as executor:
        drained = executor.submit(asyncio.run, rig.server.drain())
        with rig.server._cond:
            assert rig.server._cond.wait_for(
                lambda: rig.server.state is ServerState.DRAINING, timeout=5
            )
        rig.model.answer(request, 10)
        assert first.receive() == Result(1, 10)
        drained.result(timeout=5)
    assert isinstance(error_of(second.receive()), ModelServerStopping)


def test_leave_during_drain_releases_the_deferred_wait(rig: _Rig) -> None:
    link = rig.session()
    request = rig.step(link)
    rig.wait_deferred()
    with ThreadPoolExecutor() as executor:
        drained = executor.submit(asyncio.run, rig.server.drain())
        with rig.server._cond:
            assert rig.server._cond.wait_for(
                lambda: rig.server.state is ServerState.DRAINING, timeout=5
            )
        link.close()
        drained.result(timeout=5)
    rig.model.answer(request, 10)
