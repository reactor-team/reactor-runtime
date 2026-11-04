"""Exercise step outputs that are futures over threaded links, with bounded waits."""

from __future__ import annotations

import asyncio
import contextlib
import queue
import random
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest

from reactor_runtime.core import HealthStatus
from reactor_runtime.model_server import (
    BatchFailed,
    ModelServer,
    ModelServerStopping,
    ServerRequest,
    ServerSettings,
    StepExpired,
    StepStillRunning,
)
from reactor_runtime.model_server.protocol import (
    Cancel,
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

    def receive(self, timeout: float = 5) -> Any:
        message = self.incoming.get(timeout=timeout)
        if message is Closed:
            raise Closed()
        return message

    def silent(self, wait: float = 0.2) -> bool:
        try:
            self.incoming.get(timeout=wait)
        except queue.Empty:
            return True
        return False

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
    """Return a future for a request with a payload of at least 0, the payload itself otherwise.

    The test resolves the futures. ``start`` marks each future running in
    ``step()``, as a pipeline whose work starts there does.
    """

    max_batch = 4

    def __init__(self) -> None:
        self.requests: queue.Queue[list[ServerRequest[int]]] = queue.Queue()
        self.futures: queue.Queue[Future[int]] = queue.Queue()
        self.closed: queue.Queue[tuple[int, int]] = queue.Queue()
        self.release = threading.Event()
        self.release.set()
        self.crash = False
        self.start = True
        self.done_early = False
        self.close_blocks = threading.Event()
        self.close_blocks.set()

    def step(self, requests: list[ServerRequest[int]]) -> list[Any]:
        self.requests.put(requests)
        assert self.release.wait(5)
        if self.crash:
            raise RuntimeError("step failed")
        outputs: list[Any] = []
        for request in requests:
            if request.payload < 0:
                outputs.append(request.payload)
                continue
            future: Future[int] = Future()
            if self.start:
                assert future.set_running_or_notify_cancel()
            if self.done_early:
                future.set_result(request.payload * 10)
            self.futures.put(future)
            outputs.append(future)
        return outputs

    def session_closed(self, slot: int, generation: int) -> None:
        assert self.close_blocks.wait(5)
        self.closed.put((slot, generation))


class _Rig:
    def __init__(
        self, *, window: float = 0.0, stuck_after: float = 120.0, drain_timeout: float = 30.0
    ) -> None:
        self.model = _Model()
        self.transport = _Transport()
        self.server = ModelServerCore(
            self.model,
            self.transport,
            "unused",
            settings=ServerSettings(4, window, 4096, 4096),
            max_sessions=4,
            stuck_after=stuck_after,
            drain_timeout=drain_timeout,
        )
        asyncio.run(self.server.start())

    def session(self, slot: int = 0) -> _Link:
        link = self.transport.connect("unused", slot)
        link.send(Hello(slot=slot))
        assert isinstance(link.receive(), Welcome)
        return link

    def step(
        self, link: _Link, request_id: int = 1, payload: int = 1, timeout: float | None = None
    ) -> Future[int]:
        """Send a step, wait until the model returned its future, and return the future."""
        link.send(Step(request_id=request_id, timeout=timeout, payload=payload))
        requests = self.model.requests.get(timeout=5)
        assert len(requests) == 1
        future = self.model.futures.get(timeout=5)
        self.wait_running(lambda p: p.future is future)
        return future

    def wait_running(self, match: Any = None, count: int = 1) -> None:
        def running() -> bool:
            items = [p for p in self.server._pending.values() if p.future is not None]
            return len([p for p in items if match is None or match(p)]) == count

        with self.server._cond:
            assert self.server._cond.wait_for(running, timeout=5)

    def drain(self) -> Future[Any]:
        executor = ThreadPoolExecutor(max_workers=1)
        drained = executor.submit(asyncio.run, self.server.drain())
        executor.shutdown(wait=False)
        with self.server._cond:
            assert self.server._cond.wait_for(
                lambda: self.server.state is ServerState.DRAINING, timeout=5
            )
        return drained


@pytest.fixture
def rig(request: pytest.FixtureRequest) -> Iterator[_Rig]:
    rig = _Rig(**getattr(request, "param", {}))
    yield rig
    rig.model.release.set()
    rig.model.close_blocks.set()
    asyncio.run(rig.server.stop())
    assert rig.server._step_thread is not None
    assert not rig.server._step_thread.is_alive()
    assert not rig.server._pending


def test_a_future_keeps_its_session_busy_while_others_step(rig: _Rig) -> None:
    first, second = rig.session(0), rig.session(1)
    one = rig.step(first)
    first.send(Step(request_id=2, timeout=None, payload=2))
    assert isinstance(error_of(first.receive()), StepStillRunning)
    two = rig.step(second)
    with ThreadPoolExecutor() as executor:
        executor.submit(two.set_result, 20).result(timeout=5)
        executor.submit(one.set_result, 10).result(timeout=5)
    assert second.receive() == Result(1, 20)
    assert first.receive() == Result(1, 10)
    three = rig.step(first, 3)
    three.set_result(30)
    assert first.receive() == Result(3, 30)


def test_a_future_that_fails_fails_only_its_request(rig: _Rig) -> None:
    first, second = rig.session(0), rig.session(1)
    one, two = rig.step(first), rig.step(second)
    one.set_exception(ValueError("bad input"))
    two.set_result(20)
    assert str(error_of(first.receive())) == "bad input"
    assert second.receive() == Result(1, 20)


def test_a_future_done_before_step_returns_is_answered(rig: _Rig) -> None:
    rig.model.done_early = True
    link = rig.session()
    link.send(Step(request_id=1, timeout=None, payload=4))
    assert link.receive() == Result(1, 40)
    assert rig.model.futures.get(timeout=5).done()
    rig.model.done_early = False
    rig.step(link, 2).set_result(5)
    assert link.receive() == Result(2, 5)


def test_a_mixed_batch_releases_only_the_immediate_sessions(rig: _Rig) -> None:
    first, second, third = rig.session(0), rig.session(1), rig.session(2)
    rig.model.release.clear()
    first.send(Step(request_id=1, timeout=None, payload=1))
    rig.model.requests.get(timeout=5)
    second.send(Step(request_id=1, timeout=None, payload=-2))
    third.send(Step(request_id=1, timeout=None, payload=3))
    with rig.server._cond:
        assert rig.server._cond.wait_for(lambda: rig.server._policy.waiting == 2, timeout=5)
    rig.model.release.set()
    first_future = rig.model.futures.get(timeout=5)
    batch = rig.model.requests.get(timeout=5)
    assert sorted(r.payload for r in batch) == [-2, 3]
    assert second.receive() == Result(1, -2)
    third_future = rig.model.futures.get(timeout=5)
    rig.wait_running(count=2)
    second.send(Step(request_id=2, timeout=None, payload=-5))
    assert second.receive() == Result(2, -5)
    third.send(Step(request_id=2, timeout=None, payload=-6))
    assert isinstance(error_of(third.receive()), StepStillRunning)
    first_future.set_result(10)
    third_future.set_result(30)
    assert first.receive() == Result(1, 10)
    assert third.receive() == Result(1, 30)


def test_a_timeout_answers_a_running_request_but_holds_its_session_until_done(rig: _Rig) -> None:
    link = rig.session()
    future = rig.step(link, timeout=0.2)
    error = error_of(link.receive())
    assert isinstance(error, StepExpired)
    assert "while it ran" in str(error)
    # Its work started, so cancel() cannot stop it: the session stays busy and
    # the model never has two of its requests running.
    assert future.running()
    link.send(Step(request_id=2, timeout=None, payload=2))
    assert isinstance(error_of(link.receive()), StepStillRunning)
    assert rig.model.requests.empty()
    future.set_result(99)
    assert link.silent()
    rig.step(link, 3).set_result(30)
    assert link.receive() == Result(3, 30)


def test_an_expired_request_finishing_after_a_reset_keeps_the_new_one_busy(rig: _Rig) -> None:
    link = rig.session()
    old = rig.step(link, timeout=0.2)
    assert isinstance(error_of(link.receive()), StepExpired)
    link.send(Reset(request_id=2, session_id="new"))
    assert link.receive() == ResetDone(2, 2)
    new = rig.step(link, 3)
    old.set_result(99)
    link.send(Step(request_id=4, timeout=None, payload=4))
    assert isinstance(error_of(link.receive()), StepStillRunning)
    new.set_result(30)
    assert link.receive() == Result(3, 30)


def test_a_timeout_on_work_not_started_cancels_it_and_frees_the_session(rig: _Rig) -> None:
    rig.model.start = False
    link = rig.session()
    future = rig.step(link, timeout=0.2)
    assert isinstance(error_of(link.receive()), StepExpired)
    assert future.cancelled()
    assert not future.set_running_or_notify_cancel()
    rig.model.start = True
    rig.step(link, 2).set_result(20)
    assert link.receive() == Result(2, 20)


def test_a_cancel_stops_answering_a_running_request_and_holds_its_session(rig: _Rig) -> None:
    link = rig.session()
    future = rig.step(link)
    link.send(Cancel(request_id=1))
    with rig.server._cond:
        assert rig.server._cond.wait_for(
            lambda: all(p.answered for p in rig.server._pending.values()), timeout=5
        )
    link.send(Step(request_id=2, timeout=None, payload=2))
    assert isinstance(error_of(link.receive()), StepStillRunning)
    future.set_result(99)
    assert link.silent()
    rig.step(link, 3).set_result(30)
    assert link.receive() == Result(3, 30)


def test_a_cancel_before_the_work_starts_frees_the_session_at_once(rig: _Rig) -> None:
    rig.model.start = False
    link = rig.session()
    future = rig.step(link)
    link.send(Cancel(request_id=1))
    with rig.server._cond:
        assert rig.server._cond.wait_for(lambda: not rig.server._pending, timeout=5)
    assert future.cancelled()
    assert link.silent()
    rig.model.start = True
    rig.step(link, 2).set_result(20)
    assert link.receive() == Result(2, 20)


@pytest.mark.parametrize("rig", [{"stuck_after": 0.3}], indirect=True)
def test_reset_frees_the_session_and_health_waits_for_the_old_future(rig: _Rig) -> None:
    link = rig.session()
    old = rig.step(link)
    link.send(Reset(request_id=2, session_id="new"))
    assert isinstance(error_of(link.receive()), StepExpired)
    assert link.receive() == ResetDone(2, 2)
    assert rig.model.closed.get(timeout=5) == (0, 1)
    new = rig.step(link, 3)
    time.sleep(0.4)
    health = rig.server.health()
    assert health.status is HealthStatus.UNHEALTHY
    assert "future has not been done" in str(health.detail)
    old.set_result(99)
    new.set_result(30)
    assert link.receive() == Result(3, 30)
    assert link.silent()
    assert rig.server.health().status is HealthStatus.HEALTHY


@pytest.mark.parametrize("rig", [{"stuck_after": 0.3}], indirect=True)
def test_a_future_left_after_its_session_leaves_makes_the_server_unhealthy(rig: _Rig) -> None:
    link = rig.session()
    future = rig.step(link)
    link.close()
    assert rig.model.closed.get(timeout=5) == (0, 1)
    assert rig.server.health().status is HealthStatus.HEALTHY
    time.sleep(0.4)
    assert rig.server.health().status is HealthStatus.UNHEALTHY
    future.set_result(1)
    assert rig.server.health().status is HealthStatus.HEALTHY
    replacement = rig.session()
    rig.step(replacement).set_result(10)
    assert replacement.receive() == Result(1, 10)


@pytest.mark.parametrize("rig", [{"stuck_after": 0.3}], indirect=True)
def test_a_session_closed_that_hangs_makes_the_server_unhealthy(rig: _Rig) -> None:
    link = rig.session()
    rig.model.close_blocks.clear()
    link.close()
    time.sleep(0.4)
    health = rig.server.health()
    assert health.status is HealthStatus.UNHEALTHY
    assert "session_closed()" in str(health.detail)
    rig.model.close_blocks.set()
    assert rig.model.closed.get(timeout=5) == (0, 1)
    assert rig.server.health().status is HealthStatus.HEALTHY


def test_a_late_result_after_a_leave_does_not_reach_the_slot_s_next_session(rig: _Rig) -> None:
    link = rig.session()
    old = rig.step(link)
    link.close()
    assert rig.model.closed.get(timeout=5) == (0, 1)
    replacement = rig.session()
    new = rig.step(replacement)
    old.set_result(99)
    new.set_result(10)
    assert replacement.receive() == Result(1, 10)


def test_drain_waits_for_a_running_future(rig: _Rig) -> None:
    link = rig.session()
    future = rig.step(link)
    drained = rig.drain()
    assert not drained.done()
    future.set_result(10)
    assert link.receive() == Result(1, 10)
    drained.result(timeout=5)


@pytest.mark.parametrize("rig", [{"drain_timeout": 0.3}], indirect=True)
def test_drain_is_bounded_and_fails_what_is_left_with_the_stopping_prefix(rig: _Rig) -> None:
    link = rig.session()
    future = rig.step(link)
    rig.model.start = False
    second = rig.session(1)
    unstarted = rig.step(second)
    started = time.monotonic()
    rig.drain().result(timeout=5)
    assert 0.25 < time.monotonic() - started < 3
    for session in (link, second):
        error = error_of(session.receive())
        assert isinstance(error, ModelServerStopping)
        assert str(error).startswith("model_server_lost:")
    assert unstarted.cancelled()
    # Work already running goes on, and its result is dropped.
    future.set_result(99)


def test_drain_ends_at_a_running_request_s_own_timeout(rig: _Rig) -> None:
    link = rig.session()
    rig.step(link, timeout=0.3)
    drained = rig.drain()
    assert isinstance(error_of(link.receive()), StepExpired)
    drained.result(timeout=5)


def test_a_reset_during_a_drain_is_refused_with_the_stopping_prefix(rig: _Rig) -> None:
    link = rig.session()
    future = rig.step(link)
    drained = rig.drain()
    link.send(Reset(request_id=2, session_id="new"))
    error = error_of(link.receive())
    assert isinstance(error, ModelServerStopping)
    assert str(error).startswith("model_server_lost:")
    future.set_result(10)
    assert link.receive() == Result(1, 10)
    drained.result(timeout=5)


def test_leave_during_a_drain_releases_its_wait(rig: _Rig) -> None:
    link = rig.session()
    future = rig.step(link)
    drained = rig.drain()
    link.close()
    drained.result(timeout=5)
    future.set_result(10)


def test_drain_fails_queued_work_after_the_running_future_is_done(rig: _Rig) -> None:
    first, second, _idle = rig.session(0), rig.session(1), rig.session(2)
    future = rig.step(first)
    with rig.server._cond:
        rig.server._policy._window = 60
    second.send(Step(request_id=1, timeout=None, payload=2))
    with rig.server._cond:
        assert rig.server._cond.wait_for(lambda: rig.server._policy.waiting == 1, timeout=5)
    drained = rig.drain()
    future.set_result(10)
    assert first.receive() == Result(1, 10)
    drained.result(timeout=5)
    assert isinstance(error_of(second.receive()), ModelServerStopping)


async def test_stop_fails_running_requests_and_cancels_their_futures(rig: _Rig) -> None:
    rig.model.start = False
    link = rig.session()
    future = rig.step(link)
    await rig.server.stop()
    assert isinstance(error_of(link.receive()), ModelServerStopping)
    assert future.cancelled()


@pytest.mark.parametrize("rig", [{"window": 0.5}], indirect=True)
def test_a_session_with_a_running_future_does_not_hold_back_the_batch(rig: _Rig) -> None:
    first, second = rig.session(0), rig.session(1)
    future = rig.step(first)
    started = time.monotonic()
    second.send(Step(request_id=1, timeout=None, payload=-2))
    assert second.receive() == Result(1, -2)
    assert time.monotonic() - started < 0.4
    future.set_result(10)
    assert first.receive() == Result(1, 10)


def test_a_failing_step_fails_only_its_batch_and_keeps_earlier_futures(rig: _Rig) -> None:
    first, second = rig.session(0), rig.session(1)
    old = rig.step(first)
    rig.model.crash = True
    second.send(Step(request_id=1, timeout=None, payload=1))
    rig.model.requests.get(timeout=5)
    assert isinstance(error_of(second.receive()), BatchFailed)
    old.set_result(99)
    assert first.receive() == Result(1, 99)
    rig.model.crash = False
    rig.step(first, 2).set_result(20)
    assert first.receive() == Result(2, 20)


def test_a_step_loop_crash_fails_running_requests(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = rig.session()
    future = rig.step(link)

    def crash(now: float, open_sessions: int) -> Any:
        raise RuntimeError("policy failed")

    with rig.server._cond:
        monkeypatch.setattr(rig.server._policy, "take", crash)
        rig.server._cond.notify_all()
    assert isinstance(error_of(link.receive()), ModelServerStopping)
    assert rig.server.health().status is HealthStatus.UNHEALTHY
    future.set_result(99)


def test_a_callback_that_fails_marks_the_server_unhealthy(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = rig.session()
    future = rig.step(link)

    def broken(item: Any, output: Any) -> None:
        raise RuntimeError("complete failed")

    monkeypatch.setattr(rig.server, "_complete", broken)
    future.set_result(10)
    health = rig.server.health()
    assert health.status is HealthStatus.UNHEALTHY
    assert "complete failed" in str(health.detail)
    monkeypatch.undo()
    with rig.server._cond:
        rig.server._pending.clear()
        rig.server._policy.finished(list(rig.server._connections))


def test_a_result_the_session_cannot_take_closes_it(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = rig.session()
    future = rig.step(link)
    connection = next(iter(rig.server._connections.values()))
    monkeypatch.setattr(connection, "post", lambda message: False)
    future.set_result(10)
    with pytest.raises(Closed):
        link.receive()
    assert rig.model.closed.get(timeout=5) == (0, 1)


def test_the_endurance_probe_counts_running_requests_and_their_release(rig: _Rig) -> None:
    from endurance.model_server_launcher import pending_requests

    baseline = pending_requests()
    link = rig.session()
    future = rig.step(link)
    assert pending_requests() == baseline + 1
    future.set_result(10)
    assert link.receive() == Result(1, 10)
    assert pending_requests() == baseline


def test_many_sessions_with_resets_get_exactly_their_own_results(rig: _Rig) -> None:
    """Resolve futures from several threads, in random order, while sessions reset."""
    rng = random.Random(7)
    links = [rig.session(slot) for slot in range(4)]
    stop = threading.Event()

    def resolve() -> None:
        held: list[Future[int]] = []
        while not stop.is_set() or held:
            with contextlib.suppress(queue.Empty):
                held.append(rig.model.futures.get(timeout=0.01))
            if held and rng.random() < 0.5:
                future = held.pop(rng.randrange(len(held)))
                time.sleep(rng.random() * 0.002)
                future.set_result(-1)

    resolvers = [threading.Thread(target=resolve) for _ in range(3)]
    for thread in resolvers:
        thread.start()
    try:
        for round_ in range(1, 41):
            for slot, link in enumerate(links):
                if round_ % 10 == slot:
                    link.send(Reset(request_id=round_ * 10, session_id=f"s{round_}"))
                    while not isinstance(link.receive(), ResetDone):
                        pass
                link.send(Step(request_id=round_, timeout=None, payload=slot))
            for link in links:
                message = link.receive()
                while not isinstance(message, Result) or message.request_id != round_:
                    message = link.receive()
    finally:
        stop.set()
        for thread in resolvers:
            thread.join(5)
    with rig.server._cond:
        assert rig.server._cond.wait_for(lambda: not rig.server._pending, timeout=5)
