"""The batching policy on its own: requests and a clock in, batches and expiries out."""

from __future__ import annotations

import numpy as np

from reactor_runtime.model_server.batching import BatchPolicy, Waiting


def _request(
    connection: int, request_id: int, arrived: float, timeout: float | None = None
) -> Waiting:
    return Waiting(
        connection=connection,
        request_id=request_id,
        payload=None,
        arrived=arrived,
        deadline=None if timeout is None else arrived + timeout,
    )


def test_a_session_has_one_request_waiting_or_running_at_a_time() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=0.0)
    assert policy.add(_request(1, 1, 0.0))
    assert not policy.add(_request(1, 2, 0.0))  # still waiting

    turn = policy.take(now=0.0, open_sessions=1)
    assert [r.request_id for r in turn.batch] == [1]
    assert not policy.add(_request(1, 3, 0.1))  # still running

    policy.finished([1])
    assert policy.add(_request(1, 3, 0.2))


def test_requests_are_taken_in_arrival_order_up_to_max_batch() -> None:
    policy = BatchPolicy(max_batch=2, batch_window=0.0)
    for connection in (3, 1, 2):
        policy.add(_request(connection, connection, 0.0))

    first = policy.take(now=0.0, open_sessions=3)
    assert [r.connection for r in first.batch] == [3, 1]
    second = policy.take(now=0.0, open_sessions=3)
    assert [r.connection for r in second.batch] == [2]


def test_a_batch_waits_its_window_unless_it_is_full() -> None:
    policy = BatchPolicy(max_batch=3, batch_window=0.010)
    policy.add(_request(1, 1, arrived=1.000))

    waiting = policy.take(now=1.002, open_sessions=3)
    assert waiting.batch == []
    assert waiting.wake_at == 1.010

    policy.add(_request(2, 2, arrived=1.003))
    assert policy.take(now=1.004, open_sessions=3).batch == []
    assert [r.connection for r in policy.take(now=1.010, open_sessions=3).batch] == [1, 2]


def test_a_batch_is_taken_early_once_every_open_session_has_a_request() -> None:
    policy = BatchPolicy(max_batch=8, batch_window=1.0)
    policy.add(_request(1, 1, 0.0))
    policy.add(_request(2, 2, 0.0))

    assert [r.connection for r in policy.take(now=0.0, open_sessions=2).batch] == [1, 2]


def test_a_request_past_its_deadline_is_expired_and_frees_its_session() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=0.0)
    policy.add(_request(1, 1, arrived=0.0, timeout=0.5))
    policy.add(_request(2, 2, arrived=0.0, timeout=5.0))

    turn = policy.take(now=1.0, open_sessions=2)

    assert [r.request_id for r in turn.expired] == [1]
    assert [r.request_id for r in turn.batch] == [2]
    assert policy.add(_request(1, 3, arrived=1.0))


def test_a_filling_batch_wakes_at_the_earliest_deadline() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=10.0)
    policy.add(_request(1, 1, arrived=0.0, timeout=0.1))

    turn = policy.take(now=0.0, open_sessions=2)
    assert turn.wake_at == 0.1
    assert [r.request_id for r in policy.take(now=0.1, open_sessions=2).expired] == [1]


def test_a_request_expires_beside_others_whose_payloads_hold_arrays() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=0.0)
    for connection, timeout in ((1, 0.5), (2, None)):
        request = _request(connection, connection, arrived=0.0, timeout=timeout)
        request.payload = np.arange(8)
        policy.add(request)

    turn = policy.take(now=1.0, open_sessions=2)

    assert [r.request_id for r in turn.expired] == [1]
    assert [r.request_id for r in turn.batch] == [2]


def test_cancel_and_drop_withdraw_waiting_requests() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=0.0)
    policy.add(_request(1, 1, 0.0))
    policy.add(_request(2, 2, 0.0))

    assert policy.cancel(1, 1)
    assert not policy.cancel(1, 1)
    assert [r.request_id for r in policy.drop(2)] == [2]
    assert policy.waiting == 0
    assert policy.take(now=0.0, open_sessions=2).batch == []
