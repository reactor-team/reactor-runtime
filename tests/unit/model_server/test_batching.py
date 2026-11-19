"""The batching policy on its own: requests and a clock in, batches and expiries out."""

from __future__ import annotations

import numpy as np

from reactor_runtime.model_server.batching import LIVE_SECONDS, BatchPolicy, Waiting


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


def _live(policy: BatchPolicy, *connections: int, at: float) -> None:
    """Make *connections* live: each sends a request at *at*, which steps and finishes."""
    for connection in connections:
        policy.add(_request(connection, -connection, at))
    while policy.waiting:
        turn = policy.take(now=at, open_sessions=set(connections))
        policy.finished([r.connection for r in turn.batch])


def test_a_session_has_one_request_waiting_or_running_at_a_time() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=0.0)
    assert policy.add(_request(1, 1, 0.0))
    assert not policy.add(_request(1, 2, 0.0))  # still waiting

    turn = policy.take(now=0.0, open_sessions={1})
    assert [r.request_id for r in turn.batch] == [1]
    assert not policy.add(_request(1, 3, 0.1))  # still running

    policy.finished([1])
    assert policy.add(_request(1, 3, 0.2))


def test_requests_are_taken_in_arrival_order_up_to_max_batch() -> None:
    policy = BatchPolicy(max_batch=2, batch_window=0.0)
    for connection in (3, 1, 2):
        policy.add(_request(connection, connection, 0.0))

    first = policy.take(now=0.0, open_sessions={1, 2, 3})
    assert [r.connection for r in first.batch] == [3, 1]
    second = policy.take(now=0.0, open_sessions={1, 2, 3})
    assert [r.connection for r in second.batch] == [2]


def test_a_batch_waits_its_window_unless_it_is_full() -> None:
    policy = BatchPolicy(max_batch=3, batch_window=0.010)
    _live(policy, 1, 2, 3, at=0.9)
    policy.add(_request(1, 1, arrived=1.000))

    waiting = policy.take(now=1.002, open_sessions={1, 2, 3})
    assert waiting.batch == []
    assert waiting.wake_at == 1.010

    policy.add(_request(2, 2, arrived=1.003))
    assert policy.take(now=1.004, open_sessions={1, 2, 3}).batch == []
    assert [r.connection for r in policy.take(now=1.010, open_sessions={1, 2, 3}).batch] == [1, 2]


def test_a_batch_is_taken_early_once_every_open_session_has_a_request() -> None:
    policy = BatchPolicy(max_batch=8, batch_window=1.0)
    policy.add(_request(1, 1, 0.0))
    policy.add(_request(2, 2, 0.0))

    assert [r.connection for r in policy.take(now=0.0, open_sessions={1, 2}).batch] == [1, 2]


def test_a_session_whose_request_runs_does_not_count_toward_a_full_batch() -> None:
    policy = BatchPolicy(max_batch=8, batch_window=1.0)
    _live(policy, 1, 2, 3, at=-0.5)
    policy.add(_request(1, 1, 0.0))
    assert [r.connection for r in policy.take(now=1.0, open_sessions={1, 2, 3}).batch] == [1]

    # Session 1's request still runs, so sessions 2 and 3 are all that can send.
    policy.add(_request(2, 2, 1.1))
    assert policy.take(now=1.1, open_sessions={1, 2, 3}).batch == []
    policy.add(_request(3, 3, 1.2))
    assert [r.connection for r in policy.take(now=1.2, open_sessions={1, 2, 3}).batch] == [2, 3]


def test_a_request_past_its_deadline_is_expired_and_frees_its_session() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=0.0)
    policy.add(_request(1, 1, arrived=0.0, timeout=0.5))
    policy.add(_request(2, 2, arrived=0.0, timeout=5.0))

    turn = policy.take(now=1.0, open_sessions={1, 2})

    assert [r.request_id for r in turn.expired] == [1]
    assert [r.request_id for r in turn.batch] == [2]
    assert policy.add(_request(1, 3, arrived=1.0))


def test_a_filling_batch_wakes_at_the_earliest_deadline() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=10.0)
    _live(policy, 1, 2, at=-0.5)
    policy.add(_request(1, 1, arrived=0.0, timeout=0.1))

    turn = policy.take(now=0.0, open_sessions={1, 2})
    assert turn.wake_at == 0.1
    assert [r.request_id for r in policy.take(now=0.1, open_sessions={1, 2}).expired] == [1]


def test_a_request_expires_beside_others_whose_payloads_hold_arrays() -> None:
    policy = BatchPolicy(max_batch=4, batch_window=0.0)
    for connection, timeout in ((1, 0.5), (2, None)):
        request = _request(connection, connection, arrived=0.0, timeout=timeout)
        request.payload = np.arange(8)
        policy.add(request)

    turn = policy.take(now=1.0, open_sessions={1, 2})

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
    assert policy.take(now=0.0, open_sessions={1, 2}).batch == []


def test_a_session_that_never_sent_does_not_hold_a_batch_back() -> None:
    # A session runtime with no client connects but sends nothing: the live
    # sessions' batch steps as soon as each of them has a request waiting.
    policy = BatchPolicy(max_batch=8, batch_window=1.0)
    _live(policy, 1, 2, at=0.0)
    policy.add(_request(1, 1, 0.5))
    policy.add(_request(2, 2, 0.5))

    turn = policy.take(now=0.5, open_sessions={1, 2, 3, 4})

    assert [r.connection for r in turn.batch] == [1, 2]


def test_a_session_that_went_quiet_stops_counting_after_live_seconds() -> None:
    policy = BatchPolicy(max_batch=8, batch_window=10.0)
    _live(policy, 1, 2, 3, at=0.0)
    policy.add(_request(1, 1, 1.0))
    policy.add(_request(2, 2, 1.0))
    assert policy.take(now=1.0, open_sessions={1, 2, 3}).batch == []  # 3 is still live

    later = 0.0 + LIVE_SECONDS
    assert [r.connection for r in policy.take(now=later, open_sessions={1, 2, 3}).batch] == [1, 2]


def test_a_waiting_batch_wakes_when_a_session_it_waits_for_goes_quiet() -> None:
    # Session 2 is live at first, so session 1's request waits for it; the
    # policy asks to be woken when session 2 stops being live, well before the
    # window ends, and the batch then steps without it.
    policy = BatchPolicy(max_batch=8, batch_window=10.0)
    _live(policy, 1, 2, at=0.0)
    policy.add(_request(1, 1, 1.0))

    waiting = policy.take(now=1.0, open_sessions={1, 2})
    assert waiting.batch == []
    assert waiting.wake_at == 0.0 + LIVE_SECONDS

    turn = policy.take(now=LIVE_SECONDS, open_sessions={1, 2})
    assert [r.connection for r in turn.batch] == [1]


def test_an_old_waiting_request_does_not_stand_in_for_a_live_session() -> None:
    # With a window past LIVE_SECONDS, session 1's request is older than that,
    # but it is still waiting: it must not fill the place of session 2, which
    # is live and has not sent yet.
    policy = BatchPolicy(max_batch=2, batch_window=10.0)
    policy.add(_request(1, 1, 0.0))
    policy.live(2, 1.9)

    assert policy.take(now=2.05, open_sessions={1, 2}).batch == []
    policy.add(_request(2, 2, 2.1))
    assert [r.connection for r in policy.take(now=2.1, open_sessions={1, 2}).batch] == [1, 2]


def test_a_closed_session_stops_counting() -> None:
    policy = BatchPolicy(max_batch=8, batch_window=10.0)
    _live(policy, 1, 2, at=0.0)
    policy.add(_request(1, 1, 0.5))
    assert policy.take(now=0.5, open_sessions={1, 2}).batch == []
    assert [r.connection for r in policy.take(now=0.6, open_sessions={1}).batch] == [1]


def test_a_session_counts_as_live_from_when_it_opens_or_resets() -> None:
    # A session that just opened, or whose client just started a session, is
    # waited for before its first request comes.
    policy = BatchPolicy(max_batch=8, batch_window=1.0)
    policy.live(2, 0.0)
    policy.add(_request(1, 1, 0.1))
    assert policy.take(now=0.1, open_sessions={1, 2}).batch == []
    policy.add(_request(2, 2, 0.2))
    assert [r.connection for r in policy.take(now=0.2, open_sessions={1, 2}).batch] == [1, 2]
