"""Lock the ``/events`` stream: SSE framing, the one envelope type, and resume.

A consumer holds one long-lived subscription to ``GET /events``, parses only
the ``id:`` and ``data:`` SSE fields, resumes with ``?since=<seq>``, and opens
at ``since=0`` on a cold start expecting the retained backlog — including the
``initialization_success`` journalled before it connected — to replay in order.
"""

from __future__ import annotations

import asyncio

from contract_helpers import ENVELOPE_KEYS, Harness, SseFrame, read_sse, wait_for_state


async def _run_full_session(harness: Harness) -> None:
    """Drive one whole session over HTTP: start, stop, and the unwind to ready."""
    started = await harness.client.post("/start_session", json={})
    assert started.status_code == 200
    stopped = await harness.client.post("/stop_session")
    assert stopped.status_code == 200
    await wait_for_state(harness.runner, "ready")


async def test_a_cold_consumer_replays_initialization_from_since_zero(harness: Harness) -> None:
    frames = await read_sse(harness.app, "/events?since=0", count=1)

    assert frames[0].seq == 1
    payload = frames[0].payload
    assert payload["type"] == "transition"
    assert payload["event"] == "initialization_success"
    assert payload["from"] == "created"
    assert payload["to"] == "ready"
    assert payload["detail"] == {}


async def test_sequence_ids_are_contiguous_from_one(harness: Harness) -> None:
    await _run_full_session(harness)

    frames = await read_sse(harness.app, "/events?since=0", count=4)

    assert [frame.seq for frame in frames] == [1, 2, 3, 4]


async def test_every_envelope_is_a_transition_with_the_locked_keys(harness: Harness) -> None:
    await _run_full_session(harness)

    frames = await read_sse(harness.app, "/events?since=0", count=4)

    for frame in frames:
        payload = frame.payload
        # The exact key set: an addition here is a contract change consumers
        # must be able to tolerate, so it lands only by editing this literal.
        assert set(payload) == ENVELOPE_KEYS
        assert payload["type"] == "transition"
        assert isinstance(payload["event"], str)
        assert isinstance(payload["from"], str)
        assert isinstance(payload["to"], str)
        assert isinstance(payload["ts"], int)
        assert isinstance(payload["detail"], dict)


async def test_the_full_lifecycle_replays_in_order(harness: Harness) -> None:
    await _run_full_session(harness)

    frames = await read_sse(harness.app, "/events?since=0", count=4)

    assert [frame.payload["event"] for frame in frames] == [
        "initialization_success",
        "start_session",
        "stop_session",
        "cleanup_complete",
    ]
    assert [(frame.payload["from"], frame.payload["to"]) for frame in frames] == [
        ("created", "ready"),
        ("ready", "waiting"),
        ("waiting", "closing"),
        ("closing", "ready"),
    ]


async def test_since_resumes_strictly_after_the_given_sequence(harness: Harness) -> None:
    await _run_full_session(harness)

    frames = await read_sse(harness.app, "/events?since=2", count=2)

    assert [frame.seq for frame in frames] == [3, 4]
    assert [frame.payload["event"] for frame in frames] == ["stop_session", "cleanup_complete"]


async def test_a_live_subscriber_receives_new_transitions(harness: Harness) -> None:
    last = harness.runner.events.snapshot().last_seq

    async def live_frames() -> list[SseFrame]:
        return await read_sse(harness.app, f"/events?since={last}", count=1)

    reader = asyncio.create_task(live_frames())
    await asyncio.sleep(0.05)
    harness.runner.start_session({})
    frames = await reader

    assert frames[0].seq == last + 1
    assert frames[0].payload["event"] == "start_session"


async def test_timestamps_are_unix_epoch_milliseconds(harness: Harness) -> None:
    frames = await read_sse(harness.app, "/events?since=0", count=1)

    ts = frames[0].payload["ts"]
    # Epoch milliseconds: 13 digits for any contemporary date, never seconds.
    assert 1_000_000_000_000 < ts < 100_000_000_000_000
