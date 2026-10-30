"""room_leave_unanswered: sessions stopped while their LiveKit room has stopped answering."""

from __future__ import annotations

import os
import time
import urllib.error
import uuid

import pytest
from endurance.harness import CLOSE_TIMEOUT_S, ResourceSampler, RuntimeProcess, assert_clean_stops
from endurance.relay import StallingRelay
from endurance.report import LiveReporter, MetricResult, finish_and_check, now_iso
from endurance.room import LIVEKIT_URL, token
from endurance.trends import (
    ENDURANCE_DURATION_SECONDS,
    ENDURANCE_SMOKE_CYCLES,
    exact_metrics,
    retained_cycles,
    trend_metrics,
)

DESCRIPTION = (
    "Each cycle starts a session, connects it to a LiveKit room through a relay, and "
    "stops it while the relay has stopped answering: the room's server sees no leave "
    "and the runtime gets no reply. The session must be ready again within the stop's "
    "bound while the room is silent. The relay then answers again, and the room's "
    "leave must finish, leaving nothing behind."
)

# The session's own participant in the room.
_MODEL = "reactor-session"
_JOIN_WAIT_S = 30.0
# How soon a stopped session must be ready again while its room does not answer:
# the runtime bounds the wait for the room's leave, and the rest of teardown
# takes well under a second.
_READY_WITHIN_S = 6.0
# How long the room stays silent from the stop: past the bound by far, so a
# runtime that waits for the leave is caught.
_SILENT_S = 12.0


def _open_room(runtime: RuntimeProcess, session_id: str, url: str, room: str) -> None:
    """Connect the session to *room* at *url*, and wait for the room connection to open."""
    path = f"/sessions/{session_id}/transport/livekit/connections"
    _, accepted = runtime.request("POST", path, {"url": url, "token": token(room, _MODEL)})
    conn_id = accepted["connection_id"]
    deadline = time.monotonic() + _JOIN_WAIT_S
    while time.monotonic() < deadline:
        status, body = runtime.request("GET", f"{path}/{conn_id}")
        if status == 200:
            if body["state"] != "open":
                raise AssertionError(f"the room connection did not open: {body}")
            return
        time.sleep(0.25)
    raise AssertionError(f"the room connection was still joining after {_JOIN_WAIT_S:g}s")


def _stop_while_silent(runtime: RuntimeProcess, relay: StallingRelay) -> list[str]:
    """Stop the session with the room silent, then let it answer; return what went wrong."""
    failures: list[str] = []
    relay.stall()
    stopped_at = time.monotonic()
    try:
        runtime.request("POST", "/stop_session", {})
    except urllib.error.HTTPError as error:
        failures.append(f"/stop_session answered {error.code}")
    except OSError as error:
        failures.append(f"/stop_session did not answer: {error}")
    try:
        runtime.wait_ready(timeout_s=_READY_WITHIN_S)
    except TimeoutError:
        failures.append(
            f"the session was not ready {_READY_WITHIN_S:g}s after the stop while its room "
            "did not answer"
        )
    time.sleep(max(0.0, _SILENT_S - (time.monotonic() - stopped_at)))
    relay.resume()
    try:
        runtime.wait_ready(timeout_s=CLOSE_TIMEOUT_S)
    except TimeoutError:
        failures.append(f"the session was not ready {CLOSE_TIMEOUT_S:g}s after the room answered")
    return failures


async def test_room_leave_unanswered(runtime: RuntimeProcess) -> None:
    """Free the session at the bound while its room is silent; release the leave once it answers."""
    if not LIVEKIT_URL:
        pytest.skip("the room leave scenario needs a LiveKit server: set LIVEKIT_URL")
    descriptor = runtime.get("/session")
    if not any(t["protocol"] == "livekit" for t in descriptor.get("transports", [])):
        pytest.skip("the runtime under test serves no LiveKit transport")
    version = descriptor["server_info"]["server_version"]
    if source := os.environ.get("ENDURANCE_RUNTIME_REF"):
        version = f"from {source}"
    relay = StallingRelay(LIVEKIT_URL)
    sampler = ResourceSampler(runtime)
    live = LiveReporter("room-leave-unanswered", ENDURANCE_DURATION_SECONDS)
    started_at = now_iso()
    cycle = 0
    failures: list[list[str]] = []
    metrics: list[MetricResult] = []

    try:
        while not sampler.deadline_reached():
            if ENDURANCE_SMOKE_CYCLES and cycle >= ENDURANCE_SMOKE_CYCLES:
                break
            session = runtime.request(
                "POST", "/start_session", {"session_id": f"endurance-{cycle}"}
            )[1]
            _open_room(
                runtime, session["session_id"], relay.url, f"endurance-{uuid.uuid4().hex[:8]}"
            )
            cycle_failures = _stop_while_silent(runtime, relay)
            # The leave that ran on past the bound has finished once the room
            # answered: the loop is back to the tasks it ran before any session.
            reading, _ = runtime.settled_probe(tasks=sampler.initial_tasks)
            if reading["tasks"] > sampler.initial_tasks:
                cycle_failures.append(
                    f"{reading['tasks'] - sampler.initial_tasks} more tasks than before any "
                    "session, after the room answered: the room's leave did not finish"
                )
            failures.append(cycle_failures)
            sampler.sample(cycle=cycle, reading=reading)
            live.update(sampler.samples, retained=retained_cycles(sampler.samples))
            cycle += 1
        if not sampler.samples:
            raise AssertionError(
                f"no cycle completed in {ENDURANCE_DURATION_SECONDS:g}s: "
                "raise ENDURANCE_DURATION_SECONDS"
            )
        metrics = [assert_clean_stops(failures)]
        if not ENDURANCE_SMOKE_CYCLES:
            metrics += exact_metrics(
                sampler.samples,
                tasks_baseline=sampler.initial_tasks,
                children_baseline=sampler.initial_children,
            )
            metrics += trend_metrics(sampler.samples)
    finally:
        relay.close()
        if not metrics and sampler.samples:
            metrics = [
                assert_clean_stops(failures),
                *exact_metrics(
                    sampler.samples,
                    tasks_baseline=sampler.initial_tasks,
                    children_baseline=sampler.initial_children,
                ),
            ]
        finish_and_check(
            test_name="room-leave-unanswered",
            subject="reactor-runtime",
            description=DESCRIPTION,
            version=version,
            duration_s=ENDURANCE_DURATION_SECONDS,
            started_at=started_at,
            sampler=sampler,
            live=live,
            metrics=metrics,
            iterations=cycle,
        )
