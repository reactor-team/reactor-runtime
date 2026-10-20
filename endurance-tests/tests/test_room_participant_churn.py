"""room_participant_churn: participants of a LiveKit room joining and leaving one session."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pytest
from endurance.harness import ResourceSampler, RuntimeProcess, assert_answered
from endurance.report import LiveReporter, MetricResult, finish_and_check, now_iso
from endurance.room import LIVEKIT_URL, RoomParticipant, token
from endurance.trends import (
    ENDURANCE_DURATION_SECONDS,
    ENDURANCE_SMOKE_CYCLES,
    exact_metrics,
    retained_cycles,
    trend_metrics,
)

DESCRIPTION = (
    "One long-lived session connected to a LiveKit room. Each cycle a participant joins "
    "the room, runs a command through the Reactor protocol, an RPC method and a data "
    "topic, and leaves once each is answered; the runtime is then read once the "
    "participant's connection has closed. The room connection stays open throughout."
)

# The session's own participant in the room.
_MODEL = "reactor-session"
_JOIN_WAIT_S = 30.0


def _open_room(runtime: RuntimeProcess, session_id: str, room: str) -> None:
    """Connect the session to *room*, and wait for the room connection to open."""
    path = f"/sessions/{session_id}/transport/livekit/connections"
    _, accepted = runtime.request("POST", path, {"url": LIVEKIT_URL, "token": token(room, _MODEL)})
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


def _unheard(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """What the runtime did not take from the cycle's participant, between two readings.

    LiveKit accepts a packet, a track, or a stream whether or not the runtime
    does anything with it, so the runtime's own counts are the proof.
    """
    missing = []
    ran = after["topic_commands"]["ran"] - before["topic_commands"]["ran"]
    rejected = after["topic_commands"]["rejected"] - before["topic_commands"]["rejected"]
    if rejected:
        missing.append("set_effect (topic): the runtime rejected the packet's command")
    elif ran != 1:
        missing.append(f"set_effect (topic): the runtime ran {ran} commands from the packet, not 1")
    for track in ("webcam", "mic"):
        if after["input_frames"].get(track, 0) <= before["input_frames"].get(track, 0):
            missing.append(f"{track}: no frame reached the runtime's input track")
    return missing


async def test_room_participant_churn(runtime: RuntimeProcess) -> None:
    """Release everything a room participant held once it has left, every time."""
    if not LIVEKIT_URL:
        pytest.skip("the room churn needs a LiveKit server: set LIVEKIT_URL")
    session = runtime.request("POST", "/start_session", {})[1]
    if not any(t["protocol"] == "livekit" for t in session.get("transports", [])):
        pytest.skip("the runtime under test serves no LiveKit transport")
    version = session["server_info"]["server_version"]
    if source := os.environ.get("ENDURANCE_RUNTIME_REF"):
        version = f"from {source}"
    room = f"endurance-{uuid.uuid4().hex[:8]}"
    _open_room(runtime, session["session_id"], room)
    # Settle with the room connection open, so the task baseline includes what it keeps.
    previous, _ = runtime.settled_probe(connections=1)
    sampler = ResourceSampler(runtime)
    live = LiveReporter("room-participant-churn", ENDURANCE_DURATION_SECONDS)
    started_at = now_iso()
    cycle = 0
    unanswered: list[list[str]] = []
    metrics: list[MetricResult] = []

    try:
        while not sampler.deadline_reached():
            if ENDURANCE_SMOKE_CYCLES and cycle >= ENDURANCE_SMOKE_CYCLES:
                break
            participant = RoomParticipant(
                room, f"participant-{cycle}", _MODEL, input_frames=runtime.input_frames
            )
            await participant.run()
            # A participant's readers close their streams as it leaves.
            reading, _ = runtime.settled_probe(connections=1, tasks=sampler.initial_tasks)
            unanswered.append([*participant.unanswered, *_unheard(previous, reading)])
            previous = reading
            sampler.sample(cycle=cycle, reading=reading)
            live.update(sampler.samples, retained=retained_cycles(sampler.samples))
            cycle += 1
        if not sampler.samples:
            raise AssertionError(
                f"no cycle completed in {ENDURANCE_DURATION_SECONDS:g}s: "
                "raise ENDURANCE_DURATION_SECONDS"
            )
        metrics = [assert_answered(unanswered)]
        if not ENDURANCE_SMOKE_CYCLES:
            metrics += exact_metrics(sampler.samples, tasks_baseline=sampler.initial_tasks)
            metrics += trend_metrics(sampler.samples)
    finally:
        if not metrics and sampler.samples:
            metrics = [
                assert_answered(unanswered),
                *exact_metrics(sampler.samples, tasks_baseline=sampler.initial_tasks),
            ]
        finish_and_check(
            test_name="room-participant-churn",
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
