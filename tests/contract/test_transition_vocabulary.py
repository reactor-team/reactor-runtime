"""Lock the transition vocabulary: the wire names, the legal edges, and detail.

Consumers switch on the envelope's ``event`` string and mirror session state
from the ``from``/``to`` strings, so the enum-derived wire names, the edges the
state machine allows, and the ``detail`` payload of each move are all contract.
Every assertion serializes through the real SSE path and compares against
hard-coded literals.
"""

from __future__ import annotations

from typing import Any

import pytest
from contract_helpers import envelope

from reactor_runtime.core import (
    JOURNAL_EVENTS,
    EndReason,
    SessionEvent,
    SessionState,
    Transition,
    TransitionEvent,
)
from reactor_runtime.runner.state_machine import SessionStateMachine

_STATES = frozenset(
    {"created", "ready", "waiting", "streaming", "orphaned", "closing", "terminated"}
)
_EVENTS = frozenset(
    {
        "initialization_success",
        "initialization_fail",
        "start_session",
        "stop_session",
        "timeout",
        "connection_opened",
        "connection_closed",
        "connection_answered",
        "cleanup_complete",
        "eviction",
        "chunk_ready",
        "clip_ready",
        "command",
        "error",
        "metric",
    }
)
_JOURNAL_FACTS = frozenset({"chunk_ready", "clip_ready", "command", "error", "metric"})


def _apply(
    machine: SessionStateMachine, event: SessionEvent, **detail: Any
) -> dict[str, Any] | None:
    """Send *event* and return its serialized envelope, or ``None`` if rejected."""
    captured: list[Transition] = []
    machine.on_transition(captured.append)
    if not machine.send(event, **detail):
        assert not captured, "a rejected event must never notify listeners"
        return None
    return envelope(1, TransitionEvent(captured[-1]))


def _machine(state_name: str) -> SessionStateMachine:
    return SessionStateMachine(initial_state=SessionState[state_name.upper()])


def _streaming_machine(live: int) -> SessionStateMachine:
    machine = _machine("waiting")
    for index in range(live):
        assert machine.send(SessionEvent.CONNECTION_OPENED, conn_id=1002 + index)
    return machine


# -- vocabulary ----------------------------------------------------------------


def test_the_wire_state_vocabulary_is_locked() -> None:
    assert {state.name.lower() for state in SessionState} == _STATES


def test_the_wire_event_vocabulary_is_locked() -> None:
    assert {event.name.lower() for event in SessionEvent} == _EVENTS


def test_the_journal_fact_vocabulary_is_locked() -> None:
    assert {event.name.lower() for event in JOURNAL_EVENTS} == _JOURNAL_FACTS


def test_the_end_reason_wire_values_are_locked() -> None:
    assert {reason.value for reason in EndReason} == {
        "stopped",
        "timed_out",
        "evicted",
        "moderated",
        "error",
    }


# -- legal edges ----------------------------------------------------------------


def test_boot_edges() -> None:
    success = _apply(_machine("created"), SessionEvent.INITIALIZATION_SUCCESS)
    assert success is not None
    assert (success["event"], success["from"], success["to"]) == (
        "initialization_success",
        "created",
        "ready",
    )

    failure = _apply(_machine("created"), SessionEvent.INITIALIZATION_FAIL)
    assert failure is not None
    assert (failure["event"], failure["from"], failure["to"]) == (
        "initialization_fail",
        "created",
        "terminated",
    )


def test_session_open_edge() -> None:
    payload = _apply(_machine("ready"), SessionEvent.START_SESSION)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == (
        "start_session",
        "ready",
        "waiting",
    )


@pytest.mark.parametrize("origin", ["waiting", "orphaned"])
def test_the_first_connection_carries_the_session_into_streaming(origin: str) -> None:
    payload = _apply(_machine(origin), SessionEvent.CONNECTION_OPENED, conn_id=1002)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == (
        "connection_opened",
        origin,
        "streaming",
    )


def test_a_later_connection_self_loops_in_streaming() -> None:
    payload = _apply(_streaming_machine(live=1), SessionEvent.CONNECTION_OPENED, conn_id=1003)
    assert payload is not None
    assert (payload["from"], payload["to"]) == ("streaming", "streaming")


def test_closing_a_connection_with_others_live_self_loops() -> None:
    payload = _apply(_streaming_machine(live=2), SessionEvent.CONNECTION_CLOSED, conn_id=1002)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == (
        "connection_closed",
        "streaming",
        "streaming",
    )


def test_closing_the_last_connection_orphans_the_session() -> None:
    # The edge a consumer reads as "the session went inactive": the last
    # connection leaving moves streaming to orphaned on its close fact.
    payload = _apply(_streaming_machine(live=1), SessionEvent.CONNECTION_CLOSED, conn_id=1002)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == (
        "connection_closed",
        "streaming",
        "orphaned",
    )


@pytest.mark.parametrize("origin", ["waiting", "streaming", "orphaned"])
def test_stop_closes_from_every_active_state(origin: str) -> None:
    machine = _streaming_machine(live=1) if origin == "streaming" else _machine(origin)
    payload = _apply(machine, SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == ("stop_session", origin, "closing")


@pytest.mark.parametrize("origin", ["waiting", "orphaned"])
def test_timeout_closes_a_clientless_session(origin: str) -> None:
    payload = _apply(_machine(origin), SessionEvent.TIMEOUT, reason=EndReason.TIMED_OUT)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == ("timeout", origin, "closing")


def test_cleanup_unwinds_to_ready() -> None:
    payload = _apply(_machine("closing"), SessionEvent.CLEANUP_COMPLETE, reason=EndReason.STOPPED)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == (
        "cleanup_complete",
        "closing",
        "ready",
    )


@pytest.mark.parametrize(
    "origin", ["created", "ready", "waiting", "streaming", "orphaned", "closing"]
)
def test_eviction_is_terminal_from_every_live_state(origin: str) -> None:
    payload = _apply(_machine(origin), SessionEvent.EVICTION, reason=EndReason.EVICTED)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == ("eviction", origin, "terminated")


@pytest.mark.parametrize(
    "state", ["created", "ready", "waiting", "streaming", "orphaned", "closing", "terminated"]
)
def test_a_negotiation_answer_self_loops_in_every_state(state: str) -> None:
    payload = _apply(
        _machine(state),
        SessionEvent.CONNECTION_ANSWERED,
        conn_id=1002,
        answer={"type": "answer", "sdp": "v=0"},
    )
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == (
        "connection_answered",
        state,
        state,
    )


@pytest.mark.parametrize(
    "state", ["created", "ready", "waiting", "streaming", "orphaned", "closing", "terminated"]
)
@pytest.mark.parametrize("fact", sorted(JOURNAL_EVENTS, key=lambda event: event.name))
def test_journal_facts_self_loop_in_every_state(state: str, fact: SessionEvent) -> None:
    # A journal fact is never dropped by the table: the final chunk_ready fires
    # during closing and an error can fire after termination.
    payload = _apply(_machine(state), fact)
    assert payload is not None
    assert (payload["event"], payload["from"], payload["to"]) == (fact.name.lower(), state, state)


@pytest.mark.parametrize(
    ("state", "event"),
    [
        ("ready", SessionEvent.INITIALIZATION_SUCCESS),
        ("waiting", SessionEvent.START_SESSION),
        ("ready", SessionEvent.STOP_SESSION),
        ("ready", SessionEvent.TIMEOUT),
        ("ready", SessionEvent.CONNECTION_CLOSED),
        ("created", SessionEvent.CONNECTION_OPENED),
        ("terminated", SessionEvent.START_SESSION),
        ("terminated", SessionEvent.EVICTION),
    ],
)
def test_illegal_edges_are_rejected_without_a_journal_fact(state: str, event: SessionEvent) -> None:
    assert _apply(_machine(state), event) is None


# -- detail payloads -------------------------------------------------------------


def test_detail_rides_the_envelope_verbatim() -> None:
    payload = _apply(
        _machine("waiting"),
        SessionEvent.CONNECTION_ANSWERED,
        conn_id=1002,
        answer={"type": "answer", "sdp": "v=0\r\no=- 0 0 IN IP4 0.0.0.0"},
    )
    assert payload is not None
    assert payload["detail"] == {
        "conn_id": 1002,
        "answer": {"type": "answer", "sdp": "v=0\r\no=- 0 0 IN IP4 0.0.0.0"},
    }


def test_connection_ids_serialize_as_json_integers() -> None:
    payload = _apply(_machine("waiting"), SessionEvent.CONNECTION_OPENED, conn_id=1002)
    assert payload is not None
    assert payload["detail"] == {"conn_id": 1002}
    assert isinstance(payload["detail"]["conn_id"], int)


def test_end_reasons_serialize_as_their_wire_strings() -> None:
    payload = _apply(_machine("waiting"), SessionEvent.STOP_SESSION, reason=EndReason.STOPPED)
    assert payload is not None
    assert payload["detail"] == {"reason": "stopped"}


def test_eviction_detail_carries_reason_and_error() -> None:
    payload = _apply(
        _machine("streaming"),
        SessionEvent.EVICTION,
        reason=EndReason.ERROR,
        error="model exploded",
    )
    assert payload is not None
    assert payload["detail"] == {"reason": "error", "error": "model exploded"}


def test_chunk_ready_detail_names_the_recording_and_segment_index() -> None:
    payload = _apply(
        _machine("streaming"),
        SessionEvent.CHUNK_READY,
        recording_id="7d9f5c1e-0000-0000-0000-000000000042",
        idx=-1,
    )
    assert payload is not None
    # idx -1 is the init segment; media segments count up from 0.
    assert payload["detail"] == {
        "recording_id": "7d9f5c1e-0000-0000-0000-000000000042",
        "idx": -1,
    }
    assert isinstance(payload["detail"]["idx"], int)
