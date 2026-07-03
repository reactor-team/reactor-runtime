"""Session state machine.

The session lifecycle as a small, synchronous machine. ``send`` does one job —
resolve the event against the current state, flip the state, and notify listeners
— and runs no side effects of its own: it never starts a task, touches a
transport, or reaches into the model. The machine is composed, not inherited: the
connection manager drives it with ``send`` while the runner observes it with
``on_transition`` and funnels every downstream effect through its own dispatch.

Session occupancy is derived, not signalled. The connection manager reports only
the per-connection ``CONNECTION_OPENED`` / ``CONNECTION_CLOSED`` facts; the
machine counts live connections and moves ``WAITING``/``ORPHANED`` to
``STREAMING`` on the first connection and ``STREAMING`` to ``ORPHANED`` on the
last, while connections in between ride as self-loops. ``CONNECTION_ANSWERED``
also self-loops in every active state but leaves the count untouched: it records
a negotiation answer for a connection that has not yet connected.

Because the core is synchronous and side-effect-free, the whole machine is
exercised with ``send()`` and ``current_state`` alone, with no transport, model,
or event loop in the test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reactor_runtime.core import SessionEvent, SessionState, Transition

# The static edges: events whose target is a pure function of the current state.
# For each event, the state it may be sent from mapped to the state it lands in;
# an event with no entry for the current state is illegal and is rejected without
# changing state. Occupancy — the connection-driven moves between WAITING/ORPHANED
# and STREAMING — is not here: it is resolved from the live-connection count.
_TRANSITIONS: dict[SessionEvent, dict[SessionState, SessionState]] = {
    SessionEvent.INITIALIZATION_SUCCESS: {SessionState.CREATED: SessionState.READY},
    SessionEvent.INITIALIZATION_FAIL: {SessionState.CREATED: SessionState.TERMINATED},
    SessionEvent.START_SESSION: {SessionState.READY: SessionState.WAITING},
    SessionEvent.STOP_SESSION: {
        SessionState.STREAMING: SessionState.CLOSING,
        SessionState.WAITING: SessionState.CLOSING,
        SessionState.ORPHANED: SessionState.CLOSING,
    },
    SessionEvent.TIMEOUT: {
        SessionState.WAITING: SessionState.CLOSING,
        SessionState.ORPHANED: SessionState.CLOSING,
    },
    SessionEvent.CLEANUP_COMPLETE: {SessionState.CLOSING: SessionState.READY},
    # Eviction is terminal from every live state, not just READY: a model that
    # will not serve again — an idle eviction or a crashed run loop — lands
    # straight in TERMINATED wherever it was. CREATED is included so a crash
    # racing the initialization edge is recorded rather than rejected into
    # silence. The detail's reason distinguishes why (see EndReason).
    SessionEvent.EVICTION: {
        SessionState.CREATED: SessionState.TERMINATED,
        SessionState.READY: SessionState.TERMINATED,
        SessionState.WAITING: SessionState.TERMINATED,
        SessionState.STREAMING: SessionState.TERMINATED,
        SessionState.ORPHANED: SessionState.TERMINATED,
        SessionState.CLOSING: SessionState.TERMINATED,
    },
    # A negotiation answer is a fact about a connection that has not yet
    # connected, so it self-loops in every active state and leaves occupancy
    # alone (see _update_count). It carries the connection id and the answer
    # on the transition, journalled but driving no state change.
    SessionEvent.CONNECTION_ANSWERED: {
        SessionState.WAITING: SessionState.WAITING,
        SessionState.STREAMING: SessionState.STREAMING,
        SessionState.ORPHANED: SessionState.ORPHANED,
    },
}

# The states a connection may open from. WAITING and ORPHANED both mean "no live
# connection", so the first open carries either into STREAMING; STREAMING stays
# put for a later one. from_state alone decides the open, count-blind.
_CONNECTABLE = frozenset({SessionState.WAITING, SessionState.ORPHANED, SessionState.STREAMING})


class SessionStateMachine:
    """The session lifecycle as a validated, synchronous machine.

    Holds the current state, the count of live connections, and a list of
    listeners. ``send`` is the only mutator: it resolves the event, and on a legal
    edge flips the state first — so a listener that reads ``current_state`` sees
    the state it is being notified about — then calls every listener in
    registration order with the recorded ``Transition``. Nothing here is async;
    ordering downstream of a move is the listener's concern, not the machine's.

    The live-connection count is private and serves only to resolve occupancy: it
    rises on ``CONNECTION_OPENED`` and falls on ``CONNECTION_CLOSED``, and is
    cleared whenever any other event moves the session, since every such move
    either opens a fresh session or tears the current one down.
    """

    def __init__(self, initial_state: SessionState = SessionState.CREATED) -> None:
        """Start the machine in ``initial_state`` with no listeners or connections."""
        self._state = initial_state
        self._live = 0
        self._listeners: list[Callable[[Transition], None]] = []

    @property
    def current_state(self) -> SessionState:
        """The state the session is in right now."""
        return self._state

    def on_transition(self, callback: Callable[[Transition], None]) -> None:
        """Subscribe to applied moves.

        The callback fires once per legal ``send``, after the state has flipped,
        in the order callbacks were registered. It is never called for a rejected
        event.
        """
        self._listeners.append(callback)

    def send(self, event: SessionEvent, **detail: Any) -> bool:
        """Apply ``event`` to the current state.

        Resolves the event — a connection open or close against the live count,
        any other event against the static table. An event with no legal edge from
        the current state is rejected: the state and count are untouched and no
        listener fires. On a legal edge the count is updated, the state flips, and
        every listener is notified with the ``Transition``, whose ``detail``
        carries the keyword context the caller passed (a connection id, an end
        reason, and the like).

        Args:
            event: The event driving the move.
            **detail: Out-of-band context recorded on the resulting transition.

        Returns:
            ``True`` if the event was legal and applied, ``False`` if rejected.
        """
        target = self._resolve(event)
        if target is None:
            return False
        self._update_count(event)
        previous, self._state = self._state, target
        transition = Transition(event, previous, target, detail)
        for listener in self._listeners:
            listener(transition)
        return True

    def _resolve(self, event: SessionEvent) -> SessionState | None:
        """Resolve the target state for ``event``, or ``None`` if it is illegal.

        Connection events derive occupancy from the current state and the live
        count: an open carries the session into ``STREAMING``; a close lands in
        ``ORPHANED`` only when it removes the last connection, otherwise the
        session stays ``STREAMING``. Every other event is a static table lookup.
        """
        if event is SessionEvent.CONNECTION_OPENED:
            return SessionState.STREAMING if self._state in _CONNECTABLE else None
        if event is SessionEvent.CONNECTION_CLOSED:
            if self._state is not SessionState.STREAMING:
                return None
            return SessionState.ORPHANED if self._live <= 1 else SessionState.STREAMING
        return _TRANSITIONS.get(event, {}).get(self._state)

    def _update_count(self, event: SessionEvent) -> None:
        """Track live connections so occupancy can be resolved.

        Rises on an open and falls on a close (never below zero). A negotiation
        answer leaves the count alone: it is a self-loop about a connection that
        has not yet connected, so it neither opens nor unwinds the session. Any
        other move clears the count, since a non-connection transition either
        begins a fresh session or unwinds the current one — both of which leave
        no connection behind.
        """
        if event is SessionEvent.CONNECTION_OPENED:
            self._live += 1
        elif event is SessionEvent.CONNECTION_CLOSED:
            self._live = max(0, self._live - 1)
        elif event is SessionEvent.CONNECTION_ANSWERED:
            pass
        else:
            self._live = 0
