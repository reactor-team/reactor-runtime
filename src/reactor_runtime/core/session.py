"""Session vocabulary.

The session lifecycle as enums plus the immutable record of one move between
states. No transition table lives here — the legal edges and their validation
are the session state machine's job in a later flow. This module only names the
states and events and gives a transition its derived session-boundary
properties.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SessionState(Enum):
    """The lifecycle states one session moves through.

    States:
        CREATED: The process is up but the model has not finished loading.
        READY: The model is loaded and idle, waiting to be told to open a
            session.
        WAITING: A session is open but no connection has reached it yet.
        STREAMING: At least one connection has reached its wire-connected state
            — for WebRTC, the peer connection is ``CONNECTED`` (ICE and DTLS
            established). This is the occupancy signal; it does not assert that a
            data channel is open or that media is flowing.
        ORPHANED: The session was streaming and every connection has since left;
            it awaits a reconnection or the orphan timeout.
        CLOSING: The session is tearing its connections down and unwinding back
            to ``READY`` once cleanup completes.
        TERMINATED: The process is finished — the model failed to load or it was
            evicted — and will not serve again.
    """

    CREATED = auto()
    READY = auto()
    WAITING = auto()
    STREAMING = auto()
    ORPHANED = auto()
    CLOSING = auto()
    TERMINATED = auto()


class SessionEvent(Enum):
    """The events that drive moves between session states.

    Session occupancy — the moves between ``WAITING``/``ORPHANED`` and
    ``STREAMING`` — is not its own event. It is derived by the state machine
    from the per-connection ``CONNECTION_OPENED`` / ``CONNECTION_CLOSED`` facts,
    which it counts: the first connection to arrive carries the session into
    ``STREAMING`` and the last to leave carries it to ``ORPHANED``, while every
    connection in between rides as a self-loop.
    """

    INITIALIZATION_SUCCESS = auto()
    INITIALIZATION_FAIL = auto()
    START_SESSION = auto()
    STOP_SESSION = auto()
    TIMEOUT = auto()
    CONNECTION_OPENED = auto()
    CONNECTION_CLOSED = auto()
    CLEANUP_COMPLETE = auto()
    EVICTION = auto()


@dataclass(frozen=True)
class Transition:
    """One move between session states, recorded after the move is applied.

    Attributes:
        event: The event that caused the move.
        from_state: The state left behind.
        to_state: The state now current.
        detail: Out-of-band context for the move (e.g. a connection id or an
            end reason), keyed by name.
    """

    event: SessionEvent
    from_state: SessionState
    to_state: SessionState
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_session_start(self) -> bool:
        """Whether this move opens a session (``READY`` to ``WAITING``).

        The real session boundary, distinct from a per-connection open: it fires
        once when the session begins, not for every client that later joins.
        """
        return self.from_state is SessionState.READY and self.to_state is SessionState.WAITING

    @property
    def is_session_end(self) -> bool:
        """Whether this move closes a session (``CLOSING`` to ``READY``).

        The real close, reached once cleanup completes — typically a grace
        period after the last client disconnects — rather than the first
        disconnect.
        """
        return self.from_state is SessionState.CLOSING and self.to_state is SessionState.READY
