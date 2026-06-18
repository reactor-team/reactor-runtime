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
    """The lifecycle states one session moves through."""

    CREATED = auto()
    READY = auto()
    WAITING = auto()
    STREAMING = auto()
    ORPHANED = auto()
    CLOSING = auto()
    TERMINATED = auto()


class SessionEvent(Enum):
    """The events that drive moves between session states."""

    INITIALIZATION_SUCCESS = auto()
    INITIALIZATION_FAIL = auto()
    START_SESSION = auto()
    STOP_SESSION = auto()
    TIMEOUT = auto()
    CLIENT_CONNECTED = auto()
    CLIENT_DISCONNECTED = auto()
    CONNECTION_OPENED = auto()
    CONNECTION_CLOSED = auto()
    CLEANUP_COMPLETE = auto()
    EVICTION = auto()
    IDLING = auto()
    SESSION_ACTIVE = auto()


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
