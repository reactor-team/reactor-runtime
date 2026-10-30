"""One session's connection on the model server: its state machine and its own outbox.

Every connection moves through one explicit sequence of states, and a move the
sequence does not allow is refused, so there is no combination of flags to
keep consistent::

    CONNECTING --hello checked--> OPENING --session_opened()--> OPEN --reset--> OPEN
         |                           |                            |
         +--------------------------- leave or refusal ------------+--> CLOSING --> CLOSED

Answers to a session go through its own bounded outbox, written to its link by
its own writer thread, so the thread that runs the model's steps never waits on
a session's socket. A session that stops reading fills only its own outbox,
and is closed once the outbox is full.
"""

from __future__ import annotations

import enum
import queue
import threading
from typing import Any

from reactor_runtime.log import get_logger
from reactor_runtime.model_server.link import WorkerLink
from reactor_runtime.model_server.protocol import Closed, Result, failure

logger = get_logger(__name__)

DEFAULT_OUTBOX = 64
"""How many answers may wait for a session to read them before it is closed."""

_FLUSH = object()


class ConnectionState(enum.Enum):
    """Where a session's connection is in its life on the model server."""

    CONNECTING = "connecting"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


_MOVES: dict[ConnectionState, frozenset[ConnectionState]] = {
    ConnectionState.CONNECTING: frozenset({ConnectionState.OPENING, ConnectionState.CLOSING}),
    ConnectionState.OPENING: frozenset({ConnectionState.OPEN, ConnectionState.CLOSING}),
    ConnectionState.OPEN: frozenset({ConnectionState.OPEN, ConnectionState.CLOSING}),
    ConnectionState.CLOSING: frozenset({ConnectionState.CLOSED}),
    ConnectionState.CLOSED: frozenset(),
}


class ConnectionMoveError(RuntimeError):
    """A connection was asked to move to a state its sequence does not allow from where it is."""


class Connection:
    """One session runtime's link on the model server, with its state, slot and outbox.

    Args:
        link: The link; the connection owns it.
        number: Names the connection within the model server's life.
        outbox: How many answers may wait for the session to read them.
    """

    def __init__(self, link: WorkerLink, number: int, *, outbox: int = DEFAULT_OUTBOX) -> None:
        self.link = link
        self.number = number
        self.slot: int | None = None
        self.generation = 0
        self.session_id: str | None = None
        self._state = ConnectionState.CONNECTING
        self._lock = threading.Lock()
        self._outbox: queue.Queue[Any] = queue.Queue(maxsize=outbox)
        self._writer = threading.Thread(
            target=self._write, name=f"model-server-writer-{number}", daemon=True
        )
        self._writer.start()

    @property
    def state(self) -> ConnectionState:
        """The connection's current state."""
        return self._state

    def move(self, state: ConnectionState) -> None:
        """Move to *state*.

        Raises:
            ConnectionMoveError: The sequence does not allow it from the
                current state.
        """
        with self._lock:
            if state not in _MOVES[self._state]:
                raise ConnectionMoveError(
                    f"connection {self.number} cannot move from {self._state.value} to "
                    f"{state.value}"
                )
            self._state = state

    def post(self, message: Any) -> bool:
        """Queue *message* for the session without waiting; return ``False`` if the outbox is full.

        A closed connection takes nothing and answers ``True``: there is no
        one left to tell.
        """
        if self._state is ConnectionState.CLOSED:
            return True
        try:
            self._outbox.put_nowait(message)
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        """Send what the outbox holds, then close the link. Idempotent.

        With the outbox full there is no room to ask the writer to flush, and
        the link is closed at once.
        """
        try:
            self._outbox.put_nowait(_FLUSH)
        except queue.Full:
            self.link.close()

    def where(self) -> dict[str, Any]:
        """The fields that place a log line about this connection: its slot and session."""
        fields: dict[str, Any] = {"slot": self.slot}
        if self.session_id is not None:
            fields["session_id"] = self.session_id
        return fields

    def _write(self) -> None:
        while True:
            message = self._outbox.get()
            if message is _FLUSH:
                break
            try:
                self._send(message)
            except Exception as exc:
                logger.warning(
                    "a session's link failed while sending", error=str(exc), **self.where()
                )
                break
        self.link.close()

    def _send(self, message: Any) -> None:
        """Send *message*; a result the link cannot carry fails that request instead.

        Only ``Closed`` and ``OSError`` mean the link is gone. A result that is
        too large, or that does not pickle, was not sent, and its request is
        answered with the error instead.
        """
        try:
            self.link.send(message)
        except (Closed, OSError):
            raise
        except Exception as exc:
            if not isinstance(message, Result):
                raise
            logger.warning("a result cannot be sent", error=str(exc), **self.where())
            self.link.send(failure(message.request_id, exc))
