"""Session epochs for admitted connection offers."""

from __future__ import annotations

from reactor_runtime.core import ConnId


class OfferEpochs:
    """Track which session each admitted connection offer belongs to.

    Negotiation is asynchronous, so a wire can reach its connected state after
    the session that admitted its offer ended. While no session runs, the
    session state exposes that; once the next session starts, the state looks
    valid again and cannot. The epoch counts sessions: each admitted offer is
    stamped with the live epoch, and the wire is checked against the current
    one when it connects.

    Bounded by the connection-id space: a re-offer on the same id restamps it,
    and a checked stamp is dropped.
    """

    def __init__(self) -> None:
        """Start before any session, with no offers stamped."""
        self._current = 0
        self._stamps: dict[ConnId, int] = {}

    def session_started(self) -> None:
        """Move to the next epoch; offers stamped from now on belong to it."""
        self._current += 1

    def stamp(self, conn_id: ConnId) -> None:
        """Stamp an admitted offer with the live epoch. A re-offer restamps."""
        self._stamps[conn_id] = self._current

    def consume(self, conn_id: ConnId) -> bool:
        """Take the stamp for *conn_id* and report whether it is stale.

        Returns ``True`` when the offer was stamped in an earlier epoch, so
        its wire belongs to a session that ended. An unstamped id is not
        stale: a transport that does not stamp offers is judged on session
        state alone. The stamp is consumed either way.
        """
        stamped = self._stamps.pop(conn_id, None)
        return stamped is not None and stamped != self._current
