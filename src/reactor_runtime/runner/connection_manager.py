"""Connection manager — the live-connection multiplexer for one session.

A registry of the connections currently attached to a session, keyed by id, that
drives the session machine on the edges that matter and fans the model's outbound
traffic across the wires. It holds the session through its busy states: the first
connection to arrive moves it to streaming, the last to leave moves it to orphaned,
and every open and close in between rides the machine as a self-loop the runner
turns into a client event.

It is deliberately wire-blind. Connections enter as the neutral ``Connection``
shape and the manager only ever calls that shape — it never asks which transport
produced a connection, which is exactly what lets one session mix transports.
"""

from __future__ import annotations

from reactor_runtime.core import (
    Connection,
    ConnId,
    MediaBundle,
    SessionEvent,
)
from reactor_runtime.runner.state_machine import SessionStateMachine


class ConnectionManager:
    """Registry and multiplexer of the live connections in one session.

    Owns the ``{id: Connection}`` map, advances the session state machine on the
    first and last connection, arbitrates publisher tracks first-come-first-served,
    and exposes the broadcast / addressed / media sends the model's outbound path
    binds to. Constructed with the machine it drives; nothing here is async.
    """

    def __init__(self, *, state_machine: SessionStateMachine) -> None:
        """Bind the manager to the session machine it advances."""
        self._sm = state_machine
        self._by_id: dict[ConnId, Connection] = {}
        # The owner of each published track. First publisher wins: a track is held
        # by one connection until it releases or drops, and a later claim on a held
        # track is refused.
        self._publishers: dict[str, ConnId] = {}
        # Every id ever minted this session, so a fresh id never collides with a
        # live or a since-dropped connection. The manager owns the id namespace
        # because it is the one stateful, connection-keyed component.
        self._used_conn_ids: set[ConnId] = set()

    @property
    def count(self) -> int:
        """The number of connections currently registered."""
        return len(self._by_id)

    def new_conn_id(self) -> ConnId:
        """Mint a fresh connection id, unique within the session.

        Allocated centrally so connections arriving through several transports
        cannot collide on an id, and monotonic so a new connection never reuses
        the id of one that has since dropped.
        """
        conn_id = ConnId(max(self._used_conn_ids, default=0) + 1)
        self._used_conn_ids.add(conn_id)
        return conn_id

    def register(self, conn: Connection) -> None:
        """Add a connection and advance the session for it.

        The first connection into an empty session sends ``CLIENT_CONNECTED``,
        moving the session into streaming; every registration, first or not, then
        sends ``CONNECTION_OPENED`` so the runner can announce the individual
        client. Re-registering an id already present replaces the handle without
        re-driving the session — connection identity across a reconnect is the
        transport's concern, not the manager's.
        """
        if conn.id in self._by_id:
            self._by_id[conn.id] = conn
            return
        first = self.count == 0
        self._by_id[conn.id] = conn
        if first:
            self._sm.send(SessionEvent.CLIENT_CONNECTED)
        self._sm.send(SessionEvent.CONNECTION_OPENED, conn_id=conn.id)

    def drop(self, cid: ConnId) -> None:
        """Remove a connection and advance the session for its loss.

        Sends ``CONNECTION_CLOSED`` for the connection, and when it was the last
        one, ``CLIENT_DISCONNECTED`` to move the session into orphaned — in that
        order, because the close is a streaming self-loop that the move to orphaned
        would otherwise make illegal. Any tracks the connection still held are
        released. A drop for an id that is not registered is ignored.
        """
        if cid not in self._by_id:
            return
        del self._by_id[cid]
        self._sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=cid)
        if self.count == 0:
            self._sm.send(SessionEvent.CLIENT_DISCONNECTED)
        held = [name for name, owner in self._publishers.items() if owner == cid]
        for name in held:
            del self._publishers[name]

    async def close_all(self) -> None:
        """Close every connection and empty the registry, for session teardown.

        Used when the session itself is ending rather than a single client
        leaving: the registry is cleared and every wire is closed wholesale. It
        does not drive the session machine — the open/close moves are illegal once
        the session has left its running states — and does not announce
        per-connection losses, because the model learns the session has ended
        through the session-end reactor event, not one disconnect per client. The
        registry is emptied before the closes are awaited so the manager presents
        as session-less the moment teardown begins.
        """
        conns = list(self._by_id.values())
        self._by_id.clear()
        self._publishers.clear()
        for conn in conns:
            await conn.close()

    def publish_track(self, cid: ConnId, name: str) -> bool:
        """Claim the publisher slot for a track on behalf of a connection.

        First publisher wins: the claim is granted only when the track is unheld
        and the connection is registered, in which case the connection's outbound
        track is resumed. A claim on a track already held — by any connection,
        including the same one — is refused.

        Returns:
            ``True`` when the slot was granted, ``False`` when refused.
        """
        conn = self._by_id.get(cid)
        if conn is None or name in self._publishers:
            return False
        self._publishers[name] = cid
        conn.resume_track(name)
        return True

    def unpublish_track(self, cid: ConnId, name: str) -> None:
        """Release a track previously claimed by this connection.

        Only the holding connection can release its track; a release from any other
        connection is ignored. On release the holder's outbound track is paused and
        the slot is free for the next claim.
        """
        if self._publishers.get(name) != cid:
            return
        del self._publishers[name]
        conn = self._by_id.get(cid)
        if conn is not None:
            conn.pause_track(name)

    def broadcast(self, payload: bytes | str) -> None:
        """Send an encoded frame to every connection."""
        for conn in self._by_id.values():
            conn.send_message(payload)

    def send(self, cid: ConnId, payload: bytes | str) -> None:
        """Send an encoded frame to one connection, if it is registered."""
        conn = self._by_id.get(cid)
        if conn is not None:
            conn.send_message(payload)

    def broadcast_media(self, bundle: MediaBundle, duplicate: bool) -> None:
        """Send a media bundle to every connection whose wire carries media.

        Data-only connections are skipped rather than relying on a silent no-op, so
        a media bundle only reaches a wire that can deliver it. ``duplicate`` marks
        a re-emitted gap-fill bundle for downstream consumers (such as recording);
        the multiplexer forwards every bundle to all media-capable connections and
        does not branch on it.
        """
        for conn in self._by_id.values():
            caps = conn.capabilities
            if caps.carries_video or caps.carries_audio:
                conn.send_media(bundle)

    def note_keepalive(self, cid: ConnId) -> None:
        """Record a per-connection liveness ping.

        A keepalive is a fact about one connection's wire, which owns its own
        liveness detection; it does not advance the session. Session-level liveness
        is the runner's own concern, emitted on its session ticker rather than off
        any single connection's pings, so this neither moves the machine nor
        depends on whether the connection is still registered.
        """
