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

import random
from collections.abc import Callable

from reactor_runtime.core import (
    Connection,
    ConnId,
    MediaBundle,
    SessionEvent,
)
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.runner.state_machine import SessionStateMachine

# Connection ids are minted at random in this inclusive range, the same id space
# a production director hands out. 1000 is invalid and 1001 is reserved for
# legacy single-connection compatibility, so explicit ids start at 1002.
_MIN_CONN_ID = 1002
_MAX_CONN_ID = 9999


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
        # Ids minted within the current session, so a fresh id never collides
        # with a live or a since-dropped connection. Scoped to the session: it is
        # cleared on teardown so ids do not accumulate across sessions and the
        # full range is available again to the next one.
        self._used_conn_ids: set[ConnId] = set()

    @property
    def count(self) -> int:
        """The number of connections currently registered."""
        return len(self._by_id)

    def new_conn_id(self) -> ConnId:
        """Mint a fresh random connection id, unique within the session.

        Drawn at random from ``[1002, 9999]`` and retried until one is free,
        matching the id space a production director allocates. Allocated centrally
        so connections arriving through several transports cannot collide, and
        held in the session's used-id pool so a new connection never reuses the id
        of one that has since dropped; the pool clears on session teardown.
        """
        while True:
            conn_id = ConnId(random.randint(_MIN_CONN_ID, _MAX_CONN_ID))
            if conn_id not in self._used_conn_ids:
                self._used_conn_ids.add(conn_id)
                return conn_id

    def register(self, conn: Connection) -> None:
        """Add a connection and advance the session for it.

        A fresh registration sends a single ``CONNECTION_OPENED``; the state
        machine derives occupancy from it, carrying an empty session into
        streaming on the first connection and self-looping for the rest. The
        handle is added before the event so a listener reading the live count
        sees this connection counted. Re-registering an id already present
        replaces the handle without re-driving the session — connection identity
        across a reconnect is the transport's concern, not the manager's.
        """
        if conn.id in self._by_id:
            self._by_id[conn.id] = conn
            return
        self._by_id[conn.id] = conn
        self._sm.send(SessionEvent.CONNECTION_OPENED, conn_id=conn.id)

    def drop(self, cid: ConnId) -> None:
        """Remove a connection and advance the session for its loss.

        Sends a single ``CONNECTION_CLOSED``; the state machine derives occupancy
        from it, carrying the session into orphaned when the last connection
        leaves and self-looping while others remain. The handle is removed before
        the event so a listener reading the live count sees this connection gone.
        Any tracks the connection still held are released. A drop for an id that
        is not registered is ignored.
        """
        if cid not in self._by_id:
            return
        del self._by_id[cid]
        self._sm.send(SessionEvent.CONNECTION_CLOSED, conn_id=cid)
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
        as session-less the moment teardown begins. The used-id pool is cleared
        too, so the next session starts with the whole id range free again.
        """
        conns = list(self._by_id.values())
        self._by_id.clear()
        self._publishers.clear()
        self._used_conn_ids.clear()
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

    def resume_track(self, cid: ConnId, name: str) -> None:
        """Resume an outbound track on one connection, at the client's request.

        Per-connection: each client gates its own reception of the model's
        outbound tracks. A request for an unregistered connection is ignored.
        """
        conn = self._by_id.get(cid)
        if conn is not None:
            conn.resume_track(name)

    def pause_track(self, cid: ConnId, name: str) -> None:
        """Pause an outbound track on one connection, at the client's request.

        Per-connection: each client gates its own reception of the model's
        outbound tracks. A request for an unregistered connection is ignored.
        """
        conn = self._by_id.get(cid)
        if conn is not None:
            conn.pause_track(name)

    def broadcast(self, encode: Callable[[ProtocolVersion], bytes | str]) -> None:
        """Encode and send a frame to every connection in its own codec.

        *encode* renders the outbound frame for a given wire version. Each
        connection is sent the frame encoded for the codec it negotiated, so a
        mixed-version session reaches every client in the version it speaks.
        """
        for conn in self._by_id.values():
            conn.send_message(encode(conn.protocol_version))

    def send(self, cid: ConnId, encode: Callable[[ProtocolVersion], bytes | str]) -> None:
        """Encode and send a frame to one connection in its codec, if registered."""
        conn = self._by_id.get(cid)
        if conn is not None:
            conn.send_message(encode(conn.protocol_version))

    def send_control(self, cid: ConnId, encode: Callable[[ProtocolVersion], bytes | str]) -> None:
        """Encode and send a control frame to one connection in its codec, if registered.

        The runtime's reply to a client's control request — a publish-track
        grant or refusal — rides this, encoded for the codec the connection
        negotiated.
        """
        conn = self._by_id.get(cid)
        if conn is not None:
            conn.send_control(encode(conn.protocol_version))

    def send_response(
        self, cid: ConnId, encode: Callable[[ProtocolVersion], tuple[Channel, bytes | str]]
    ) -> None:
        """Encode and send a server reply to one connection on the channel the codec picks.

        A reply's physical channel is version-dependent — a v0 platform reply
        (e.g. the model schema) rides the data channel, while v1 places it on the
        control channel — so *encode* returns the channel alongside the frame and
        the manager routes to the matching send.
        """
        conn = self._by_id.get(cid)
        if conn is None:
            return
        channel, frame = encode(conn.protocol_version)
        if channel is Channel.CONTROL:
            conn.send_control(frame)
        else:
            conn.send_message(frame)

    def broadcast_media(self, bundle: MediaBundle, is_fresh_black: bool) -> None:
        """Send a media bundle to every connection whose wire carries media.

        Data-only connections are skipped rather than relying on a silent no-op, so
        a media bundle only reaches a wire that can deliver it. ``is_fresh_black``
        is the flag the model bridge's media sink forwards — the synthesised black
        frame emitted at a session boundary; the multiplexer sends every bundle to
        all media-capable connections and does not branch on it.
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
