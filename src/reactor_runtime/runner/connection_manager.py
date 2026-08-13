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

Threading
---------
Writers run on one thread and readers run on several, so the registry is
replaced rather than edited.

* Every write — a register, a drop, a teardown — arrives on the runtime's event
  loop, because a transport reports a connection opening or closing through the
  sink from that loop. Writers therefore never race each other, and the manager
  holds no lock.
* Reads arrive from any thread. The model's own thread broadcasts messages and
  drives the playout controls, and a worker thread off that loop fans each
  emitted media chunk out, so a fan-out is iterating the registry while a
  client's disconnect is landing on the loop.
* The registry is bound as an immutable ``Mapping`` and swapped for a fresh one
  on every change. A reader takes one attribute load and iterates a mapping
  nothing will ever mutate, so a disconnect mid-fan-out is invisible to it — the
  chunk simply reaches a connection that has just gone, whose stopped pacer no
  longer puts anything on the wire. A lock would be worse than unnecessary
  here: a fan-out can block for seconds inside a connection's pacer waiting for
  queue room, and a lock spanning that wait would hold the event loop out of
  the very disconnect it needs to process.
* The publisher table and the used-id pool stay ordinary mutable containers.
  Both are read and written only on the event loop, and nothing iterates them
  from another thread.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping

from reactor_runtime.core import (
    Connection,
    ConnId,
    MediaChunk,
    SessionEvent,
)
from reactor_runtime.log import get_logger
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.runner.state_machine import SessionStateMachine
from reactor_runtime.transport.router import ConnectionsExhaustedError

logger = get_logger(__name__)

# Connection ids are minted at random in this inclusive range, the same id space
# a production director hands out. 1000 is invalid and 1001 is reserved for
# legacy single-connection compatibility, so explicit ids start at 1002.
_MIN_CONN_ID = 1002
_MAX_CONN_ID = 9999

# The most times a mint redraws before giving up. Bounding the draw count keeps
# allocation from spinning the event loop when the id space is dense, without
# imposing a lifetime allowance that would let a client lock out registration by
# minting far below the space. With any realistic number of live-plus-retained
# ids the first draw is almost always free, so this is only ever approached when
# the space is genuinely near-full.
_MAX_MINT_ATTEMPTS = 100


def _deliver(conn: Connection, operation: str, act: Callable[[Connection], None]) -> None:
    """Apply one fan-out step to one connection, containing what it raises.

    A fan-out reaches every client in the session, so a failure on one wire is
    the fan-out's to absorb: the remaining connections are still owed the frame,
    and the caller — often the model's own thread, mid-``emit`` — has no wire of
    its own to fail. Left to propagate, one connection's exception would surface
    as a crash of the model's run loop and end the session for everyone on it.
    The failure is logged with its traceback so a wire that fails every time is
    still visible rather than silently dark.
    """
    try:
        act(conn)
    except Exception:
        logger.exception("connection rejected an outbound fan-out", operation=operation)


class ConnectionManager:
    """Registry and multiplexer of the live connections in one session.

    Owns the ``{id: Connection}`` map, advances the session state machine on the
    first and last connection, arbitrates publisher tracks first-come-first-served,
    and exposes the broadcast / addressed / media sends the model's outbound path
    binds to. Constructed with the machine it drives; nothing here is async.

    Reads are safe from any thread and writes belong to the runtime's event
    loop; see the module docstring for what that buys and what it costs.
    """

    def __init__(self, *, state_machine: SessionStateMachine) -> None:
        """Bind the manager to the session machine it advances."""
        self._sm = state_machine
        # Bound as a read-only mapping and swapped whole on every change, so a
        # reader on another thread iterates a snapshot that cannot move under
        # it. The type is what holds the invariant: an in-place write here is a
        # type error rather than a race discovered in production.
        self._by_id: Mapping[ConnId, Connection] = {}
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

        Raises:
            ConnectionsExhaustedError: If a free id is not drawn within the
                bounded number of attempts, which only happens when the id space
                is near-full. Raising bounds the draw: it never loops looking for
                a free id in a full pool.
        """
        for _ in range(_MAX_MINT_ATTEMPTS):
            conn_id = ConnId(random.randint(_MIN_CONN_ID, _MAX_CONN_ID))
            if conn_id not in self._used_conn_ids:
                self._used_conn_ids.add(conn_id)
                return conn_id
        raise ConnectionsExhaustedError

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
        known = conn.id in self._by_id
        self._by_id = {**self._by_id, conn.id: conn}
        if not known:
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
        self._by_id = {other: conn for other, conn in self._by_id.items() if other != cid}
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
        self._by_id = {}
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
            _deliver(conn, "broadcast", lambda c: c.send_message(encode(c.protocol_version)))

    def send(self, cid: ConnId, encode: Callable[[ProtocolVersion], bytes | str]) -> None:
        """Encode and send a frame to one connection in its codec, if registered."""
        conn = self._by_id.get(cid)
        if conn is not None:
            conn.send_message(encode(conn.protocol_version))

    def send_command_ack(
        self, cid: ConnId, encode: Callable[[ProtocolVersion], bytes | str]
    ) -> None:
        """Send a command acknowledgement to one connection, if its codec carries one.

        The ack correlates a client's awaited command to its completion on the
        data channel. A legacy (v0) client issues fire-and-forget commands and
        expects no reply, so it is sent none.
        """
        conn = self._by_id.get(cid)
        if conn is None or conn.protocol_version is ProtocolVersion.V0:
            return
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
        self._send_on_channel(conn, encode)

    def broadcast_response(
        self, encode: Callable[[ProtocolVersion], tuple[Channel, bytes | str]]
    ) -> None:
        """Encode and send an unsolicited server frame to every connection.

        The all-connections analogue of :meth:`send_response`: each connection
        receives the frame encoded for its negotiated codec, on the physical
        channel that codec picks for it. A runtime-authored notice with no single
        addressee — a moderation verdict — rides this.
        """
        for conn in self._by_id.values():
            _deliver(conn, "broadcast_response", lambda c: self._send_on_channel(c, encode))

    @staticmethod
    def _send_on_channel(
        conn: Connection, encode: Callable[[ProtocolVersion], tuple[Channel, bytes | str]]
    ) -> None:
        """Encode a frame for one connection and route it to the channel picked."""
        channel, frame = encode(conn.protocol_version)
        if channel is Channel.CONTROL:
            conn.send_control(frame)
        else:
            conn.send_message(frame)

    def broadcast_media(
        self, chunk: MediaChunk, *, abort: Callable[[], bool] | None = None
    ) -> None:
        """Send a media chunk to every connection whose wire carries media.

        Data-only connections are skipped rather than relying on a silent no-op, so
        a chunk only reaches a wire that can deliver it. Each connection paces the
        chunk itself, so this fans the same unpaced chunk out and does no timing.

        Args:
            chunk: The unpaced chunk to fan out.
            abort: Checked before each connection; a truthy result abandons
                the rest of the fan-out. The runner points it at its flush
                generation so a chunk flushed mid-broadcast reaches no
                further connection.
        """
        for conn in self._by_id.values():
            if abort is not None and abort():
                return
            caps = conn.capabilities
            if caps.carries_video or caps.carries_audio:
                _deliver(conn, "broadcast_media", lambda c: c.send_media(chunk))

    def flush_media(self) -> None:
        """Drop every connection's queued media and cut playout to black."""
        for conn in self._by_id.values():
            _deliver(conn, "flush_media", lambda c: c.flush_media())

    def set_media_rate(self, fps: float) -> None:
        """Re-pace every connection's queued media at *fps* immediately."""
        for conn in self._by_id.values():
            _deliver(conn, "set_media_rate", lambda c: c.set_media_rate(fps))

    def set_media_depth(self, depth: int) -> None:
        """Bound every connection's media queue at *depth* frames."""
        for conn in self._by_id.values():
            _deliver(conn, "set_media_depth", lambda c: c.set_media_depth(depth))

    def note_keepalive(self, cid: ConnId) -> None:
        """Record a per-connection liveness ping.

        A keepalive is a fact about one connection's wire, which owns its own
        liveness detection; it does not advance the session. Session-level liveness
        is the runner's own concern, emitted on its session ticker rather than off
        any single connection's pings, so this neither moves the machine nor
        depends on whether the connection is still registered.
        """
