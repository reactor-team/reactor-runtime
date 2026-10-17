"""Transport-boundary protocols.

The two structural protocols that form the seam between a transport and the
runner, pointing opposite ways. ``Connection`` carries commands down to one
client; ``ConnectionSink`` carries facts up from the transport. Both are
``Protocol``s, so a transport author or a test fake conforms by shape and
signaling never reaches the runner. ``TransportStatsSource`` and
``FrameStageSource`` are optional capabilities a connection adds when its
transport measures its wire, and ``MessageObserver`` one it adds when its wire
needs what it is sent as plain data rather than as encoded frames.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Protocol, runtime_checkable

from reactor_runtime.core.stats import TransportReading
from reactor_runtime.core.values import (
    ClientStatsBatch,
    ConnectionCapabilities,
    ConnId,
    FrameStage,
    InputFrame,
    MediaChunk,
    MessageStage,
)
from reactor_runtime.protocol import Channel, ProtocolVersion


@runtime_checkable
class Connection(Protocol):
    """One client connection, regardless of wire — the commands-down half.

    A neutral handle the connection manager holds and the runner sends through.
    Implemented concretely per transport: it moves opaque bytes and media on its
    wire, arbitrates publisher tracks, and owns its own liveness detection. The
    runner only ever holds it as this shape and cannot tell one wire from
    another.

    The object exists before it is live: a transport may build it during its
    handshake (for WebRTC, while negotiating SDP/ICE) and hold it itself. It is
    reported up to the sink only once its wire is actually connected, so a
    half-open connection never reaches the runner.
    """

    id: ConnId
    capabilities: ConnectionCapabilities

    @property
    def protocol_version(self) -> ProtocolVersion:
        """The wire codec negotiated for this connection at its handshake.

        Fixed for the connection's life and read-only here. The runner reads it
        to encode each outbound frame for the codec this client speaks, so a
        mixed-version session addresses each connection in its own version.
        """

    def send_message(self, payload: bytes | str) -> None:
        """Send an already-encoded frame to this client.

        A frame is text or binary depending on the wire version — a v0 JSON
        frame is ``str`` (a text frame), a v1 protobuf frame is ``bytes`` (a
        binary frame) — so the connection carries both.
        """

    def send_media(self, chunk: MediaChunk) -> None:
        """Hand a media chunk to the connection, or do nothing when it carries no media.

        The chunk is unpaced — a batch of frames tagged with the rate they should
        play out at. The connection owns pacing it to a steady wire cadence; the
        runner never sees a single frame. A chunk that asks for backpressure
        (``chunk.wait``) may make this call block until the wire has drained
        room for it.
        """

    def flush_media(self, *, hold: bool = False) -> None:
        """Drop this connection's queued media, then cut playout to black or hold it.

        Drops what the connection holds for its wire as well as what its pacer
        queues, audio included, so nothing queued before the flush plays after
        it. With *hold*, no black frame is sent and the last frame stays on
        screen until new media arrives. A plain flush is called with no
        argument.
        """

    def set_media_rate(self, fps: float) -> None:
        """Re-pace this connection's queued media at *fps* immediately."""

    def set_media_depth(self, depth: int) -> None:
        """Bound how many frames may queue between the model and this wire."""

    def resume_track(self, name: str) -> None:
        """Resume the named outbound track (publisher arbitration)."""

    def pause_track(self, name: str) -> None:
        """Pause the named outbound track (publisher arbitration)."""

    def send_control(self, payload: bytes | str) -> None:
        """Send an already-encoded control frame to this client.

        The control channel carries the runtime's replies to a client's control
        requests — a publish-track grant or refusal — encoded for the wire
        version this connection negotiated.
        """

    async def close(self) -> None:
        """Tear the connection down."""


@runtime_checkable
class TransportStatsSource(Protocol):
    """A connection whose transport samples statistics about its own wire.

    Optional, beside :class:`Connection`: a transport that measures its wire
    implements it on its connections, and a reader checks for it with
    ``isinstance`` before reading. A connection without it is a connection
    with nothing to report, not an error, so a transport that predates it
    keeps working unchanged.
    """

    @property
    def latest_reading(self) -> TransportReading | None:
        """The most recent reading of this connection's wire.

        ``None`` before the transport has taken its first. The reading is
        replaced as the transport takes new ones, so reading it never waits on
        the wire.
        """


@runtime_checkable
class FrameStageSource(Protocol):
    """A connection whose transport times each frame's stages on its side of the wire.

    Optional, beside :class:`TransportStatsSource`, and checked for the same
    way. A track's times add up until they are taken, so each take covers the
    window since that track's previous one.
    """

    def take_frame_stages(self, tracks: Collection[str]) -> Mapping[str, tuple[FrameStage, ...]]:
        """Return the stage times of *tracks* since their previous take, and start them again.

        Keyed by track name. A track with no frames in its window is left out,
        and so is a stage with no frames in it. Tracks not named keep their
        times for a later take.
        """


@runtime_checkable
class MessageStageSource(Protocol):
    """A connection whose transport times its data-channel messages stage by stage.

    Optional, beside :class:`FrameStageSource`, and checked for the same way.
    The times add up until they are taken, so each take covers the window since
    the previous one.
    """

    def take_message_stages(self) -> tuple[MessageStage, ...]:
        """Return the message stage times since the previous take, and start them again."""


@runtime_checkable
class MessageObserver(Protocol):
    """A connection that also takes what it is sent as plain data.

    Optional, beside :class:`Connection`, and checked for with ``isinstance``.
    A connection whose wire does not carry the Reactor protocol, such as one
    that answers a command as a remote procedure call, needs the content of a
    reply or a model message rather than its encoded frame. The runner calls
    these next to the encoded send, on every connection that implements them,
    from whichever thread sent the message: the model's own, for its replies
    and messages.
    """

    def message_sent(self, name: str, data: Mapping[str, Any], request_id: str | None) -> None:
        """Take a model message sent to this connection.

        *request_id* is set when the message is the reply to a command this
        connection sent, and ``None`` for a broadcast or a ``client.send``.
        *data* holds the message's fields, without the ``type`` envelope.
        """

    def command_acknowledged(self, request_id: str) -> None:
        """Take the end of a command this connection sent whose handler returned nothing."""

    def command_failed(self, request_id: str, code: str, message: str) -> None:
        """Take the failure of a command this connection sent.

        Covers every refusal the client is told about: arguments the contract
        rejects, an upload that cannot be resolved, and a handler that raised.
        *code* is the stable failure code, such as ``invalid_command`` or the
        code of a raised ``CommandError``.
        """


@runtime_checkable
class ConnectionSink(Protocol):
    """The transport-to-runner upward channel — the facts-up half.

    A transport pushes facts through these as connections come and go and as
    frames arrive; the runner never pulls signaling. Implemented by the runner,
    so several transports can point at the same sink and land in one registry —
    the basis for mixed-transport sessions.
    """

    def connection_opened(self, conn: Connection) -> None:
        """Register a connection whose wire can carry frames to the client.

        Fired when the transport is actually connected — for WebRTC, when the
        peer connection is connected and both data channels are open — never at
        handshake or offer time. A transport that builds a connection during
        signaling holds it itself until then, so a client that offers but never
        connects never advances the session. Because the wire is open by the
        time this fires, a message the model sends from its ``@connected`` hook
        reaches the joining client.
        """

    def connection_closed(self, conn_id: ConnId) -> None:
        """Drop a previously opened connection that has gone away.

        Reports the loss of a connection that was reported via
        ``connection_opened``. A connection that fails before it ever connects is
        discarded inside the transport and is never seen here.
        """

    def message_received(
        self, conn_id: ConnId, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        """Hand an inbound encoded frame up for decoding and dispatch.

        The frame is text or binary as the wire version dictates (``str`` for a
        v0 JSON frame, ``bytes`` for a v1 protobuf frame), mirroring what the
        codec decodes. *version* is the codec the connection negotiated, so the
        frame is decoded as the client that sent it speaks — per connection,
        not a single runtime-wide codec. *channel* is the physical channel it
        arrived on (data or control), so the gateway decodes it as the right
        message family rather than guessing.
        """

    def command_received(
        self, conn_id: ConnId, name: str, args: Mapping[str, Any], request_id: str
    ) -> None:
        """Hand up a command that arrived outside the Reactor protocol.

        For a transport that maps a signal of its own wire, such as an RPC
        call or a data packet, onto one of the model's commands. The command
        goes through the same contract validation, upload resolution, and
        dispatch as one decoded from a frame. The transport chooses
        *request_id*, and a connection that is a :class:`MessageObserver`
        receives the command's reply, acknowledgement, or failure under it.
        """

    def media_received(self, conn_id: ConnId, track: str, frame: InputFrame) -> None:
        """Hand an inbound media frame up for the named track."""

    def keepalive(self, conn_id: ConnId) -> None:
        """Note liveness for a connection."""

    def resume_track(self, conn_id: ConnId, name: str) -> None:
        """Resume an outbound track for one connection, at the client's request."""

    def pause_track(self, conn_id: ConnId, name: str) -> None:
        """Pause an outbound track for one connection, at the client's request."""

    def publish_requested(self, conn_id: ConnId, name: str, request_id: str) -> None:
        """Arbitrate a client's claim to publish an inbound track, and reply.

        First publisher wins across the session; the outcome is sent back to the
        requesting connection on its control channel, correlated by *request_id*.
        """

    def unpublish_track(self, conn_id: ConnId, name: str) -> None:
        """Release an inbound track a connection had claimed."""

    def file_uploaded(self, conn_id: ConnId, upload_id: str) -> None:
        """Note a client's out-of-band file upload.

        Fired when a client signals it has finished uploading a file outside any
        command (the ``fileUploaded`` notification, carrying the id of the
        uploaded slot). The runner fetches the stored bytes and hands the model
        a file-uploaded reactor event; a model with no upload hook ignores it.
        """

    def schema_requested(self, conn_id: ConnId, request_id: str) -> None:
        """Answer a client's request for the model schema.

        The rendered schema is sent back to the requesting connection,
        correlated by *request_id*, on whichever channel the connection's wire
        version places it (data for v0, control for v1).
        """

    def clip_requested(self, conn_id: ConnId, duration_seconds: float, request_id: str) -> None:
        """Answer a client's request for a snap clip of recent output.

        The recorder resolves the last *duration_seconds* to a marker range and a
        playlist URL the client polls, and the result is sent back to the
        requesting connection correlated by *request_id*. A request the recorder
        cannot serve is answered with a clip-failed reply instead.
        """

    def recording_requested(self, conn_id: ConnId, request_id: str) -> None:
        """Answer a client's request for a recording of the whole session.

        Resolves the same way as :meth:`clip_requested` but over the full session
        timeline, correlated by *request_id*.
        """

    def client_stats_received(self, conn_id: ConnId, batch: ClientStatsBatch) -> None:
        """Hand up a batch of client-observed quality samples for one connection.

        The client is the only vantage point onto its own receive-side
        quality, so a transport reports the batch as decoded rather than
        validating it. The connection identity comes from *conn_id*, which the
        transport already holds — never from anything the payload itself
        might claim, because the payload carries none.
        """

    def connection_answered(self, conn_id: ConnId, answer: Mapping[str, str]) -> None:
        """Relay the transport's negotiation answer for one connection.

        Fired once the transport has produced the answer to a client's
        connection offer (for WebRTC, the SDP answer as a ``{"type", "sdp"}``
        mapping). The payload is transport-agnostic and opaque here: the runner
        records it as a self-loop session move that journals the answer for an
        external consumer to hand back to the client, and never parses it.
        Distinct from ``connection_opened``, which fires later when the wire
        actually connects.
        """
