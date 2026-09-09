"""WebRTC as a connection.

:class:`WebRTCConnection` is the concrete
:class:`~reactor_runtime.core.transport.Connection` for the WebRTC wire. It moves
encoded messages and media through its peer, arbitrates publisher tracks, owns
its own liveness through a ping watchdog, and samples connection stats — all the
per-connection behaviour, with none of the signalling, which lives in the
acceptor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import replace

from reactor_runtime.core import (
    ConnectionCapabilities,
    ConnId,
    InputFrame,
    MediaChunk,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.protocol import Channel, ProtocolVersion
from reactor_runtime.transport.webrtc.config import WebRtcConfig
from reactor_runtime.transport.webrtc.pacer import MediaPacer
from reactor_runtime.transport.webrtc.peer import WebRTCPeer, WebRtcPeerFactory
from reactor_runtime.transport.webrtc.signaling import IceCandidate, SdpAnswer, SdpOffer, TrackMap
from reactor_runtime.transport.webrtc.stats import OutboundMediaHealth, PeerStats

logger = logging.getLogger(__name__)


def _capabilities_for(tracks: TrackMap) -> ConnectionCapabilities:
    """Derive a connection's outbound capabilities from the model's tracks."""
    outbound = tracks.by_direction(TrackDirection.OUT)
    return ConnectionCapabilities(
        carries_video=any(t.kind is TrackKind.VIDEO for t in outbound),
        carries_audio=any(t.kind is TrackKind.AUDIO for t in outbound),
    )


def _outbound_video_tracks(tracks: TrackMap) -> dict[str, TrackInfo]:
    """Return the outbound video tracks the pacer synthesises black frames for."""
    return {t.name: t for t in tracks.by_direction(TrackDirection.OUT) if t.kind is TrackKind.VIDEO}


class WebRTCConnection:
    """One WebRTC client connection, driven through its media peer.

    Built by the acceptor during negotiation and held until its wire connects.
    The acceptor registers the inbound callbacks before the wire goes live; the
    connection starts its watchdog and stats sampling on connect and stops them
    on loss or close.
    """

    _WATCHDOG_POLL_SECONDS = 2.0
    _STATS_INTERVAL_SECONDS = 2.0

    def __init__(
        self,
        conn_id: ConnId,
        peer: WebRTCPeer,
        capabilities: ConnectionCapabilities,
        *,
        ping_timeout: float,
        video_tracks: dict[str, TrackInfo] | None = None,
    ) -> None:
        """Bind the connection to its peer and outbound capabilities.

        Args:
            conn_id: The centrally-minted id for this connection.
            peer: The media engine driving the wire.
            capabilities: What media this connection's wire can carry outbound.
            ping_timeout: Seconds without a client ping before the watchdog
                declares the connection lost; ``0`` or less disables it.
            video_tracks: The connection's outbound video tracks, so its pacer can
                synthesise black frames before the first real one arrives.
        """
        self.id = conn_id
        self.capabilities = capabilities
        self._peer = peer
        self._ping_timeout = ping_timeout
        video_tracks = video_tracks or {}
        initial_fps = max((t.rate for t in video_tracks.values() if t.rate > 0), default=30.0)
        self._pacer = MediaPacer(video_tracks, self._peer.send_media, fps=initial_fps)

        self._on_message: Callable[[bytes | str, ProtocolVersion, Channel], None] | None = None
        self._on_media: Callable[[str, InputFrame], None] | None = None
        self._on_ping: Callable[[], None] | None = None
        self._on_connected: Callable[[], None] | None = None
        self._on_disconnect: Callable[[], None] | None = None
        self._on_closed: Callable[[], None] | None = None
        self._on_stats: Callable[[PeerStats], None] | None = None

        self._alive = True
        self._last_ping: float | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._stats_task: asyncio.Task[None] | None = None
        self._latest_stats: PeerStats | None = None
        # The counters as of the previous sample, so each report is what the
        # window cost rather than what the connection has cost since it opened.
        self._reported_media = OutboundMediaHealth()

    @classmethod
    async def create(
        cls,
        conn_id: ConnId,
        offer: SdpOffer,
        tracks: TrackMap,
        config: WebRtcConfig,
        version: ProtocolVersion,
        *,
        peer_factory: WebRtcPeerFactory,
    ) -> tuple[WebRTCConnection, SdpAnswer]:
        """Negotiate a peer for *offer* and wrap it as a connection.

        Returns the connection and the SDP answer the exchange produced. The
        peer is built for *version*, the wire codec negotiated for this
        connection, which it holds for its life. The connection subscribes to
        its peer's inbound events here, so it is ready to forward facts the
        moment the acceptor has registered its callbacks.
        """
        peer, answer = await peer_factory(conn_id, offer, tracks, config, version)
        conn = cls(
            conn_id,
            peer,
            _capabilities_for(tracks),
            ping_timeout=config.ping_timeout,
            video_tracks=_outbound_video_tracks(tracks),
        )
        conn._subscribe()
        return conn, answer

    def _subscribe(self) -> None:
        """Wire this connection onto its peer's inbound events."""
        self._peer.on_message(self._handle_message)
        self._peer.on_media(self._handle_media)
        self._peer.on_ping(self._handle_ping)
        self._peer.on_connected(self._handle_connected)
        self._peer.on_disconnect(self._report_loss)

    @property
    def latest_stats(self) -> PeerStats | None:
        """The most recent stats sample, or ``None`` before the first cycle."""
        return self._latest_stats

    @property
    def protocol_version(self) -> ProtocolVersion:
        """The wire codec the connection negotiated, held by its peer."""
        return self._peer.protocol_version

    def on_message(self, callback: Callable[[bytes | str, ProtocolVersion, Channel], None]) -> None:
        """Register the sink for inbound frames, with the codec version and channel."""
        self._on_message = callback

    def on_media(self, callback: Callable[[str, InputFrame], None]) -> None:
        """Register the sink for inbound media frames, by track name."""
        self._on_media = callback

    def on_ping(self, callback: Callable[[], None]) -> None:
        """Register the sink for client liveness pings."""
        self._on_ping = callback

    def on_connected(self, callback: Callable[[], None]) -> None:
        """Register the sink for the wire reaching its connected state."""
        self._on_connected = callback

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register the sink for the connection being lost.

        Fires on an involuntary loss — a peer drop or a ping-watchdog timeout —
        not on a commanded :meth:`close`.
        """
        self._on_disconnect = callback

    def on_closed(self, callback: Callable[[], None]) -> None:
        """Register the observer for a commanded :meth:`close`.

        The mirror of :meth:`on_disconnect`: it fires only when the connection
        is torn down on command (session teardown), never on an involuntary
        loss. It lets the owner that initiated the close drop its own bookkeeping
        for this connection without the loss being reported back as if it came
        from the wire.
        """
        self._on_closed = callback

    def on_stats(self, callback: Callable[[PeerStats], None]) -> None:
        """Register an optional observer for each stats sample."""
        self._on_stats = callback

    def send_message(self, payload: bytes | str) -> None:
        """Send an already-encoded frame (text or binary) to this client."""
        self._peer.send_message(payload)

    def send_control(self, payload: bytes | str) -> None:
        """Send an already-encoded control frame (text or binary) to this client."""
        self._peer.send_control(payload)

    def send_media(self, chunk: MediaChunk) -> None:
        """Submit a media chunk to this connection's pacer.

        The pacer splits the chunk into single frames and drains them to the peer
        at the chunk's declared rate, so the model's bursty, unpaced emission
        reaches the wire as a steady stream. A chunk that asks for backpressure
        (``chunk.wait``) makes this call wait for queue room, throttling the
        producer to the playout rate; otherwise overflow is dropped.
        """
        self._pacer.submit(chunk)

    def flush_media(self) -> None:
        """Drop this connection's queued media and cut playout to black."""
        self._pacer.flush()

    def set_media_rate(self, fps: float) -> None:
        """Re-pace this connection's queued media at *fps* immediately."""
        self._pacer.set_rate(fps)

    def set_media_depth(self, depth: int) -> None:
        """Bound how many frames may queue between the model and this wire."""
        self._pacer.set_depth(depth)

    def resume_track(self, name: str) -> None:
        """Resume the named outbound track (publisher arbitration)."""
        self._peer.resume_track(name)

    def pause_track(self, name: str) -> None:
        """Pause the named outbound track (publisher arbitration)."""
        self._peer.pause_track(name)

    async def add_ice(self, candidate: IceCandidate) -> None:
        """Add a trickle-ICE candidate, valid before and after the wire connects."""
        await self._peer.add_ice(candidate)

    async def close(self) -> None:
        """Tear the connection down without reporting a loss upward.

        A commanded close — session teardown — is silent toward the sink: the
        watchdog and stats sampling stop and the peer is closed, but the
        disconnect callback does not fire, because the runner initiated this and
        is not waiting to hear its own command back. The ``on_closed`` observer
        does fire, once, so the owner that built the connection can forget it.
        """
        if not self._alive:
            return
        self._alive = False
        self._cancel_tasks()
        await self._close_peer()
        if self._on_closed is not None:
            self._on_closed()

    def _handle_message(
        self, payload: bytes | str, version: ProtocolVersion, channel: Channel
    ) -> None:
        if self._on_message is not None:
            self._on_message(payload, version, channel)

    def _handle_media(self, track: str, frame: InputFrame) -> None:
        if self._on_media is not None:
            self._on_media(track, frame)

    def _handle_ping(self) -> None:
        self._last_ping = time.monotonic()
        if self._on_ping is not None:
            self._on_ping()

    def _handle_connected(self) -> None:
        if not self._alive:
            return
        self._pacer.start()
        self._start_watchdog()
        self._stats_task = asyncio.create_task(self._stats_loop())
        if self._on_connected is not None:
            self._on_connected()

    def _report_loss(self) -> None:
        """Mark the connection lost, stop its work, and report it once.

        Idempotent and silent on a second call. A peer reports disconnect only
        after releasing its own wire, so this does not close the peer; the
        watchdog path, which declares loss while the peer is still live, closes
        the peer itself before calling this.
        """
        if not self._alive:
            return
        self._alive = False
        self._cancel_tasks()
        if self._on_disconnect is not None:
            self._on_disconnect()

    def _start_watchdog(self) -> None:
        if self._ping_timeout <= 0:
            return
        self._last_ping = time.monotonic()
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        """Declare the connection lost when no client ping arrives in time.

        The peer is still live when the watchdog fires, so it is closed here
        before the loss is reported — nothing else tears it down on this path.
        """
        try:
            while self._alive:
                await asyncio.sleep(self._WATCHDOG_POLL_SECONDS)
                last = self._last_ping
                if last is None:
                    return
                if time.monotonic() - last > self._ping_timeout:
                    await self._close_peer()
                    self._report_loss()
                    return
        except asyncio.CancelledError:
            return

    async def _stats_loop(self) -> None:
        """Sample peer stats on a fixed cadence while connected.

        A single failed sample is logged and skipped rather than ending the
        sampler: stats are best-effort, so one transient error must not stop
        sampling for the life of the connection.
        """
        try:
            while self._alive:
                await asyncio.sleep(self._STATS_INTERVAL_SECONDS)
                try:
                    stats = await self._peer.stats()
                except Exception:
                    logger.debug("WebRTC stats sample failed", exc_info=True)
                    continue
                # The pacer is the one place outbound media is discarded that
                # the peer cannot see, so the connection folds its count in.
                stats = replace(
                    stats,
                    media=replace(stats.media, dropped_frames=self._pacer.dropped_frames),
                )
                self._report_media_health(stats.media)
                self._latest_stats = stats
                if self._on_stats is not None:
                    self._on_stats(stats)
        except asyncio.CancelledError:
            return

    def _report_media_health(self, media: OutboundMediaHealth) -> None:
        """Log what the last window cost this connection's outbound media.

        The counters are cumulative and nothing else reads them, so a window
        that manufactured or discarded nothing says nothing. One that did says
        it once, in the units it happened in, which is the whole reason they
        are counted: none of it is visible in the sound until it is bad enough
        to hear, and none of it appears in transport statistics at all.
        """
        previous, self._reported_media = self._reported_media, media
        silence = media.silence_frames - previous.silence_frames
        samples = media.dropped_samples - previous.dropped_samples
        bundles = media.dropped_bundles - previous.dropped_bundles
        frames = media.dropped_frames - previous.dropped_frames
        if not (silence or samples or bundles or frames):
            return
        logger.info(
            "outbound media over the last %.0fs: %d ms of silence inserted, "
            "%d ms of audio discarded to cap the buffer, %d bundles and %d frames dropped",
            self._STATS_INTERVAL_SECONDS,
            silence * 10,
            round(samples / 48),
            bundles,
            frames,
        )

    async def _close_peer(self) -> None:
        """Close the media peer, logging but not raising on failure."""
        try:
            await self._peer.close()
        except Exception:
            logger.exception("error closing WebRTC peer")

    def _cancel_tasks(self) -> None:
        self._pacer.stop()
        for task in (self._watchdog_task, self._stats_task):
            if task is not None:
                task.cancel()
        self._watchdog_task = None
        self._stats_task = None
