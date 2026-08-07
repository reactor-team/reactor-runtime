"""libwebrtc WebRTC peer.

The WebRTC peer backed by ``reactor_webrtc`` (a PyO3 wrapper around libwebrtc). The peer delegates
encoding, decoding, RTP/RTCP, ICE, and DTLS-SRTP to libwebrtc and works only at
the media boundary:
it feeds outbound video to a track as BGRA and outbound audio to the shared
device as int16 PCM, and surfaces inbound media and data-channel frames back
through the callbacks the connection registers on it.

The work at that boundary is timing, not codecs. libwebrtc's video track and its
audio device each play out on their own real-time clock, so the peer's job is to
feed the model's paced frames at real time and keep the two streams aligned; the
section below is how it does that without letting audio drift from video or
garble.

Per-peer audio isolation
------------------------
Each outbound audio track is created with
``PeerConnectionFactory.create_audio_track_with_local_source()``, which backs the
track with a ``LocalAudioSource`` — a custom ``AudioSourceInterface`` that
maintains the list of sinks registered by each peer connection's voice send
channel.  When ``track.push_pcm()`` is called, it delivers PCM directly to that
track's encoder via ``AudioTrackSinkInterface::OnData``, bypassing the shared
ADM entirely.  This means:

* Audio pushed to peer A's track never reaches peer B's encoder.
* Different audio content can be sent to different peer connections.
* The single process-wide ``PeerConnectionFactory`` is kept (libwebrtc requires
  at most one factory per process), while per-peer audio isolation is achieved
  at the track level.

Audio/video sync
----------------
The synthetic audio device consumes exactly one 10 ms frame (480 samples at
48 kHz) per push and plays out on its own real-time clock, so audio is fed in
480-sample frames at a steady 10 ms cadence by a dedicated thread, drawn from a
small buffer the drain thread fills. Two things keep it aligned with video and
free of the artefacts a naive feeder introduces: the buffer is shallow, so the
audio it holds ahead of the wire is a small, bounded offset rather than a
growing lag; and an empty buffer feeds nothing rather than repeating the last
frame — the device fills the gap with real-time silence itself, so a model
stall leaves audio and video advancing on the same real-time clock instead of
drifting apart.

Threading
---------
* The asyncio loop thread runs negotiation, ``add_ice``, stats, and close, and
  is shared across every connection this process holds. ``reactor_webrtc``'s
  signaling methods (``create_offer``, ``create_answer``,
  ``set_local_description``, ``set_remote_description``, ``add_ice_candidate``,
  ``get_stats``, ``set_bitrate``) are awaitable: the blocking libwebrtc round-trip runs on a
  Rust-side thread pool, never on this loop. Every other blocking call this
  peer makes into ``reactor_webrtc`` or ``threading`` — attaching outbound
  tracks to their transceivers, waiting on ICE gathering to complete — is
  dispatched to an executor, so a connection negotiating never stalls the
  loop other connections are running on.
* A frame-drain thread pushes outbound video and buffers outbound audio; a
  second thread feeds that audio to the peer's audio track in steady 10 ms frames.
* libwebrtc's own signaling and network threads fire the observer callbacks;
  each one marshals its work onto the asyncio loop before touching peer state, so
  the callbacks the connection registered only ever run on that one loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import reactor_webrtc as rw

from reactor_runtime.core import (
    ConnId,
    InputFrame,
    MediaBundle,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.protocol import Channel, ProtocolVersion, sniff
from reactor_runtime.transport.webrtc.config import WebRtcConfig
from reactor_runtime.transport.webrtc.frames import (
    bgra_to_rgb,
    inbound_audio_to_mono,
    rgb_to_bgra,
    to_int16_mono,
)
from reactor_runtime.transport.webrtc.sdp import (
    Candidate,
    deduplicate_bundle_pts,
    embed_ice_candidates,
)
from reactor_runtime.transport.webrtc.signaling import IceCandidate, SdpAnswer, SdpOffer, TrackMap
from reactor_runtime.transport.webrtc.stats import PeerStats, TrackStat

logger = logging.getLogger(__name__)

DATA_CHANNEL_LABEL = "data"
CONTROL_CHANNEL_LABEL = "control"

# Outbound frame queue depth: a bundle that arrives when the drain thread is
# behind is dropped rather than allowed to grow unbounded latency.
_FRAME_QUEUE_MAX = 10

# Outbound audio is 48 kHz mono, matching the runtime's audio frames and the
# rate the synthetic audio device plays out at. The device takes one 10 ms frame
# (480 samples) per push, so audio is fed in 480-sample frames on a 10 ms clock.
_AUDIO_SAMPLE_RATE = 48_000
_AUDIO_FRAME_SAMPLES = 480
_AUDIO_FRAME_SECONDS = 0.010
# Shallow buffer bounding how far audio may run ahead of the wire: deep enough to
# absorb one model bundle's worth of samples between frames, shallow enough that
# the residual audio-behind-video offset stays small. Oldest samples are dropped
# on overflow (sustained over-production) rather than letting latency grow.
_AUDIO_BUFFER_MAX_SAMPLES = 9_600  # 200 ms at 48 kHz

# Shared media engine: one PeerConnectionFactory per process (libwebrtc requires
# this). Audio isolation between peers is achieved at the track level via
# LocalAudioSource, not via separate factories.
_factory_lock = threading.Lock()
_factory: rw.PeerConnectionFactory | None = None


def _get_factory() -> rw.PeerConnectionFactory:
    """Return the process-wide media engine, creating it on first use."""
    global _factory
    if _factory is None:
        with _factory_lock:
            if _factory is None:
                _factory = rw.PeerConnectionFactory()
    return _factory


def _build_rtc_config(config: WebRtcConfig) -> rw.RtcConfiguration:
    """Translate the transport config's ICE servers and port range into a libwebrtc config."""
    rtc = rw.RtcConfiguration()
    rtc.ice_transport_type = str(config.transport_policy)
    if config.ice_servers:
        rtc.ice_servers = [
            rw.IceServer(
                urls=list(server.urls),
                username=server.username or "",
                password=server.credential or "",
            )
            for server in config.ice_servers
        ]
    if config.port_range is not None:
        rtc.min_port, rtc.max_port = config.port_range
    return rtc


async def _apply_bitrate_limits(pc: rw.PeerConnection, config: WebRtcConfig) -> None:
    """Apply the configured congestion-control bitrate limits to a fresh peer connection.

    ``set_bitrate`` takes bits per second in ``(min, start, max)`` order; the config
    holds kbps floor/starting-point/ceiling, so this is the one place that converts.
    """
    await pc.set_bitrate(
        config.bwe_min_kbps * 1000,
        config.bwe_initial_kbps * 1000,
        config.bwe_max_kbps * 1000,
    )


def _is_terminal_state(state: rw.PeerConnectionState) -> bool:
    """Return whether a peer-connection state means the wire is gone.

    The binding delivers a fresh enum instance to each callback rather than a
    stable singleton, so states are matched by value equality, never identity.
    """
    return (
        state == rw.PeerConnectionState.Disconnected
        or state == rw.PeerConnectionState.Failed
        or state == rw.PeerConnectionState.Closed
    )


class WebRTCPeer:
    """The libwebrtc media engine behind one WebRTC connection.

    Built by :func:`libwebrtc_peer_factory`, which negotiates the offer into an
    answer; the connection then registers its inbound callbacks and drives the
    peer for the rest of its life. Do not instantiate directly.
    """

    def __init__(self) -> None:
        # The wire codec for this connection's data channels. Seeded by the
        # factory, then latched to whatever the first inbound frame sniffs as.
        self.protocol_version: ProtocolVersion = ProtocolVersion.V0
        self._protocol_sniffed = False

        self._loop: asyncio.AbstractEventLoop | None = None

        # Set by the factory before negotiation.
        self._config = WebRtcConfig()
        self._track_map = TrackMap()
        self._track_by_mid: dict[str, TrackInfo] = {}

        # Inbound callbacks the connection registers; invoked on the asyncio loop.
        self._cb_message: Callable[[bytes | str, ProtocolVersion, Channel], None] | None = None
        self._cb_media: Callable[[str, InputFrame], None] | None = None
        self._cb_ping: Callable[[], None] | None = None
        self._cb_connected: Callable[[], None] | None = None
        self._cb_disconnect: Callable[[], None] | None = None

        # libwebrtc objects.
        self._pc: rw.PeerConnection | None = None
        self._data_channel: rw.DataChannel | None = None
        self._control_channel: rw.DataChannel | None = None

        # OUT tracks (model to client) by track name, attached before the answer.
        self._out_tracks: dict[str, rw.Track] = {}
        # IN tracks (client to model) by track name. Held for the peer's life: the
        # binding stops delivering frames once a track is dropped, so a receiver
        # that is only referenced from its own frame callback would go silent.
        self._in_tracks: dict[str, rw.Track] = {}
        # Counters that match inbound tracks to IN track metadata by arrival order.
        self._in_video_seen = 0
        self._in_audio_seen = 0

        # ICE gathering, collected off the callback threads.
        self._ice_candidates: list[Candidate] = []
        self._ice_lock = threading.Lock()
        self._ice_gathering_done = threading.Event()

        # Outbound media: the drain thread pushes video and buffers audio; the
        # audio thread feeds that buffer to this peer's LocalAudioSource track in
        # steady 10 ms frames via track.push_pcm().
        self._frame_queue: queue.Queue[MediaBundle] = queue.Queue(maxsize=_FRAME_QUEUE_MAX)
        self._frame_thread: threading.Thread | None = None
        self._audio_track: rw.Track | None = None
        self._audio_buf: npt.NDArray[np.int16] = np.array([], dtype=np.int16)
        self._audio_lock = threading.Lock()
        self._audio_thread: threading.Thread | None = None

        self._paused_tracks: set[str] = set()
        self._stop_event = threading.Event()
        self._connected = threading.Event()

        logger.info("WebRTCPeer initialized")

    # =========================================================================
    # Seam: inbound callback registrars
    # =========================================================================

    def on_message(self, callback: Callable[[bytes | str, ProtocolVersion, Channel], None]) -> None:
        """Register the sink for inbound frames on either channel."""
        self._cb_message = callback

    def on_media(self, callback: Callable[[str, InputFrame], None]) -> None:
        """Register the sink for inbound media frames, by track name."""
        self._cb_media = callback

    def on_ping(self, callback: Callable[[], None]) -> None:
        """Register the sink for client liveness pings."""
        self._cb_ping = callback

    def on_connected(self, callback: Callable[[], None]) -> None:
        """Register the sink for the wire reaching its connected state."""
        self._cb_connected = callback

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register the sink for the wire being lost."""
        self._cb_disconnect = callback

    # =========================================================================
    # Negotiation
    # =========================================================================

    async def _negotiate(self, sdp_offer: str) -> str:
        """Create the peer connection, answer the offer, and gather ICE.

        Returns the answer SDP with the runtime's gathered candidates embedded.

        Raises:
            ValueError: If the offer is empty.
            RuntimeError: If the peer is stopped mid-negotiation.
        """
        if not sdp_offer.strip():
            raise ValueError("empty SDP offer")

        loop = asyncio.get_running_loop()
        self._loop = loop
        factory = _get_factory()

        observer = rw.PeerConnectionObserver()
        observer.on_connection_state_change = self._on_connection_state_change
        observer.on_ice_gathering_change = self._on_ice_gathering_change
        observer.on_ice_candidate = self._on_ice_candidate
        observer.on_track = self._on_track
        observer.on_data_channel = self._on_data_channel

        pc = factory.create_peer_connection(_build_rtc_config(self._config), observer)
        self._pc = pc
        await _apply_bitrate_limits(pc, self._config)

        offer = rw.SessionDescription("offer", deduplicate_bundle_pts(sdp_offer))
        await pc.set_remote_description(offer)
        self._raise_if_stopped()

        await self._attach_out_tracks(loop, pc, factory)

        answer = await pc.create_answer()
        await pc.set_local_description(answer)
        self._raise_if_stopped()

        timeout_s = self._config.ice_gathering_timeout_ms / 1000.0
        gathered = await loop.run_in_executor(
            None, lambda: self._ice_gathering_done.wait(timeout=timeout_s)
        )
        if not gathered:
            logger.warning("ICE gathering timed out; answering with partial candidates")

        with self._ice_lock:
            candidates = list(self._ice_candidates)

        self._start_pumps()
        return embed_ice_candidates(answer.sdp, candidates)

    async def _attach_out_tracks(
        self,
        loop: asyncio.AbstractEventLoop,
        pc: rw.PeerConnection,
        factory: rw.PeerConnectionFactory,
    ) -> None:
        """Bind an outbound track to each transceiver the model sends on.

        After ``set_remote_description`` libwebrtc has a transceiver per m-section
        carrying the offer's mid. A mid that maps to an ``OUT`` track (the client
        offered it recvonly) gets a fresh sender track and a send-only direction.

        ``transceivers()``/``set_track()``/``set_direction()`` are synchronous
        libwebrtc calls — dispatched to an executor so they never stall the
        shared event loop other peers' connections are also running on.
        """
        transceivers = await loop.run_in_executor(None, pc.transceivers)
        for transceiver in transceivers:
            mid = transceiver.mid()
            info = self._track_by_mid.get(mid) if mid is not None else None
            if info is None or info.direction is not TrackDirection.OUT:
                continue
            if info.kind is TrackKind.VIDEO:
                track = factory.create_video_track(info.name)
            else:
                track = factory.create_audio_track_with_local_source(info.name)
                self._audio_track = track
            await loop.run_in_executor(None, transceiver.set_track, track)
            await loop.run_in_executor(
                None, transceiver.set_direction, rw.TransceiverDirection.SendOnly
            )
            self._out_tracks[info.name] = track

    def _raise_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("peer stopped during negotiation")

    # =========================================================================
    # Observer callbacks (libwebrtc threads)
    # =========================================================================

    def _on_connection_state_change(self, state: rw.PeerConnectionState) -> None:
        logger.debug("connection state changed to %r", state)
        if state == rw.PeerConnectionState.Connected:
            if not self._connected.is_set():
                self._connected.set()
                self._fire(self._cb_connected)
        elif _is_terminal_state(state):
            loop = self._loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(self._report_loss)

    def _on_ice_gathering_change(self, state: rw.IceGatheringState) -> None:
        if state == rw.IceGatheringState.Complete:
            self._ice_gathering_done.set()

    def _on_ice_candidate(self, candidate: rw.IceCandidate) -> None:
        with self._ice_lock:
            self._ice_candidates.append((candidate.sdp_mline_index, candidate.candidate))

    def _on_track(self, kind: rw.MediaKind, track: rw.Track) -> None:
        """Wire an inbound track's frames to the media callback, by track name."""
        in_tracks = self._track_map.by_direction(TrackDirection.IN)
        if kind == rw.MediaKind.Video:
            name = self._inbound_name(in_tracks, TrackKind.VIDEO, self._in_video_seen)
            self._in_video_seen += 1
            self._in_tracks[name] = track
            track.on_video_frame(self._make_video_sink(name))
        elif kind == rw.MediaKind.Audio:
            name = self._inbound_name(in_tracks, TrackKind.AUDIO, self._in_audio_seen)
            self._in_audio_seen += 1
            self._in_tracks[name] = track
            track.on_audio_frame(self._make_audio_sink(name))

    @staticmethod
    def _inbound_name(in_tracks: list[TrackInfo], kind: TrackKind, seen: int) -> str:
        matching = [t for t in in_tracks if t.kind is kind]
        if seen < len(matching):
            return matching[seen].name
        return f"{kind.value}-{seen + 1}"

    def _make_video_sink(self, name: str) -> Callable[[bytes, int, int], None]:
        def sink(bgra: bytes, width: int, height: int) -> None:
            if self._stop_event.is_set():
                return
            frame = InputFrame(data=bgra_to_rgb(bgra, width, height))
            self._fire(self._cb_media, name, frame)

        return sink

    def _make_audio_sink(self, name: str) -> Callable[[bytes, int, int, int], None]:
        def sink(pcm: bytes, sample_rate: int, channels: int, frames: int) -> None:
            if self._stop_event.is_set():
                return
            frame = InputFrame(data=inbound_audio_to_mono(pcm, channels))
            self._fire(self._cb_media, name, frame)

        return sink

    def _on_data_channel(self, channel: rw.DataChannel) -> None:
        if channel.label() == CONTROL_CHANNEL_LABEL:
            self._control_channel = channel
            channel.on_message(self._make_message_sink(Channel.CONTROL))
        else:
            self._data_channel = channel
            channel.on_message(self._make_message_sink(Channel.DATA))

    def _make_message_sink(self, channel: Channel) -> Callable[[bytes, bool], None]:
        def sink(data: bytes, binary: bool) -> None:
            if self._stop_event.is_set():
                return
            payload: bytes | str = bytes(data) if binary else data.decode("utf-8", "replace")
            if not self._protocol_sniffed:
                self.protocol_version = sniff(payload)
                self._protocol_sniffed = True
            self._fire(self._cb_message, payload, self.protocol_version, channel)
            self._fire(self._cb_ping)

        return sink

    def _report_loss(self) -> None:
        """Release the wire and report an involuntary loss, exactly once.

        Runs on the asyncio loop. Dropping the peer connection here — off the
        libwebrtc callback thread — lets its destructor close cleanly, and honours
        the seam's contract that a peer fires disconnect only after it has let go
        of its own wire, so the connection does not close it again.
        """
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._connected.clear()
        self._release_wire()
        self._fire(self._cb_disconnect)

    def _release_wire(self) -> None:
        self._pc = None
        self._data_channel = None
        self._control_channel = None
        self._out_tracks.clear()
        self._in_tracks.clear()

    def _fire(self, callback: Callable[..., None] | None, *args: Any) -> None:
        """Invoke an inbound callback on the asyncio loop, if one is registered."""
        loop = self._loop
        if loop is None or callback is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(callback, *args)

    # =========================================================================
    # Outbound media pumps (background threads)
    # =========================================================================

    def _start_pumps(self) -> None:
        self._frame_thread = threading.Thread(
            target=self._frame_drain_loop, name="libwebrtc-frame-drain", daemon=True
        )
        self._frame_thread.start()
        self._audio_thread = threading.Thread(
            target=self._audio_feed_loop, name="libwebrtc-audio-feed", daemon=True
        )
        self._audio_thread.start()

    def _frame_drain_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                bundle = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._push_bundle(bundle)
            except Exception:
                logger.debug("outbound frame push failed", exc_info=True)

    def _push_bundle(self, bundle: MediaBundle) -> None:
        """Push a bundle's video and buffer its audio for the feeder thread.

        Video crosses the boundary immediately. Audio is appended to the shallow
        buffer the feeder drains in 10 ms frames via ``track.push_pcm()``.
        """
        for track_name, data in bundle.tracks.items():
            info = data.info
            if track_name in self._paused_tracks or info.name in self._paused_tracks:
                continue
            if info.kind is TrackKind.VIDEO:
                track = self._out_tracks.get(info.name) or self._out_tracks.get(track_name)
                if track is None:
                    continue
                bgra, width, height = rgb_to_bgra(data.data)
                track.push_video_frame(bgra, width, height)
            else:
                self._enqueue_audio(to_int16_mono(data.data))

    def _enqueue_audio(self, samples: npt.NDArray[np.int16]) -> None:
        """Append mono samples to the outbound audio buffer, capping its depth."""
        if not samples.size:
            return
        with self._audio_lock:
            self._audio_buf = np.concatenate([self._audio_buf, samples])
            if self._audio_buf.size > _AUDIO_BUFFER_MAX_SAMPLES:
                self._audio_buf = self._audio_buf[-_AUDIO_BUFFER_MAX_SAMPLES:]

    def _audio_feed_loop(self) -> None:
        """Feed one 10 ms frame per tick to this peer's LocalAudioSource track.

        The device plays out at real time, so the feed is paced to real time. When
        the buffer holds a full frame it is handed over via ``track.push_pcm()``;
        when it does not, nothing is pushed — the device fills the gap with
        real-time silence, which keeps the audio clock advancing with the
        (repeated) video instead of injecting a stale frame that would slide audio
        behind.
        """
        track = self._audio_track
        next_tick = time.monotonic() + _AUDIO_FRAME_SECONDS
        while not self._stop_event.is_set():
            chunk: npt.NDArray[np.int16] | None = None
            with self._audio_lock:
                if self._audio_buf.size >= _AUDIO_FRAME_SAMPLES:
                    chunk = self._audio_buf[:_AUDIO_FRAME_SAMPLES]
                    self._audio_buf = self._audio_buf[_AUDIO_FRAME_SAMPLES:]
            if chunk is not None and track is not None:
                try:
                    track.push_pcm(chunk.tobytes(), _AUDIO_SAMPLE_RATE, 1)
                except Exception:
                    logger.debug("audio feed push failed", exc_info=True)
            now = time.monotonic()
            sleep = next_tick - now
            next_tick += _AUDIO_FRAME_SECONDS
            if next_tick < now:
                next_tick = now + _AUDIO_FRAME_SECONDS
            if sleep > 0:
                time.sleep(sleep)

    # =========================================================================
    # Seam: sending
    # =========================================================================

    def send_media(self, bundle: MediaBundle) -> None:
        """Enqueue a media bundle for the drain thread, dropping it when full."""
        if self._stop_event.is_set() or not self._out_tracks:
            return
        with contextlib.suppress(queue.Full):
            self._frame_queue.put_nowait(bundle)

    def send_message(self, payload: bytes | str) -> None:
        """Send an already-encoded frame over the data channel (text or binary)."""
        self._send_on(self._data_channel, payload)

    def send_control(self, payload: bytes | str) -> None:
        """Send an already-encoded frame over the control channel (text or binary)."""
        self._send_on(self._control_channel, payload)

    def _send_on(self, channel: rw.DataChannel | None, payload: bytes | str) -> None:
        if self._stop_event.is_set() or channel is None:
            return
        try:
            if isinstance(payload, str):
                channel.send(payload.encode("utf-8"), binary=False)
            else:
                channel.send(payload, binary=True)
        except Exception:
            logger.debug("data-channel send failed", exc_info=True)

    async def add_ice(self, candidate: IceCandidate) -> None:
        """Add a trickle-ICE candidate; valid before and after the wire connects."""
        pc = self._pc
        if self._stop_event.is_set() or pc is None:
            return
        ice = rw.IceCandidate(
            candidate=candidate.candidate,
            sdp_mid=candidate.sdp_mid,
            sdp_mline_index=candidate.sdp_mline_index,
        )
        await pc.add_ice_candidate(ice)

    # =========================================================================
    # Seam: track arbitration
    # =========================================================================

    def resume_track(self, name: str) -> None:
        """Resume the named outbound track (publisher arbitration)."""
        self._paused_tracks.discard(name)

    def pause_track(self, name: str) -> None:
        """Pause the named outbound track (publisher arbitration)."""
        self._paused_tracks.add(name)

    # =========================================================================
    # Seam: stats and teardown
    # =========================================================================

    async def stats(self) -> PeerStats:
        """Sample current transport statistics from libwebrtc."""
        pc = self._pc
        if self._stop_event.is_set() or pc is None:
            return PeerStats()
        try:
            report = await pc.get_stats()
        except Exception:
            logger.debug("stats sample failed", exc_info=True)
            return PeerStats()
        return self._stats_from_report(report)

    def _stats_from_report(self, report: rw.StatsReport) -> PeerStats:
        rtt: float | None = None
        for pair in report.candidate_pairs:
            if (
                pair.state == rw.IceCandidatePairState.Succeeded
                and pair.current_round_trip_time_s > 0.0
            ):
                rtt = pair.current_round_trip_time_s
                break

        tracks: list[TrackStat] = []
        out_infos = self._track_map.by_direction(TrackDirection.OUT)
        for i, out in enumerate(report.outbound_rtp):
            if i >= len(out_infos):
                break
            tracks.append(
                TrackStat(
                    name=out_infos[i].name,
                    direction=TrackDirection.OUT,
                    bitrate_bps=int(out.target_bitrate_bps) if out.target_bitrate_bps else None,
                )
            )

        in_infos = self._track_map.by_direction(TrackDirection.IN)
        for i, inbound in enumerate(report.inbound_rtp):
            if i >= len(in_infos):
                break
            tracks.append(
                TrackStat(
                    name=in_infos[i].name,
                    direction=TrackDirection.IN,
                    packet_loss=max(0, inbound.packets_lost),
                    jitter=inbound.jitter_s,
                )
            )

        return PeerStats(rtt_seconds=rtt, tracks=tuple(tracks))

    async def close(self) -> None:
        """Tear the peer connection down, joining the pump threads off the loop."""
        await asyncio.to_thread(self._teardown)

    def _teardown(self) -> None:
        if self._stop_event.is_set() and self._frame_thread is None:
            return
        self._stop_event.set()
        self._release_wire()
        for thread in (self._frame_thread, self._audio_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
        self._frame_thread = None
        self._audio_thread = None


async def libwebrtc_peer_factory(
    conn_id: ConnId,
    offer: SdpOffer,
    tracks: TrackMap,
    config: WebRtcConfig,
    version: ProtocolVersion,
) -> tuple[WebRTCPeer, SdpAnswer]:
    """Negotiate *offer* into a :class:`WebRTCPeer` and its SDP answer.

    Threads the connection's config and the client's declared track map into the
    peer, then drives the offer/answer exchange. *version* seeds the wire codec as
    the pre-first-frame default; the first inbound data-channel frame is sniffed
    and latches the codec the peer holds for its life.
    """
    logger.debug("negotiating libwebrtc peer for connection %s", conn_id)
    peer = WebRTCPeer()
    peer.protocol_version = version
    peer._config = config
    peer._track_map = tracks
    peer._track_by_mid = {mapped.mid: mapped.info for mapped in tracks.tracks}
    answer_sdp = await peer._negotiate(offer.sdp)
    return peer, SdpAnswer(sdp=answer_sdp)


WebRtcPeerFactory = Callable[
    [ConnId, SdpOffer, TrackMap, WebRtcConfig, ProtocolVersion],
    Awaitable[tuple[WebRTCPeer, SdpAnswer]],
]
"""Build a negotiated :class:`WebRTCPeer` for *(conn id, offer, tracks, config, version)*.

Returns the peer and the SDP answer produced during the exchange. *version* is
the wire codec negotiated for the connection, which the peer holds for its life.
"""
