"""libwebrtc WebRTC peer.

The WebRTC peer backed by ``reactor_webrtc`` (a PyO3 wrapper around libwebrtc). The peer delegates
encoding, decoding, RTP/RTCP, ICE, and DTLS-SRTP to libwebrtc and works only at
the media boundary:
it feeds outbound video to a track as BGRA and outbound audio to that track's
own source as int16 PCM, and surfaces inbound media and data-channel frames back
through the callbacks the connection registers on it.

The work at that boundary is timing, not codecs. libwebrtc timestamps each video
frame from the moment it is pushed and derives the audio timestamp from the count
of samples it has been handed, so both clocks are the peer's to drive: its job is
to feed the model's paced frames at real time and keep the two streams aligned.
The section below is how it does that without letting audio drift from video or
garble.

Per-peer audio isolation
------------------------
Each outbound audio track is created with
``PeerConnectionFactory.create_audio_track_with_options(source=LocalPush)``,
which backs the track with a ``LocalAudioSource`` — a custom ``AudioSourceInterface`` that
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
Outbound audio carries no timestamp of its own. A track's RTP timestamp is a
sample counter, advanced by however many samples each ``push_pcm`` hands over,
so wall-clock time reaches the wire only as samples. A 10 ms tick that pushes
nothing is 10 ms the stream never accounts for — and since the packets either
side of it stay contiguous in both sequence number and timestamp, the client
cannot see that a gap happened. It sees the whole stream arriving late and
irregularly, and grows its jitter buffer to absorb what looks like network
burstiness.

The feeder therefore pushes exactly one 10 ms frame (480 samples at 48 kHz) per
10 ms tick, for the whole life of the track, without exception: the model's
samples when the buffer holds a frame, and silence when it does not. Silence is
how a gap is expressed, and keeping the pushed-sample count locked to the wall
clock is what keeps the stream on real time.

Simply feeding it less is not an option the wire leaves open. The sender reports
libwebrtc emits alongside the stream estimate their RTP timestamp as the last
frame's plus the time since it, so a feeder that stops leaves the reports and the
packets disagreeing by exactly the time it stopped for; the client dates the
audio that follows from the reports, finds it older than it is, and plays it
ahead of the video. An idle stretch therefore changes only who is blamed for the
silence, never whether it is sent: past a short grace it stops counting as
under-production and keeps going.

An outbound track is negotiated as sending and then paused, so nothing goes out
until a client asks for it: subscription is the client's to declare, and a wire
that sends before anyone says they are watching spends the model's output and
the client's bandwidth on nothing. Answering ``inactive`` would say as much in
the session itself, which a joining client is not expecting to read — so the
answer describes a normal sending track and the pause lands behind it.

Pausing takes that section out of the session, which stops the stream at the
engine and is what lets ``ChannelSend`` re-anchor its sample counter when the
section comes back, so a gap reaches the client as the gap it was. Pushes
meanwhile are dropped before they reach the counter, so the feeder needs no
case for it — see :meth:`WebRTCPeer.pause_track`.

A short scheduling stall is repaid in frames, for the same reason silence
covers a gap: the time has to reach the wire somehow. The buffer between the
drain thread and the feeder stays shallow, so the audio
held ahead of the wire is a small bounded offset rather than a growing lag;
trimming it discards samples, which is reported through
:class:`~reactor_runtime.transport.webrtc.stats.OutboundMediaHealth` rather than
left to be inferred from the sound.

A model may send several audio tracks, and each is a separate stream on the
wire — its own libwebrtc track, buffer, capture anchor, pause state and silence
count, all keyed by track name, exactly as video already was. One feeder thread
serves them all on one tick, so they share a cadence instead of drifting apart
on N clocks; it picks them out of the outbound tracks by kind rather than
holding a second reference that could go stale against the first.

Running on real time is what keeps each stream honest. Tying the two to each
other takes a shared capture timestamp, which reaches the wire for audio only
through the ``abs-capture-time`` header extension — and libwebrtc leaves that
unoffered, so neither peer here negotiates it. Neither track is stamped: with
audio unable to carry one, stamping video alone would move one stream's clock
and leave the other's where it was, which pairs worse than dating both by when
they reached the encoder.

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
  second thread feeds that audio to each of the peer's audio tracks in steady
  10 ms frames.
* libwebrtc's own signaling and network threads fire the observer callbacks;
  each one marshals its work onto the asyncio loop before touching peer state, so
  the callbacks the connection registered only ever run on that one loop.
"""

from __future__ import annotations

import asyncio
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
    bump_session_version,
    deduplicate_bundle_pts,
    embed_ice_candidates,
    set_media_direction,
)
from reactor_runtime.transport.webrtc.signaling import IceCandidate, SdpAnswer, SdpOffer, TrackMap
from reactor_runtime.transport.webrtc.stats import OutboundMediaHealth, PeerStats, TrackStat

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
# What one silent 10 ms frame looks like on the wire.
_AUDIO_SILENT_FRAME = np.zeros(_AUDIO_FRAME_SAMPLES, dtype=np.int16).tobytes()
# Stands in for a track that has not buffered anything yet, so a first push
# reads the same as any other.
_EMPTY_AUDIO: npt.NDArray[np.int16] = np.array([], dtype=np.int16)
# How much of a scheduling stall the feeder repays in frames before resyncing.
# Repaying keeps the sample count on the wall clock; the cap keeps a long stall
# (host suspend) from bursting seconds of audio onto the wire at once.
_AUDIO_MAX_CATCHUP_FRAMES = 5
# Share of a stats window that may be runtime-inserted silence before the
# under-production is worth a warning.
_AUDIO_SILENCE_WARN_RATIO = 0.05
# How long the feeder keeps covering for a track that has stopped delivering.
# Silence fills a gap in a stream that is running; past this the track is not
# stalled, it is idle — the model has nothing to say, or the client has
# unpublished — and there is no gap left to describe.
#
# Sized from the capture that prompted this work: its stalls ran 1.5 to 2 s, with a
# median of 1.54 s, so anything shorter leaves the tail of a real stall
# uncovered and hands the client back the gap it cannot see. Three seconds
# clears them with margin and still bounds what an idle track costs.
_AUDIO_GRACE_TICKS = 300  # 3 s

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


# Derived from the binding itself rather than hand-listed, so a codec
# reactor_webrtc adds later is recognized here with no code change — the
# name reactor-webrtc's own capability matching expects is one `.upper()`
# away from each variant's Rust name for every current codec (Vp8 -> "VP8",
# Av1 -> "AV1", ...). isinstance guards against a future non-variant
# attribute (a method PyO3 adds to the class) being mistaken for one.
_VIDEO_CODEC_BY_NAME: dict[str, rw.VideoCodec] = {
    name.upper(): value
    for name in dir(rw.VideoCodec)
    if isinstance(value := getattr(rw.VideoCodec, name), rw.VideoCodec)
}


def _video_codec_preferences(config: WebRtcConfig) -> list[rw.VideoCodec]:
    """Map the configured video codec preference order to the binding's enum.

    A name ``reactor_webrtc`` does not recognize is skipped rather than
    raising, matching ``Transceiver.set_codec_preferences``'s own behaviour
    for a codec this build did not compile in (e.g. hardware H264 on a
    software-only build): the caller's fallback order still applies to
    whatever the endpoint actually supports.
    """
    return [
        _VIDEO_CODEC_BY_NAME[entry["codec"]]
        for entry in config.supported_video_codecs
        if entry["codec"] in _VIDEO_CODEC_BY_NAME
    ]


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


async def _apply_sender_bitrate(transceiver: rw.Transceiver, config: WebRtcConfig) -> None:
    """Apply the configured per-sender bitrate bounds to one sendonly video transceiver.

    A different limit from ``_apply_bitrate_limits``: that one bounds the whole
    connection's congestion-control estimate, this one bounds what a single
    track's encoder may spend of it. They are conjunctive, so a stream runs fast
    only when both allow it — and this is the one that has to be raised, because
    with nothing set libwebrtc derives a sender's ceiling from the frame size
    alone and it is 2500 kbps for anything above 960x540.

    **Video only**, for two reasons that agree. The ceiling this lifts is
    ``GetMaxDefaultVideoBitrateKbps``, so an audio sender has nothing to lift —
    Opus is bounded by its own codec parameters, and a 10 Mbps ceiling on a
    64 kbps stream says nothing. And an audio sender has no encodings to write
    until the answer is applied, where a video one has them as soon as the
    transceiver exists, so calling this here would fail on every audio track.

    ``0`` or less in the config leaves that bound at the libwebrtc default, which
    is how the rest of this config spells an absent limit; the binding takes
    ``None`` for it. Applied per negotiation, since every offer rebuilds the peer
    and its transceivers come up on library defaults.
    """
    await transceiver.set_send_bitrate(
        min_bps=config.sender_min_kbps * 1000 if config.sender_min_kbps > 0 else None,
        max_bps=config.sender_max_kbps * 1000 if config.sender_max_kbps > 0 else None,
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
        # Outbound audio, keyed by track name. A model may send several audio
        # tracks and each is its own stream on the wire: its own buffer, its
        # own capture anchor, its own pause state. The tracks themselves live
        # in _out_tracks with the video ones; holding a second reference to
        # them here is how the two views drift apart.
        self._audio_bufs: dict[str, npt.NDArray[np.int16]] = {}
        self._audio_lock = threading.Lock()
        self._audio_thread: threading.Thread | None = None

        # Outbound media the peer manufactured or discarded. Each counter is
        # written by one thread and read by the stats sampler, which only ever
        # needs a recent value, so plain ints are enough.
        self._silence_frames: dict[str, int] = {}
        # Consecutive ticks a track has gone without a frame of the model's
        # audio. A track that has never delivered starts idle, so nothing is
        # sent or counted until it does.
        self._idle_ticks: dict[str, int] = {}
        self._dropped_samples = 0
        self._dropped_bundles = 0
        # (monotonic time, per-track silence counts) of the previous stats
        # sample, the baseline the under-production warning is measured
        # against. Per track, so one starved stream is not averaged away by
        # the others keeping up.
        self._last_silence_sample: tuple[float, dict[str, int]] | None = None

        self._paused_tracks: set[str] = set()
        # The descriptions negotiation settled on, never mutated. Every pause
        # rewrites them from scratch against the current pause set rather than
        # editing the last rewrite: an edit only carries the change it makes, so
        # resuming one track would hand back a description that had forgotten
        # every other track was paused.
        self._offer_sdp = ""
        self._answer_sdp = ""
        # Which tracks the descriptions libwebrtc holds are paused for. Starts
        # unset so the first change always applies, then tracks what was
        # actually accepted so a request that changes nothing does nothing.
        self._applied_pauses: frozenset[str] | None = None
        # How many local renegotiations have gone out. Each description has to
        # carry a session version past the last one, or it describes no change.
        self._renegotiations = 0
        # Serialises those re-applications: a pause and a resume racing would
        # leave libwebrtc holding whichever description landed last.
        self._sdp_lock = asyncio.Lock()
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

        self._offer_sdp = deduplicate_bundle_pts(sdp_offer)
        offer = rw.SessionDescription("offer", self._offer_sdp)
        await pc.set_remote_description(offer)
        self._raise_if_stopped()

        await self._attach_out_tracks(pc, factory)

        answer = await pc.create_answer()
        self._answer_sdp = answer.sdp
        await pc.set_local_description(answer)
        # The answer describes every outbound track as sending, which is the
        # session a joining client expects to read.
        self._applied_pauses = frozenset()
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
        # Nothing goes out until a client asks for it. The answer already
        # describes these tracks as sending, so this is a pause like any other
        # rather than a special case in the negotiation: it takes each section
        # out of the session behind the answer, and the client's `resume_track`
        # puts back the ones it wants.
        for info in self._track_by_mid.values():
            if info.direction is TrackDirection.OUT:
                self.pause_track(info.name)
        return embed_ice_candidates(answer.sdp, candidates)

    async def _attach_out_tracks(
        self,
        pc: rw.PeerConnection,
        factory: rw.PeerConnectionFactory,
    ) -> None:
        """Bind an outbound track to each transceiver the model sends on.

        Also applies the configured video codec preference order to every video
        transceiver.

        After ``set_remote_description`` libwebrtc has a transceiver per m-section
        carrying the offer's mid. A mid that maps to an ``OUT`` track (the client
        offered it recvonly) gets a fresh sender track and a send-only direction.
        Every video transceiver — sending or receiving — gets the codec preference
        order so both directions of a bundled connection negotiate the same
        preferred codec, not just the ones this side sends on.

        ``set_local_description`` makes each video transceiver's own sender
        actually encode with whichever preferred codec was negotiated, once
        negotiation completes — ``set_codec_preferences`` here only shapes what
        gets negotiated.

        ``transceivers()``, ``set_direction()``, ``set_track()``, and
        ``set_codec_preferences()`` are all natively awaitable. Must run before
        ``create_answer()``: a codec preference only takes effect on the next
        SDP generated for that transceiver.
        """
        transceivers = await pc.transceivers()
        codec_preferences = _video_codec_preferences(self._config)
        for transceiver in transceivers:
            mid = transceiver.mid()
            info = self._track_by_mid.get(mid) if mid is not None else None
            if info is not None and info.direction is TrackDirection.OUT:
                if info.kind is TrackKind.VIDEO:
                    track = factory.create_video_track(info.name)
                else:
                    # LocalPush, not the factory ADM: each outbound audio track
                    # gets its own source, so one peer's audio cannot reach
                    # another's encoder. See the module docstring.
                    track = factory.create_audio_track_with_options(
                        info.name, source=rw.AudioTrackSource.LocalPush
                    )
                await transceiver.set_track(track)
                await transceiver.set_direction(rw.TransceiverDirection.SendOnly)
                if info.kind is TrackKind.VIDEO:
                    await _apply_sender_bitrate(transceiver, self._config)
                self._out_tracks[info.name] = track
            if codec_preferences and transceiver.kind() == rw.MediaKind.Video:
                await transceiver.set_codec_preferences(codec_preferences)

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
        """Wire an inbound track's frames to the media callback, by track name.

        Fires on a libwebrtc thread while the remote description is being applied.

        A frame's metadata arrives on ``VideoFrame.metadata`` whenever the sender
        attached a trailer and both peers negotiated support for it: the binding
        advertises the capability in the SDP and reads the trailer off the wire on
        its own.
        """
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

    def _make_video_sink(
        self, name: str
    ) -> Callable[[bytes, int, int, rw.FrameMetadata | None], None]:
        """Build the callback that turns one decoded video frame into an InputFrame.

        The fourth argument is the trailer the sender attached, when it was there
        to read and the receiver transform is in place to strip it. It carries the
        sender's own bytes and, beside them, the capture time the sender declared
        for the frame, which reaches the model as that frame's capture time.
        """

        def sink(
            bgra: bytes, width: int, height: int, meta: rw.FrameMetadata | None = None
        ) -> None:
            if self._stop_event.is_set():
                return
            metadata = bytes(meta.user_data) if meta is not None else b""
            # Zero is the trailer's "unset", and it is not a clock reading: a frame
            # captured at the epoch is not what a sender means by it.
            capture_us = meta.capture_time_us if meta is not None else 0
            frame = InputFrame(
                data=bgra_to_rgb(bgra, width, height),
                # An empty trailer is a frame the sender attached nothing to.
                metadata=metadata or None,
                capture_time_us=capture_us or None,
            )
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

        Video crosses the boundary immediately, carrying the frame's metadata
        into the encoded packet's trailer. Audio is appended to the shallow
        buffer the feeder drains in 10 ms frames via ``track.push_pcm()``.

        """
        for track_name, data in bundle.tracks.items():
            info = data.info
            if track_name in self._paused_tracks or info.name in self._paused_tracks:
                continue
            name = self._attached_name(track_name, info)
            if name is None:
                continue
            if info.kind is TrackKind.VIDEO:
                bgra, width, height = rgb_to_bgra(data.data)
                # A bundle reaching the wire holds one frame, so it holds one
                # metadata value: a batch's list is resolved when it is split.
                metadata = data.metadata if isinstance(data.metadata, bytes) else None
                # Deliberately unstamped: see the module's Audio/video sync note.
                self._out_tracks[name].push_video_frame(bgra, width, height, user_data=metadata)
            else:
                self._enqueue_audio(name, to_int16_mono(data.data))

    def _attached_name(self, track_name: str, info: TrackInfo) -> str | None:
        """Resolve a bundle entry to the name its outbound track is held under.

        A producer may key a bundle by either the track's own name or the name
        the model declared it under; the wire knows it by one of them.
        """
        if info.name in self._out_tracks:
            return info.name
        if track_name in self._out_tracks:
            return track_name
        return None

    def _enqueue_audio(self, name: str, samples: npt.NDArray[np.int16]) -> None:
        """Append mono samples to *name*'s outbound buffer, capping its depth.

        Trimming an overfull buffer discards the oldest samples, which the wire
        can only express by pulling everything after them earlier. The count is
        reported so sustained over-production is a number rather than a
        complaint about the sound.
        """
        if not samples.size:
            return
        with self._audio_lock:
            buf = self._audio_bufs.get(name, _EMPTY_AUDIO)
            buf = np.concatenate([buf, samples])
            overflow = buf.size - _AUDIO_BUFFER_MAX_SAMPLES
            if overflow > 0:
                buf = buf[overflow:]
                self._dropped_samples += overflow
            self._audio_bufs[name] = buf

    def _audio_feed_loop(self) -> None:
        """Feed one 10 ms frame per tick to every LocalAudioSource track.

        A track's RTP timestamp counts the samples handed to it, so the feed is
        what carries wall-clock time onto the wire: every tick pushes a frame to
        each track, the model's when that track's buffer holds one and silence
        when it does not. One thread serves them all — a 10 ms tick is far more
        time than N buffer reads need, and a shared tick keeps the tracks on one
        cadence rather than N drifting ones. A stall long enough to miss whole
        ticks is repaid in frames, up to ``_AUDIO_MAX_CATCHUP_FRAMES``, so the
        sample counts stay on the clock.
        """
        next_tick = time.monotonic() + _AUDIO_FRAME_SECONDS
        while not self._stop_event.is_set():
            self._push_audio_tick()
            next_tick += _AUDIO_FRAME_SECONDS
            now = time.monotonic()
            if now > next_tick:
                missed = min(
                    int((now - next_tick) / _AUDIO_FRAME_SECONDS) + 1, _AUDIO_MAX_CATCHUP_FRAMES
                )
                for _ in range(missed):
                    self._push_audio_tick()
                next_tick += missed * _AUDIO_FRAME_SECONDS
                if next_tick < now:
                    next_tick = now + _AUDIO_FRAME_SECONDS
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

    def _push_audio_tick(self) -> None:
        """Hand every unpaused outbound track its one frame for this tick.

        Which of them are audio is settled in :meth:`_push_audio_frame`, along
        with whether they are still there at all; this decides only who the
        client is still asking for.

        Iterates a snapshot of the names: negotiation and teardown both rewrite
        the map from other threads.
        """
        for name in list(self._out_tracks):
            self._push_audio_frame(name)

    def _push_audio_frame(self, name: str) -> None:
        """Hand one audio track exactly one 10 ms frame, the model's or silence.

        Resolves the track itself and does nothing unless *name* still names an
        audio track on this wire — a session sending only video is owed nothing
        and must not be charged for it.

        Every other tick pushes, without exception. The track's RTP timestamp is
        a sample counter that only moves when samples are handed over, while the
        sender reports libwebrtc emits alongside it estimate their timestamp as
        "the last frame's plus the time since" — so a feeder that stops leaves
        the two disagreeing by exactly the time it stopped for. The client dates
        the audio that follows from the reports, finds it older than it is, and
        plays it ahead of the video. Nothing on the wire can undo that: the
        counter advances only by sending, so time only reaches the client as
        samples.

        What varies is which samples, and whose fault the silence is. A paused
        track is fed silence rather than the model's audio — the client asked
        not to hear it, not to lose the clock. A track past
        ``_AUDIO_GRACE_TICKS`` without a frame is idle rather than stalled, so
        its silence stops counting as under-production, but it does not stop.

        """
        track = self._out_tracks.get(name)
        # The binding hands out a fresh enum per call, so match by value.
        if track is None or track.kind() != rw.MediaKind.Audio:
            return
        paused = name in self._paused_tracks
        chunk: npt.NDArray[np.int16] | None = None
        if not paused:
            with self._audio_lock:
                buf = self._audio_bufs.get(name, _EMPTY_AUDIO)
                if buf.size >= _AUDIO_FRAME_SAMPLES:
                    chunk = buf[:_AUDIO_FRAME_SAMPLES]
                    self._audio_bufs[name] = buf[_AUDIO_FRAME_SAMPLES:]
        if chunk is not None:
            self._idle_ticks[name] = 0
        elif not paused and name in self._idle_ticks:
            self._idle_ticks[name] = min(self._idle_ticks[name] + 1, _AUDIO_GRACE_TICKS)
            self._silence_frames[name] = self._silence_frames.get(name, 0) + 1
        payload = _AUDIO_SILENT_FRAME if chunk is None else chunk.tobytes()
        try:
            track.push_pcm(payload, _AUDIO_SAMPLE_RATE, 1)
        except Exception:
            logger.debug("audio feed push failed", exc_info=True)

    # =========================================================================
    # Seam: sending
    # =========================================================================

    def send_media(self, bundle: MediaBundle) -> None:
        """Enqueue a media bundle for the drain thread, dropping it when full.

        A dropped bundle takes its audio with it, so the drop is counted rather
        than left to surface as a shift between audio and video.
        """
        if self._stop_event.is_set() or not self._out_tracks:
            return
        try:
            self._frame_queue.put_nowait(bundle)
        except queue.Full:
            self._dropped_bundles += 1

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
        """Add a trickle-ICE candidate; valid before and after the wire connects.

        An empty candidate string is the end-of-candidates marker (RFC 8838);
        the native binding accepts it as a no-op.
        """
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
        """Resume the named outbound track — the client subscribing to it.

        Outbound tracks start paused, so this is how a track first reaches the
        wire as well as how it comes back.

        Audio resumes on the buffer the pause left behind, which by then holds
        less than a frame, so the feeder re-anchors to whatever arrives next
        rather than dating it from before the pause.
        """
        self._paused_tracks.discard(name)
        self._schedule_direction(name, active=True)

    def pause_track(self, name: str) -> None:
        """Pause the named outbound track (publisher arbitration).

        That track alone stops reaching the wire, and the others carry on: its
        video frames are dropped as bundles arrive, and its sender is taken out
        of the session so nothing at all goes out for it.
        """
        self._paused_tracks.add(name)
        self._schedule_direction(name, active=False)

    def _schedule_direction(self, name: str, *, active: bool) -> None:
        """Run the direction change on the loop, from whatever thread asked."""
        loop = self._loop
        if loop is None or loop.is_closed() or self._stop_event.is_set():
            return
        loop.call_soon_threadsafe(lambda: loop.create_task(self._apply_pauses()))

    async def _apply_pauses(self) -> None:
        """Put the session's directions in step with which tracks are paused.

        A paused track's section goes inactive, which stops the stream at the
        engine; starting it again is the one event that makes ``ChannelSend``
        re-anchor its sample counter from the capture time, so the gap reaches
        the client as the gap it was rather than as audio dated too early. A
        sender that merely stops being fed cannot get that, because the sender
        reports carry on estimating their timestamp forward while the packets
        do not.

        Every outbound section is written on every pass, from the descriptions
        negotiation settled on. Editing the previous rewrite instead would carry
        only the change being made, so resuming one of two paused tracks would
        hand libwebrtc a description that had forgotten the other one was
        paused, and resume both.

        Nothing is signalled: the client's view of the session is unchanged and
        it simply receives nothing on a paused track meanwhile.
        """
        pc = self._pc
        if pc is None or not self._offer_sdp:
            return
        async with self._sdp_lock:
            if self._stop_event.is_set() or self._pc is None:
                return
            out_names = {info.name for info in self._track_by_mid.values()}
            paused = frozenset(self._paused_tracks & out_names)
            # A claim on an unpaused track asks for what the session already
            # says, and renegotiating that would restart streams for nothing.
            if paused == self._applied_pauses:
                return
            self._renegotiations += 1
            offer_sdp = bump_session_version(self._offer_sdp, self._renegotiations)
            answer_sdp = bump_session_version(self._answer_sdp, self._renegotiations)
            for mid, info in self._track_by_mid.items():
                if info.direction is not TrackDirection.OUT:
                    continue
                inactive = info.name in paused
                offer_sdp = set_media_direction(
                    offer_sdp, mid, "inactive" if inactive else "recvonly"
                )
                answer_sdp = set_media_direction(
                    answer_sdp, mid, "inactive" if inactive else "sendonly"
                )
            try:
                # The same order negotiation itself uses: the remote description
                # first, then the transceivers, then the local description.
                # Rewriting the SDP alone leaves each transceiver's own
                # direction where `_attach_out_tracks` set it, and libwebrtc
                # answers from the transceiver, not from the string.
                await pc.set_remote_description(rw.SessionDescription("offer", offer_sdp))
                await self._restate_out_tracks(pc, paused)
                await pc.set_local_description(rw.SessionDescription("answer", answer_sdp))
            except Exception:
                logger.warning("could not apply track pauses %s", sorted(paused), exc_info=True)
                return
            self._applied_pauses = paused

    async def _restate_out_tracks(self, pc: rw.PeerConnection, paused: frozenset[str]) -> None:
        """Put each outbound transceiver back in the state the pauses describe.

        Both halves matter. A transceiver carries its own direction, which the
        SDP does not set — libwebrtc answers from the transceiver — so a section
        written inactive while its transceiver still says send-only is a
        description that disagrees with the engine behind it. And re-applying the
        descriptions runs the negotiation again, so a sender that comes out of it
        without its track sends nothing at all.

        The tracks themselves are reused: creating fresh ones would restart the
        streams this exists to leave alone.
        """
        for transceiver in await pc.transceivers():
            mid = transceiver.mid()
            info = self._track_by_mid.get(mid) if mid is not None else None
            if info is None or info.direction is not TrackDirection.OUT:
                continue
            track = self._out_tracks.get(info.name)
            if track is not None:
                await transceiver.set_track(track)
            await transceiver.set_direction(
                rw.TransceiverDirection.Inactive
                if info.name in paused
                else rw.TransceiverDirection.SendOnly
            )
            if info.kind is TrackKind.VIDEO:
                await _apply_sender_bitrate(transceiver, self._config)

    # =========================================================================
    # Seam: stats and teardown
    # =========================================================================

    async def stats(self) -> PeerStats:
        """Sample current transport statistics from libwebrtc.

        Sampling is also when sustained audio under-production is noticed, so a
        session that is quietly padding its stream with silence says so in the
        log rather than only in the sound.
        """
        self._warn_on_audio_underrun()
        pc = self._pc
        if self._stop_event.is_set() or pc is None:
            return PeerStats(media=self._media_health())
        try:
            report = await pc.get_stats()
        except Exception:
            logger.debug("stats sample failed", exc_info=True)
            return PeerStats(media=self._media_health())
        return self._stats_from_report(report)

    def _media_health(self) -> OutboundMediaHealth:
        """Snapshot the peer's outbound manufacture-and-discard counters.

        The silence count totals every audio track, since the figure is how much
        the runtime manufactured across this peer; which track fell short is the
        warning's job to name.
        """
        return OutboundMediaHealth(
            silence_frames=sum(self._silence_frames.values()),
            dropped_samples=self._dropped_samples,
            dropped_bundles=self._dropped_bundles,
        )

    def _warn_on_audio_underrun(self) -> None:
        """Warn for each track too much of whose window since the last sample was silence.

        The silence is the runtime's own: samples the model never produced, which
        the wire cannot distinguish from samples it did. Measuring it here — at
        the source, per track, in whole 10 ms frames — reports the shortfall
        exactly, without inferring it from a packet rate whose expected value
        depends on the encoder's packetisation, and without one starved track
        being averaged away by the others keeping up.

        A track silent for longer than ``_AUDIO_GRACE_TICKS`` is taken to have
        nothing to say rather than to be failing to say it, and is passed over:
        a model that only speaks in bursts would otherwise warn through every
        gap between them.
        """
        now = time.monotonic()
        frames = dict(self._silence_frames)
        baseline = self._last_silence_sample
        self._last_silence_sample = (now, frames)
        if baseline is None:
            return
        elapsed = now - baseline[0]
        if elapsed <= 0:
            return
        for name, count in frames.items():
            if self._idle_ticks.get(name, 0) >= _AUDIO_GRACE_TICKS:
                continue
            silence = (count - baseline[1].get(name, 0)) * _AUDIO_FRAME_SECONDS
            if silence / elapsed < _AUDIO_SILENCE_WARN_RATIO:
                continue
            logger.warning(
                "Outbound audio track %r ran below real time: %.1fs of the last %.1fs was "
                "silence inserted because the model produced no audio for it",
                name,
                silence,
                elapsed,
            )

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
                    packets_sent=int(out.packets_sent),
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

        return PeerStats(rtt_seconds=rtt, tracks=tuple(tracks), media=self._media_health())

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
