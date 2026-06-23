"""GStreamer WebRTC peer.

The concrete :class:`~reactor_runtime.transport.webrtc.peer.WebRtcPeer` for the
WebRTC wire, ported from the original runtime. It owns one negotiated peer
connection on a dedicated GLib thread with its own main loop: it moves encoded
messages and media, samples transport statistics, and reports inbound facts
back through the callbacks the connection registers on it.

The liveness watchdog, the stats-sampling cadence, and close orchestration live
in the connection above this peer; here we drive GStreamer and surface facts.

.. note::

    ``send_media()`` currently forwards only the first video track from the
    bundle to the GStreamer appsrc; audio tracks are dropped. Full multi-track
    routing is future work.
"""

from __future__ import annotations

import asyncio
import json
import queue
import random
import threading
import uuid
from collections.abc import Callable
from os import environ
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from reactor_runtime.core import (
    ConnId,
    InputFrame,
    MediaBundle,
    TrackDirection,
    TrackInfo,
    TrackKind,
)
from reactor_runtime.transport.webrtc.config import IceServer, IceTransportPolicy, WebRtcConfig
from reactor_runtime.transport.webrtc.gstreamer._log import get_logger
from reactor_runtime.transport.webrtc.gstreamer.errors import (
    WebRTCNoMediaError,
    WebRTCSupersededError,
)
from reactor_runtime.transport.webrtc.gstreamer.gst import GLib, Gst, GstSdp, GstWebRTC
from reactor_runtime.transport.webrtc.gstreamer.gst_helpers import (
    add_many,
    link_pads,
    make_element,
    structure_field_uint,
    try_set_property,
)
from reactor_runtime.transport.webrtc.gstreamer.ice_uris import to_stun_turn_uris
from reactor_runtime.transport.webrtc.gstreamer.quality import aggregate_qos_score
from reactor_runtime.transport.webrtc.gstreamer.receiver import AudioReceiver, VideoReceiver
from reactor_runtime.transport.webrtc.gstreamer.sdp import (
    SdpExtmap,
    add_answer_webrtc_attributes,
    add_extmaps_per_mid_to_sdp,
    detect_bundle_policy_from_sdp,
    fix_sdp_to_max_compat_if_bundle_invalid,
    get_codec_from_sdp_by_mid,
    get_mids_by_mline,
    get_rtx_payload_type_by_mid,
    negotiated_sdp_extmaps_by_mid,
    normalize_sdp_for_supported_codecs,
    strip_ice_candidates_from_sdp,
)
from reactor_runtime.transport.webrtc.gstreamer.sdp.ice import IceCandidate
from reactor_runtime.transport.webrtc.gstreamer.sender import AudioSender, VideoSender
from reactor_runtime.transport.webrtc.gstreamer.settings import (
    SUPPORTED_RTP_HEADER_EXTENSION_URIS,
    VIDEO_BWE_MIN_BITRATE_KBPS,
    VIDEO_BWE_MAX_BITRATE_KBPS,
    VIDEO_BWE_TARGET_UPDATE_RELATIVE_THRESHOLD,
    VIDEO_RTX_MAX_SIZE_PACKETS,
    VIDEO_RTX_MAX_SIZE_TIME_MS,
)
from reactor_runtime.transport.webrtc.gstreamer.signals import SignalManager
from reactor_runtime.transport.webrtc.peer import PeerStats, TrackStat
from reactor_runtime.transport.webrtc.signaling import (
    IceCandidate as TrickleCandidate,
)
from reactor_runtime.transport.webrtc.signaling import (
    SdpAnswer,
    SdpOffer,
    TrackMap,
)

logger = get_logger(__name__)

# Valid (non-privileged) UDP port bounds for ICE when no range is configured.
MIN_VALID_PORT = 1024
MAX_VALID_PORT = 65535


# Default STUN server used when no custom ICE servers are configured
# ICE Gathering
ICE_GATHERING_TIMEOUT_MS = 3000

# Frame queue drain timeout interval in milliseconds
FRAME_Q_DRAIN_TIMEOUT_MS = 10

# Quality score measurement interval
QUALITY_METER_INTERVAL_MS = 1000

# =============================================================================
# Helpers for converting transport-agnostic types to GStreamer expected ones
# =============================================================================


def _to_gst_transport_policy(
    policy: IceTransportPolicy,
) -> GstWebRTC.WebRTCICETransportPolicy:
    """Map :class:`IceTransportPolicy` to ``GstWebRTC.WebRTCICETransportPolicy``."""
    if policy == IceTransportPolicy.RELAY:
        return GstWebRTC.WebRTCICETransportPolicy.RELAY
    return GstWebRTC.WebRTCICETransportPolicy.ALL


def _gst_buffer_pts_seconds(buf: object) -> Optional[float]:
    """Return a ``Gst.Buffer``'s PTS in seconds, or ``None`` if unset.

    GStreamer encodes PTS in nanoseconds (``Gst.SECOND == 1e9``); an
    unset PTS is signalled by the sentinel ``Gst.CLOCK_TIME_NONE``
    (``2**64 - 1``).  Any unexpected buffer shape maps to ``None`` so
    the ingress path degrades gracefully on teardown / probe buffers.
    """
    pts = getattr(buf, "pts", None)
    if pts is None or pts == Gst.CLOCK_TIME_NONE:
        return None
    try:
        return float(pts) / float(Gst.SECOND)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# =============================================================================
# GStreamerTransport
# =============================================================================


class GStreamerPeer:
    """The GStreamer media engine behind one WebRTC connection.

    Conforms to :class:`~reactor_runtime.transport.webrtc.peer.WebRtcPeer`. Built
    by :func:`gstreamer_peer_factory`, which negotiates the offer into an answer;
    the connection then registers its inbound callbacks and drives the peer for
    the rest of its life.

    Manages a single WebRTC connection on a dedicated GLib thread with its own
    main loop. Cooperative shutdown via a stop event allows a connection to be
    superseded quickly.
    """

    def __init__(self, ping_timeout_seconds: float = 20.0) -> None:
        # Inbound callbacks the connection registers; invoked on the runtime loop.
        self._cb_message: Optional[Callable[[bytes | str], None]] = None
        self._cb_media: Optional[Callable[[str, InputFrame], None]] = None
        self._cb_ping: Optional[Callable[[], None]] = None
        self._cb_connected: Optional[Callable[[], None]] = None
        self._cb_disconnect: Optional[Callable[[], None]] = None

        # Configuration threaded in by the factory.
        self._config = WebRtcConfig()
        self._ping_timeout_seconds = ping_timeout_seconds

        # Track lookups built from the client's TrackMap at negotiation.
        self._track_by_mid: Dict[str, TrackInfo] = {}
        self._mid_by_name: Dict[str, str] = {}

        # GStreamer initialization
        major, minor, micro, nano = Gst.version()
        logger.info(f"GStreamer version: {major}.{minor}.{micro} (nano={nano})")

        # Thread-safe stop flag
        self._stop_event = threading.Event()
        self._stopping = False

        # asyncio bridge
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._gst_ready_fut: Optional[asyncio.Future[None]] = None
        self._answer_fut: Optional[asyncio.Future[str]] = None

        # GLib/GStreamer thread + dedicated context/loop
        self._gst_thread: Optional[threading.Thread] = None
        self._main_loop: Optional[GLib.MainLoop] = None
        self._main_ctx: Optional[GLib.MainContext] = None
        self._main_loop_ready = threading.Event()

        self._gst_sources: Set[GLib.Source] = set()
        self._gst_sources_lock = threading.Lock()

        # Pipeline objects (GLib thread only)
        self._pipeline: Optional[Gst.Pipeline] = None
        # Sender streams by name/mid
        self._senders: Dict[str, Union[VideoSender, AudioSender]] = {}
        # Receiver streams by name/mid
        self._video_receivers: Dict[str, VideoReceiver] = {}
        self._audio_receivers: Dict[str, AudioReceiver] = {}

        self._webrtc: Optional[Gst.Element] = None
        self._data_channel = None
        self._control_channel = None
        self._transceiver_by_track_name: Dict[str, Any] = {}
        self._paused_tracks: Set[str] = set()

        self._sender_tracks: List[TrackInfo] = []
        self._receiver_tracks: List[TrackInfo] = []
        # SSRC per sendonly track (mid -> ssrc), set when creating senders;
        # used in SDP answer so it matches the RTP payloader.
        self._ssrc_by_mid: Dict[str, int] = {}
        # RTX SSRC per video mid (matches rtprtxsend ssrc-map); used in SDP FID / a=ssrc.
        self._rtx_ssrc_by_mid: Dict[str, int] = {}
        # SSRC for inbound (receiver) tracks, populated in _gst_on_pad_added from
        # the negotiated RTP caps; used to correlate inbound-rtp stats by track name.
        self._inbound_ssrc_by_track_name: Dict[str, int] = {}
        # Single CNAME for all senders (RFC 3550); set when creating senders.
        self._cname: Optional[str] = None

        # ICE
        self._ice_agent: Optional[GstWebRTC.WebRTCICE] = None
        self._remote_candidates: List[IceCandidate] = []
        # Last offer SDP passed to set-remote-description (for per-mid extmap ids).
        self._remote_offer_sdp: Optional[str] = None
        self._ice_gathering_timeout_source: Optional[GLib.Source] = None
        self._transport_policy: GstWebRTC.WebRTCICETransportPolicy = (
            GstWebRTC.WebRTCICETransportPolicy.ALL
        )

        # Outgoing frame queue (thread-safe producer, GLib consumer)
        self._frame_q: "queue.Queue[MediaBundle]" = queue.Queue(maxsize=10)

        # GLib main loop (runs GStreamer event processing)
        self._signals = SignalManager()

        # Connection state
        self._connected = threading.Event()

        # Track if we were superseded (stopped during setup)
        self._was_superseded = False

        # Optional port range for ICE UDP binding
        self._port_range: Optional[Tuple[int, int]] = None

        # Answer future for async waiting
        self._answer_sdp: Optional[str] = None

        # Incoming video frame info
        self._incoming_width = 0
        self._incoming_height = 0

    # =========================================================================
    # Seam: inbound callback registrars
    # =========================================================================

    def on_message(self, callback: Callable[[bytes | str], None]) -> None:
        """Register the sink for inbound data-channel frames (text or binary)."""
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
    # Seam: stats and teardown
    # =========================================================================

    async def stats(self) -> PeerStats:
        """Sample current transport statistics."""
        rtt, tracks = await self._get_rtc_stats()
        return PeerStats(rtt_seconds=rtt, tracks=tuple(tracks))

    async def close(self) -> None:
        """Tear the peer connection down, joining the GLib thread off the loop."""
        await asyncio.to_thread(self.stop)

    # =========================================================================
    # Negotiation
    # =========================================================================

    async def _negotiate(
        self,
        sdp_offer: str,
        ice_servers: Optional[List[IceServer]],
    ) -> str:
        """Start the GLib thread, set up the pipeline, and resolve the answer.

        Args:
            sdp_offer: The SDP offer string from the client.
            ice_servers: The STUN/TURN servers to gather against, or ``None``.

        Returns:
            The SDP answer string.

        Raises:
            WebRTCSupersededError: If the peer was closed during setup.
        """
        self._loop = asyncio.get_running_loop()
        self._gst_ready_fut = self._loop.create_future()
        self._answer_fut = self._loop.create_future()

        self._gst_thread = threading.Thread(
            target=self._run_gst_thread,
            name="gst-thread",
            daemon=False,
        )
        self._gst_thread.start()

        await asyncio.wait_for(self._gst_ready_fut, timeout=5.0)

        if self._stop_event.is_set():
            raise WebRTCSupersededError("Client stopped before setup")

        self._run_on_gst_thread(self._gst_setup_pipeline, sdp_offer, ice_servers)

        return await asyncio.wait_for(self._answer_fut, timeout=10.0)

    def _run_gst_thread(self) -> None:
        logger.debug(f"_run_gst_thread() [thread:{threading.current_thread().name}]")

        # Create a dedicated MainContext for this thread (recommended)
        self._main_ctx = GLib.MainContext()
        self._main_ctx.push_thread_default()

        self._main_loop = GLib.MainLoop(context=self._main_ctx)

        # Signal readiness back to asyncio
        self._complete_future_threadsafe(self._gst_ready_fut, result=None)

        try:
            logger.debug(
                f"_run_gst_thread() starting main loop [thread:{threading.current_thread().name}]"
            )
            self._main_loop.run()
        except Exception as e:
            self._fail_future_threadsafe(self._gst_ready_fut, e)
            self._fail_future_threadsafe(self._answer_fut, e)
        finally:
            try:
                self._main_ctx.pop_thread_default()
            except Exception:
                pass

    # =========================================================================
    # Thread-Safe Future helpers
    # =========================================================================

    def _complete_future_threadsafe(self, fut: Optional[asyncio.Future], result):
        loop = self._loop
        if loop is None or fut is None:
            return

        def _do():
            if not fut.done():
                fut.set_result(result)

        loop.call_soon_threadsafe(_do)

    def _fail_future_threadsafe(self, fut: Optional[asyncio.Future], exc: Exception):
        loop = self._loop
        if loop is None or fut is None:
            return

        def _do():
            if not fut.done():
                fut.set_exception(exc)

        loop.call_soon_threadsafe(_do)

    def _run_on_gst_thread(self, fn, *args, **kwargs) -> None:
        if self._main_ctx is None:
            return

        src = GLib.idle_source_new()
        src.set_priority(GLib.PRIORITY_HIGH)
        self._track_source(src)

        def _cb(*_cb_args):
            try:
                if getattr(self, "_stopping", False):
                    return GLib.SOURCE_REMOVE

                fn(*args, **kwargs)

            except Exception as e:
                self._fail_future_threadsafe(self._answer_fut, e)
                try:
                    self._gst_request_stop()
                except Exception:
                    pass

            finally:
                self._untrack_source(src)

            return GLib.SOURCE_REMOVE  # False (one-shot)

        src.set_callback(_cb)
        src.attach(self._main_ctx)

    def _track_source(self, src: GLib.Source) -> None:
        with self._gst_sources_lock:
            self._gst_sources.add(src)

    def _untrack_source(self, src: GLib.Source) -> None:
        with self._gst_sources_lock:
            self._gst_sources.discard(src)

    # =========================================================================
    # GLib thread: Frame pushing (appsrc)
    # =========================================================================

    def _gst_arm_frame_pusher(self) -> None:
        if self._main_ctx is None:
            return

        def _drain(*_args) -> bool:
            if len(self._senders) == 0:
                return GLib.SOURCE_REMOVE

            if self._stopping:
                return GLib.SOURCE_REMOVE

            while True:
                try:
                    bundle = self._frame_q.get_nowait()
                except queue.Empty:
                    break

                for track_data in bundle.get_tracks():
                    track_name = track_data.info.name
                    if track_name in self._paused_tracks:
                        continue
                    mid = self._mid_by_name.get(track_name, track_name)
                    sender = self._senders.get(mid)
                    if sender is None:
                        continue

                    if not sender.push_buffer(track_data.data):
                        # Debug-level: this fires at video / audio frame
                        # rate (30+ Hz) when the sender's appsrc rejects a
                        # buffer (e.g. pipeline not yet PLAYING, or audio
                        # sender not fully wired up).  It's useful when
                        # debugging a stalled encoder but should not flood
                        # production logs at steady state.
                        logger.debug(f"Failed to push buffer for track {mid}")

            return GLib.SOURCE_CONTINUE

        src = GLib.timeout_source_new(FRAME_Q_DRAIN_TIMEOUT_MS)
        self._track_source(src)
        src.set_callback(_drain)
        src.attach(self._main_ctx)

    def _gst_arm_quality_meter(self) -> None:
        """GLib thread: log a per-second video quality score for every active video sender."""
        if self._main_ctx is None:
            return

        src = GLib.timeout_source_new(QUALITY_METER_INTERVAL_MS)
        self._track_source(src)

        def _tick(*_args) -> bool:
            if self._stopping:
                return GLib.SOURCE_REMOVE

            scores = [
                s
                for sender in self._senders.values()
                if isinstance(sender, VideoSender)
                for s in [sender.qos()]
                if s is not None
            ]
            score = aggregate_qos_score(scores)
            if score is not None:
                logger.info(
                    "video quality score",
                    score=score,
                    senders=len(scores),
                )

            return GLib.SOURCE_CONTINUE

        src.set_callback(_tick)
        src.attach(self._main_ctx)

    def _require_webrtc(self) -> Gst.Element:
        if self._webrtc is None:
            raise RuntimeError("webrtcbin element is not initialized")
        return self._webrtc

    def _require_data_channel(self) -> GstWebRTC.WebRTCDataChannel:
        if self._data_channel is None:
            raise RuntimeError("DataChannel is not initialized")
        return self._data_channel

    def _require_control_channel(self) -> GstWebRTC.WebRTCDataChannel:
        if self._control_channel is None:
            raise RuntimeError("ControlChannel is not initialized")
        return self._control_channel

    # =========================================================================
    # GLib thread: Pipeline setup
    # =========================================================================
    def _gst_setup_pipeline(
        self,
        sdp_offer: str,
        ice_servers: Optional[List[IceServer]],
    ) -> None:
        """Set up the GStreamer WebRTC pipeline from the offer."""
        logger.debug(
            f"_gst_setup_pipeline() [thread:{threading.current_thread().name}]"
        )

        if self._stop_event.is_set():
            raise WebRTCSupersededError("Client stopped")

        self._mids_by_mline_index = get_mids_by_mline(sdp_offer)
        mids = self._mids_by_mline_index

        if len(mids) == 0:
            raise WebRTCNoMediaError("No media tracks found in SDP offer")

        # Build track list from SDP mids and the track map only
        tracks = []
        for mid in mids:
            info = self._track_by_mid.get(mid)
            if info is not None:
                tracks.append(info)

        # If no tracks in the track map matched SDP mids, raise an error
        if len(tracks) == 0:
            raise WebRTCNoMediaError(
                f"No tracks in the track map matched SDP mids: {mids}"
            )

        self._sender_tracks = [
            track for track in tracks if track.direction == TrackDirection.OUT
        ]
        self._receiver_tracks = [
            track for track in tracks if track.direction == TrackDirection.IN
        ]

        # Create pipeline
        self._pipeline = Gst.Pipeline.new("pipeline")
        self._webrtc = make_element("webrtcbin", "webrtc")

        bundle_policy = detect_bundle_policy_from_sdp(sdp_offer)

        if bundle_policy.mode == "bundle-invalid":
            self._webrtc.set_property("bundle-policy", "max-compat")
            sdp_fixed, fixed_analysis = fix_sdp_to_max_compat_if_bundle_invalid(
                sdp_offer
            )

            logger.warning(
                f"Offer had invalid BUNDLE; trying to rewrite it to max-compat. Reason: {bundle_policy.reason}. SDP Offer: {sdp_offer}; SDP Fixed {sdp_fixed}"
            )

            sdp_offer = sdp_fixed
        else:
            self._webrtc.set_property("bundle-policy", bundle_policy.mode)

        # Normalize SDP for supported codecs
        sdp_offer = normalize_sdp_for_supported_codecs(sdp_offer)

        self._webrtc.set_property("latency", self._config.webrtcbin_latency_ms)

        # Setup ICE
        self._webrtc.set_property("ice-transport-policy", self._transport_policy)

        # No silent fallback to a public STUN service: if iceServers is empty
        # or contains no `stun:` entry, leave webrtcbin's `stun-server`
        # property at its default (None) rather than setting it. webrtcbin
        # validates the string as a `stun://host:port` URI on assignment, so
        # writing "" yields a `Stun server has no host` warning rather than
        # the intended "STUN disabled" behaviour.
        if ice_servers is not None:
            stun_servers, turn_uris = to_stun_turn_uris(ice_servers)
            if stun_servers:
                self._webrtc.set_property("stun-server", stun_servers[0])
            for turn_uri in turn_uris:
                self._webrtc.emit("add-turn-server", turn_uri)

        self._ice_agent = self._webrtc.get_property("ice-agent")
        if self._port_range is None:
            self._ice_agent.set_property("min-rtp-port", MIN_VALID_PORT)
            self._ice_agent.set_property("max-rtp-port", MAX_VALID_PORT)
        else:
            self._ice_agent.set_property("min-rtp-port", self._port_range[0])
            self._ice_agent.set_property("max-rtp-port", self._port_range[1])

        # Disable ICE-TCP and UPnP on the underlying NiceAgent to speed up
        # ICE gathering. WebRTC media uses UDP (DTLS-SRTP); TCP candidates are
        # a rare fallback that adds ~1-2s to gathering.  UPnP/NAT-PMP probes
        # add another 200-500ms and are not useful in server environments.
        try:
            nice_agent = self._ice_agent.get_property("agent")
            nice_agent.set_property("ice-tcp", self._config.ice_tcp)
            nice_agent.set_property("upnp", self._config.upnp)
        except Exception:
            logger.debug("Could not configure NiceAgent properties (ice-tcp/upnp)")

        # This signal should be used for isolated test env (no influency related to user network)
        if _use_local_ip_addr():
            self._ice_agent.emit("add-local-ip-address", "127.0.0.1")

        self._pipeline.add(self._webrtc)

        negotiated_hdr_extmaps_by_mid = negotiated_sdp_extmaps_by_mid(
            sdp_offer, SUPPORTED_RTP_HEADER_EXTENSION_URIS
        )

        # Create a sender for each OUT track, keyed by MID.
        self._cname = uuid.uuid4().hex[:16]
        for track in self._sender_tracks:
            mid = self._mid_by_name.get(track.name)
            if mid is None:
                logger.warning(
                    f"Track {track.name} not found in track map, skipping"
                )
                continue

            codec_entry = get_codec_from_sdp_by_mid(sdp_offer, mid)
            logger.info(
                f"Outgoing {track.kind.value} encoder: "
                f"mid={mid} codec={codec_entry.get('codec')} "
                f"pt={codec_entry.get('payload_type')} "
                f"params={codec_entry.get('parameters', {})}"
            )
            ssrc = random.randint(1, 0x7FFFFFFF)
            self._ssrc_by_mid[mid] = ssrc

            if track.kind == TrackKind.VIDEO:
                video_hdr_exts = list(negotiated_hdr_extmaps_by_mid.get(mid, []))
                primary_pt = codec_entry.get("payload_type")
                rtx_pt = (
                    get_rtx_payload_type_by_mid(sdp_offer, mid, int(primary_pt))
                    if primary_pt is not None
                    else None
                )

                if rtx_pt is not None:
                    rtx_ssrc = random.randint(1, 0x7FFFFFFF)
                    while rtx_ssrc == ssrc:
                        rtx_ssrc = random.randint(1, 0x7FFFFFFF)
                    self._rtx_ssrc_by_mid[mid] = rtx_ssrc
                else:
                    rtx_ssrc = None
                    rtx_pt = None

                sender = VideoSender(
                    codec_entry,
                    name=mid,
                    ssrc=ssrc,
                    rtx_ssrc=rtx_ssrc,
                    rtx_payload_type=rtx_pt,
                    rtp_header_extensions=video_hdr_exts,
                )
            else:
                sender = AudioSender(codec_entry, name=mid, ssrc=ssrc)
            self._senders[mid] = sender
            self._paused_tracks.add(track.name)

            add_many(
                self._pipeline,
                sender,
                sync_with_parent=True,
            )

        # Create transceivers in exact m-line order so that webrtcbin's
        # sequential matching maps each transceiver to the correct m-line.
        # For receiver (inbound) m-lines we use add-transceiver to create a
        # RECVONLY placeholder; for sender (outbound) m-lines we request a
        # sink pad and link the encoder.  Without this, webrtcbin matches
        # purely by media type: the first N video sender-transceivers bind
        # to the first N video m-lines regardless of SDP direction, which
        # breaks when inbound m-lines precede outbound ones.
        transceiver_idx = 0
        for mid in self._mids_by_mline_index:
            if mid is None:
                continue

            sender = self._senders.get(mid)
            track_info = self._track_by_mid.get(mid)

            if sender is not None:
                webrtc_sinkpad = self._webrtc.request_pad_simple("sink_%u")
                if not webrtc_sinkpad:
                    raise RuntimeError("Failed to request sink_%u pad from webrtcbin.")
                sender.link_src_to(webrtc_sinkpad)

                transceiver = self._webrtc.emit("get-transceiver", transceiver_idx)
                if transceiver:
                    transceiver.set_property(
                        "direction",
                        GstWebRTC.WebRTCRTPTransceiverDirection.SENDONLY,
                    )
                    if track_info is not None:
                        self._transceiver_by_track_name[track_info.name] = transceiver

                try_set_property(transceiver, "do-nack", True)
            elif track_info is not None and track_info.direction == TrackDirection.IN:
                media_type = track_info.kind.value
                caps = Gst.Caps.from_string(
                    f"application/x-rtp,media=(string){media_type}"
                )
                transceiver = self._webrtc.emit(
                    "add-transceiver",
                    GstWebRTC.WebRTCRTPTransceiverDirection.RECVONLY,
                    caps,
                )
                if transceiver is not None:
                    self._transceiver_by_track_name[track_info.name] = transceiver
            else:
                continue

            transceiver_idx += 1

        # Setup WebRTC handlers
        self._setup_webrtc_handlers()

        # Frame pusher (GLib thread)
        self._gst_arm_frame_pusher()

        # Start pipeline
        logger.debug(
            f"_gst_setup_pipeline() set state to PLAYING [thread:{threading.current_thread().name}]"
        )
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")

        if self._stop_event.is_set():
            raise WebRTCSupersededError("Client stopped")

        # Set remote description (the offer)
        logger.debug(
            f"_gst_setup_pipeline() set remote description [thread:{threading.current_thread().name}]"
        )
        self._gst_set_remote_description(sdp_offer)

    def _start_ice_gathering_timeout(self, time_ms=ICE_GATHERING_TIMEOUT_MS) -> None:
        if self._ice_gathering_timeout_source is None:
            src = GLib.timeout_source_new(time_ms)

            def _cb(*_args):
                self._gst_on_ice_gathering_timeout()
                return False

            src.set_callback(_cb)
            src.attach(self._main_ctx)

            self._ice_gathering_timeout_source = src

    def _link_to_webrtc(self, element: Gst.Element):
        webrtc = self._require_webrtc()

        # Request pad no webrtcbin: sink_%u
        sinkpad = webrtc.request_pad_simple("sink_%u")
        if not sinkpad:
            raise RuntimeError("Failed to request sink_%u pad from webrtcbin.")

        srcpad = element.get_static_pad("src")
        if not srcpad:
            raise RuntimeError("Failed to get src pad from element.")

        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link element -> webrtcbin(sink_%u).")

    def _setup_webrtc_handlers(self) -> None:
        """Setup signal handlers for webrtcbin."""
        webrtc = self._require_webrtc()
        self._signals.connect(webrtc, "on-data-channel", self._gst_on_data_channel)

        self._signals.connect(
            webrtc, "on-negotiation-needed", self._gst_on_negotiation_needed
        )
        self._signals.connect(webrtc, "on-ice-candidate", self._gst_on_ice_candidate)
        self._signals.connect(
            webrtc, "notify::ice-gathering-state", self._gst_on_ice_gathering_state
        )
        self._signals.connect(
            webrtc, "notify::ice-connection-state", self._gst_on_ice_connection_state
        )
        self._signals.connect(
            webrtc, "notify::connection-state", self._gst_on_connection_state
        )
        self._signals.connect(webrtc, "pad-added", self._gst_on_pad_added)
        self._signals.connect(
            webrtc, "request-aux-sender", self._gst_on_request_aux_sender
        )

    def _setup_data_channel_handlers(self) -> None:
        """Setup handlers for the data channel."""
        data_channel = self._require_data_channel()

        self._signals.connect(data_channel, "on-open", self._gst_on_data_channel_open)
        self._signals.connect(data_channel, "on-close", self._gst_on_data_channel_close)
        self._signals.connect(
            data_channel, "on-message-string", self._gst_on_data_channel_message
        )

    def _setup_control_channel_handlers(self) -> None:
        """Setup handlers for the control data channel."""
        control_channel = self._require_control_channel()

        self._signals.connect(
            control_channel, "on-open", self._gst_on_data_channel_open
        )
        self._signals.connect(
            control_channel, "on-close", self._gst_on_data_channel_close
        )
        self._signals.connect(
            self._control_channel,
            "on-message-string",
            self._gst_on_control_channel_message,
        )

    # =========================================================================
    # WebRTC Signal Handlers
    # =========================================================================

    def _gst_on_negotiation_needed(self, element) -> None:
        """Handle negotiation-needed signal."""
        logger.debug("Negotiation needed")

    def _gst_on_ice_candidate(self, element, mline_index: int, candidate: str) -> None:
        """Handle ICE candidate -- resolve the SDP answer on the first non-host candidate.

        Host candidates use container-internal IPs (e.g. 172.17.0.2) that are
        unreachable from outside.  We wait for a srflx or relay candidate which
        guarantees a routable address is included in the SDP answer.
        """
        logger.debug(f"ICE candidate gathered: {candidate[:80]}...")
        if " typ host" not in candidate:
            logger.info(
                f"Resolving SDP answer on first routable ICE candidate: {candidate[:80]}..."
            )
            self._gst_resolve_answer(reason="first-routable-candidate")

    def _gst_on_ice_gathering_state(self, element, pspec) -> None:
        """Handle ICE gathering state changes."""
        if self._stopping:
            return

        state = element.get_property("ice-gathering-state")
        logger.info(f"ICE gathering state: {state}")

        if state == GstWebRTC.WebRTCICEGatheringState.COMPLETE:
            self._gst_resolve_answer(reason="ice-completed")

    def _gst_on_connection_state(self, element, pspec) -> None:
        """Handle connection state changes."""
        if self._stopping:
            return

        state = element.get_property("connection-state")
        logger.info(f"Connection state: {state}")

        if state == GstWebRTC.WebRTCPeerConnectionState.CONNECTED:
            if not self._connected.is_set():
                self._connected.set()
                self._fire(self._cb_connected)
                self._gst_arm_quality_meter()
        elif state in (
            GstWebRTC.WebRTCPeerConnectionState.FAILED,
            GstWebRTC.WebRTCPeerConnectionState.CLOSED,
            GstWebRTC.WebRTCPeerConnectionState.DISCONNECTED,
        ):
            self._connected.clear()
            # Release the wire before reporting loss: the connection's disconnect
            # path does not close the peer, trusting it has already let go.
            self._run_on_gst_thread(self._gst_request_stop)
            self._fire(self._cb_disconnect)

    def _gst_on_ice_connection_state(self, element, pspec) -> None:
        """Handle ICE connection state changes."""
        if self._stopping:
            return

        state = element.get_property("ice-connection-state")
        logger.debug(
            f"_gst_on_ice_connection_state() ICE connection state changed [state:{state}]"
        )

    def _gst_on_ice_gathering_timeout(self) -> bool:
        logger.warning(
            "_gst_on_ice_gathering_timeout() trying to send answer with incomplete ICE"
        )

        self._gst_resolve_answer(reason="timeout")
        return False

    def _gst_on_data_channel(self, element, data_channel) -> None:
        label = data_channel.get_property("label")
        logger.info(f"New data channel: {label!r}")

        if label == "control":
            self._control_channel = data_channel
            self._setup_control_channel_handlers()
        else:
            self._data_channel = data_channel
            self._setup_data_channel_handlers()

    def _gst_on_pad_added(self, element, pad) -> None:
        if self._stopping or self._stop_event.is_set():
            return

        if pad.direction != Gst.PadDirection.SRC:
            return

        pad_name = pad.get_name()
        logger.debug(f"New incoming pad: {pad_name}")

        # Get the pad's capabilities to determine media type
        caps = pad.get_current_caps()
        if caps is None:
            # Try to get caps from pad template
            caps = pad.query_caps(None)

        if caps is None or caps.is_empty():
            logger.warning(f"No caps for pad {pad_name}")
            return

        struct = caps.get_structure(0)
        media_type = struct.get_name()

        # Only handle video (ignore audio for now)
        if not media_type.startswith("application/x-rtp"):
            logger.debug(f"Ignoring non-RTP pad: {media_type}")
            return

        # Get encoding name to determine codec
        encoding = struct.get_string("encoding-name")
        logger.debug(f"Incoming video using {encoding} codec")

        # Check if it's video or audio
        media = struct.get_string("media")
        if media != "video" and media != "audio":
            logger.warning(f"Unknown media type: {media}")
            return

        # Map webrtcbin pad (src_0, src_1, ...) to track name via MID.
        # src_N corresponds to m-line index N in the SDP offer.
        track_name = None
        if pad_name.startswith("src_"):
            try:
                idx = int(pad_name[4:], 10)
                if 0 <= idx < len(self._mids_by_mline_index):
                    mid = self._mids_by_mline_index[idx]
                    if mid:
                        track_info = self._track_by_mid.get(mid)
                        if track_info:
                            track_name = track_info.name
            except ValueError:
                pass

        if track_name is None or track_name == "":
            logger.warning(
                f"No track name for pad {pad_name}, using media type {media}"
            )
            track_name = f"{media}_{pad_name}"

        ok, ssrc = struct.get_uint("ssrc")
        if ok:
            self._inbound_ssrc_by_track_name[track_name] = ssrc

        receiver = None
        if media == "video":
            receiver = VideoReceiver(encoding, name=track_name)
            receiver.set_on_new_sample(self._gst_on_incoming_video_sample)
            self._video_receivers[track_name] = receiver
        else:  # audio
            receiver = AudioReceiver(encoding, name=track_name)
            receiver.set_on_new_sample(self._gst_on_incoming_audio_sample)
            self._audio_receivers[track_name] = receiver

        # It's safe to assume that receiver is not None at this point
        add_many(
            self._pipeline,
            receiver,
            sync_with_parent=True,
        )
        link_pads(pad, receiver.get_sink_pad())

    def _gst_on_request_aux_sender(
        self, element, dtls_transport: GstWebRTC.WebRTCDTLSTransport
    ) -> Gst.Element:
        """Build the aux sender bin: ``rtprtxsend ! rtpgccbwe`` (or just ``rtprtxsend``)."""
        logger.debug(
            f"_gst_on_request_aux_sender() [thread:{threading.current_thread().name}]"
        )

        session_id = dtls_transport.get_property("session-id")
        rtprtxsend = make_element("rtprtxsend", f"rtprtxsend_{session_id}")
        try_set_property(
            rtprtxsend, "max-size-packets", int(VIDEO_RTX_MAX_SIZE_PACKETS)
        )
        try_set_property(
            rtprtxsend, "max-size-time", int(VIDEO_RTX_MAX_SIZE_TIME_MS) * Gst.MSECOND
        )

        smap = Gst.Structure.new_empty("application/x-rtp-ssrc-map")
        pmap = Gst.Structure.new_empty("application/x-rtp-pt-map")
        for sender in self._senders.values():
            if not isinstance(sender, VideoSender):
                continue
            ids = sender.get_rtprtx_sender_ids()
            if ids.ssrc is not None and ids.rtx_ssrc is not None:
                structure_field_uint(smap, str(int(ids.ssrc)), int(ids.rtx_ssrc))
            if ids.rtx_pt is not None and int(ids.rtx_pt) != int(ids.pt):
                structure_field_uint(pmap, str(int(ids.pt)), int(ids.rtx_pt))

        if smap.n_fields() > 0:
            try_set_property(rtprtxsend, "ssrc-map", smap)
        if pmap.n_fields() > 0:
            try_set_property(rtprtxsend, "payload-type-map", pmap)

        if Gst.ElementFactory.find("rtpgccbwe") is None:
            return rtprtxsend

        # Wrap rtprtxsend and rtpgccbwe in a single bin so webrtcbin sees one element.
        # Data path: [bin.sink] -> rtprtxsend -> rtpgccbwe -> [bin.src]
        rtpgccbwe = make_element("rtpgccbwe", f"rtpgccbwe_{session_id}")
        rtpgccbwe.set_property("min-bitrate", VIDEO_BWE_MIN_BITRATE_KBPS * 1000)
        rtpgccbwe.set_property("max-bitrate", VIDEO_BWE_MAX_BITRATE_KBPS * 1000)
        self._signals.connect(
            rtpgccbwe, "notify::estimated-bitrate", self._on_bwe_estimate
        )

        rtx_sink = rtprtxsend.get_static_pad("sink")
        rtx_src = rtprtxsend.get_static_pad("src")
        bwe_sink = rtpgccbwe.get_static_pad("sink")
        bwe_src = rtpgccbwe.get_static_pad("src")
        if not rtx_sink or not rtx_src or not bwe_sink or not bwe_src:
            logger.warning(
                "aux sender: missing static pad(s); falling back to rtprtxsend only"
            )
            return rtprtxsend

        aux_bin = Gst.Bin.new(f"aux_sender_{session_id}")
        aux_bin.add(rtprtxsend)
        aux_bin.add(rtpgccbwe)

        if rtx_src.link(bwe_sink) != Gst.PadLinkReturn.OK:
            logger.warning(
                "aux sender: failed to link rtprtxsend ! rtpgccbwe; falling back to rtprtxsend only"
            )
            # Remove rtprtxsend from aux_bin to be able to use it as a fallback
            aux_bin.remove(rtprtxsend)
            return rtprtxsend

        ghost_sink = Gst.GhostPad.new("sink", rtx_sink)
        ghost_sink.set_active(True)
        aux_bin.add_pad(ghost_sink)

        ghost_src = Gst.GhostPad.new("src", bwe_src)
        ghost_src.set_active(True)
        aux_bin.add_pad(ghost_src)

        return aux_bin

    def _on_bwe_estimate(self, element: Gst.Element, pspec: object) -> None:
        """Apply GCC bandwidth estimate when it differs enough from current aggregate usage.

        If the relative gap between the estimate and the sum of current sender targets
        exceeds :data:`~reactor_runtime.transport.webrtc.gstreamer.settings.VIDEO_BWE_TARGET_UPDATE_RELATIVE_THRESHOLD`,
        the estimate (kbps) is split evenly across video senders and each sender's target
        is forwarded to the encoder via :meth:`VideoSender.set_target_bitrate_kbps`.
        """
        estimated_bps = element.get_property("estimated-bitrate")
        estimated_kbps = estimated_bps // 1000

        video_senders = [
            s for s in self._senders.values() if isinstance(s, VideoSender)
        ]
        if not video_senders:
            return

        logger.debug(
            f"GCC bitrate estimate: {estimated_bps} bps -> {estimated_kbps} kbps "
            f"({len(video_senders)} video sender(s))"
        )

        aggregate_current_kbps = sum(s.current_bitrate_kbps for s in video_senders)
        threshold = VIDEO_BWE_TARGET_UPDATE_RELATIVE_THRESHOLD
        if aggregate_current_kbps > 0:
            relative_diff = abs(estimated_kbps - aggregate_current_kbps) / float(
                aggregate_current_kbps
            )
            if relative_diff <= threshold:
                return

        target_bitrate_kbps = int(estimated_kbps / len(video_senders))
        for sender in video_senders:
            sender.set_target_bitrate_kbps(target_bitrate_kbps)

    def _gst_on_incoming_video_sample(
        self, track_name: str, sink: Gst.Element
    ) -> Gst.FlowReturn:
        """
        Handle incoming decoded video frame from appsink.

        Converts GStreamer buffer to numpy array and emits VideoFrameEvent.
        """
        if self._stop_event.is_set() or self._stopping:
            return Gst.FlowReturn.EOS

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        if track_name in self._paused_tracks:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()

        if buf is None or caps is None:
            return Gst.FlowReturn.OK

        # Get frame dimensions from caps
        struct = caps.get_structure(0)
        width = struct.get_int("width")[1]
        height = struct.get_int("height")[1]

        # Update dimensions if changed
        if width != self._incoming_width or height != self._incoming_height:
            logger.debug(
                f"Video resolution changed: {self._incoming_width}x{self._incoming_height} -> {width}x{height}"
            )
            self._incoming_width = width
            self._incoming_height = height

        receiver = self._video_receivers.get(track_name)
        if receiver is not None:
            receiver.last_width = width
            receiver.last_height = height

        # Extract frame data
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            # Create numpy array from buffer (RGB format)
            frame: np.ndarray = np.ndarray(
                shape=(height, width, 3), dtype=np.uint8, buffer=map_info.data
            ).copy()  # Copy to own the data

            self._fire(
                self._cb_media,
                track_name,
                InputFrame(data=frame, pts=_gst_buffer_pts_seconds(buf)),
            )

        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _gst_on_incoming_audio_sample(
        self, track_name: str, sink: Gst.Element
    ) -> Gst.FlowReturn:
        """
        Handle incoming decoded audio frame from appsink.

        Converts GStreamer buffer to numpy array and emits AudioFrameEvent.
        """
        if self._stop_event.is_set() or self._stopping:
            return Gst.FlowReturn.EOS

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        if track_name in self._paused_tracks:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        if buf is None:
            return Gst.FlowReturn.OK

        # Extract frame data
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            # map_info.size is bytes; S16LE => 2 bytes per sample. Raw buffer is
            # interleaved PCM; as 1D int16 in wire byte order (little-endian).
            if map_info.size < 2 or (map_info.size % 2) != 0:
                return Gst.FlowReturn.OK
            m = map_info.size // 2
            # ``(1, M)`` int16 mono — same as :class:`~reactor_runtime.transports.media.TrackData` audio.
            frame = (
                np.frombuffer(map_info.data, dtype=np.dtype("<i2"), count=m)
                .copy()
                .reshape(1, m)
            )

            self._fire(
                self._cb_media,
                track_name,
                InputFrame(data=frame, pts=_gst_buffer_pts_seconds(buf)),
            )

        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    # =========================================================================
    # Data Channel Handlers
    # =========================================================================

    def _gst_on_data_channel_open(self, channel) -> None:
        """Handle data channel open."""
        if self._stopping:
            return

        logger.info("Data channel opened")

    def _gst_on_data_channel_close(self, channel) -> None:
        """Handle data channel close."""
        logger.info("Data channel closed")

        if self._stopping is False:
            self._run_on_gst_thread(self._gst_request_stop)

    def _gst_on_data_channel_message(self, channel, message: str) -> None:
        """Surface an inbound data-channel frame and note client liveness."""
        if self._stopping or self._stop_event.is_set():
            return
        self._fire(self._cb_message, message)
        self._fire(self._cb_ping)

    def _gst_on_control_channel_message(self, channel, message: str) -> None:
        """Note client liveness and apply this connection's track verbs.

        Every inbound control frame is evidence the client is alive, feeding the
        ping watchdog. A client also drives its own track reception over this
        channel: ``resume_track`` / ``pause_track`` notifications gate whether
        this connection's outbound senders push frames. That gate is
        per-connection — each client in a multi-client session controls its own
        streams — so it is applied here on the peer. Publisher arbitration for
        inbound tracks (``publish_track``) is cross-connection and is decided
        above the transport, not here.
        """
        if self._stopping or self._stop_event.is_set():
            return
        self._fire(self._cb_ping)
        try:
            parsed = json.loads(message)
        except (ValueError, TypeError):
            return
        if not isinstance(parsed, dict) or parsed.get("type") != "notification":
            return
        event = parsed.get("event")
        name = str((parsed.get("data") or {}).get("name", ""))
        if event == "resume_track":
            self._gst_resume_track(name)
        elif event == "pause_track":
            self._gst_pause_track(name)

    def resume_track(self, name: str) -> None:
        self._gst_resume_track(name)

    def pause_track(self, name: str) -> None:
        self._gst_pause_track(name)

    def resume_sender_tracks(self) -> None:
        for track in self._sender_tracks:
            self._gst_resume_track(track.name)

    def _gst_pause_track(self, track_name: str) -> None:
        self._paused_tracks.add(track_name)
        logger.info(f"Paused track {track_name!r}")

    def _gst_resume_track(self, track_name: str) -> None:
        self._paused_tracks.discard(track_name)
        logger.info(f"Resumed track {track_name!r}")

    # =========================================================================
    # SDP Handling
    # =========================================================================

    def _gst_set_remote_description(self, sdp_offer: str) -> None:
        """Set the remote SDP description."""
        logger.debug(
            f"_gst_set_remote_description() [thread:{threading.current_thread().name}]"
        )
        webrtc = self._require_webrtc()

        sanitized_sdp, candidates = strip_ice_candidates_from_sdp(sdp_offer)
        self._remote_candidates = candidates
        self._remote_offer_sdp = sdp_offer

        res, sdpmsg = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(bytes(sanitized_sdp, "utf-8"), sdpmsg)

        offer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.OFFER, sdpmsg
        )

        promise = Gst.Promise.new_with_change_func(self._gst_on_offer_set)
        logger.debug(
            f"_gst_set_remote_description() emitting set-remote-description with offer [thread:{threading.current_thread().name}]"
        )
        webrtc.emit("set-remote-description", offer, promise)

    def _gst_on_offer_set(self, p) -> None:
        """Callback when remote description is set."""
        logger.debug(f"_gst_on_offer_set() [thread:{threading.current_thread().name}]")
        if self._webrtc is None:
            return

        self._run_on_gst_thread(self._gst_add_remote_candidates)

        # Create answer
        promise = Gst.Promise.new_with_change_func(self._gst_on_answer_created)
        logger.debug(
            f"_gst_on_offer_set() emitting create-answer [thread:{threading.current_thread().name}]"
        )
        self._webrtc.emit("create-answer", None, promise)

    def _gst_on_answer_created(self, promise) -> None:
        """Callback when answer is created."""
        logger.debug(
            f"_gst_on_answer_created() [thread:{threading.current_thread().name}]"
        )

        reply = promise.get_reply()

        if reply is None or self._webrtc is None:
            self._fail_future_threadsafe(
                self._answer_fut, RuntimeError("No reply for answer")
            )
            return

        answer = reply.get_value("answer")
        if answer is None:
            self._fail_future_threadsafe(
                self._answer_fut, RuntimeError("No answer in reply")
            )
            return

        logger.debug(f"Answer SDP (raw): {answer.sdp.as_text()}")

        offer_sdp = self._remote_offer_sdp or ""
        negotiated = negotiated_sdp_extmaps_by_mid(
            offer_sdp, SUPPORTED_RTP_HEADER_EXTENSION_URIS
        )

        extmaps_by_mid: Dict[str, List[SdpExtmap]] = {}
        for mid in self._mids_by_mline_index:
            exts = negotiated.get(mid)
            if exts:
                extmaps_by_mid[mid] = list(exts)
        if extmaps_by_mid:
            answer = add_extmaps_per_mid_to_sdp(answer, extmaps_by_mid)

        logger.debug(
            f"Answer SDP (after negotiated RTP header extmaps per mid): {answer.sdp.as_text()}"
        )

        # Rewrite webrtc SSRC / FID signaling before set-local-description so the
        # SDP webrtcbin stores (and any later get_property("local-description")) matches
        # what we send to the browser. Previously this ran only in _gst_resolve_answer,
        # leaving raw answers with invalid FID primary SSRC (e.g. 0) in local-desc.
        sdp_text = answer.sdp.as_text()
        sdp_text = add_answer_webrtc_attributes(
            sdp_text,
            ssrc_by_mid=self._ssrc_by_mid,
            cname=self._cname,
            rtx_ssrc_by_mid=self._rtx_ssrc_by_mid,
        )
        _, sdpmsg = GstSdp.SDPMessage.new()
        parse_ret = GstSdp.sdp_message_parse_buffer(bytes(sdp_text, "utf-8"), sdpmsg)
        if parse_ret != GstSdp.SDPResult.OK:
            self._fail_future_threadsafe(
                self._answer_fut,
                RuntimeError(
                    f"Failed to parse SDP after SSRC/FID rewrite: {parse_ret!r}"
                ),
            )
            return

        answer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg
        )
        logger.debug(f"Answer SDP (after SSRC / FID rewrite): {answer.sdp.as_text()}")

        # Set local description
        self._webrtc.emit("set-local-description", answer, None)

    def _gst_add_remote_candidates(self):
        logger.debug(
            f"_gst_add_remote_candidates() [thread:{threading.current_thread().name}]"
        )

        bundle_policy = self._webrtc.get_property("bundle-policy")

        for c in self._remote_candidates:
            # if c.mline_index == 0:
            self._webrtc.emit(
                "add-ice-candidate",
                (
                    c.mline_index
                    if bundle_policy == GstWebRTC.WebRTCBundlePolicy.MAX_COMPAT
                    else 0
                ),
                c.candidate,
            )

        self._start_ice_gathering_timeout(self._config.ice_gathering_timeout_ms)

    def _gst_clear_ice_gathering_timeout(self) -> None:
        logger.debug(
            f"_gst_clear_ice_gathering_timeout() [thread:{threading.current_thread().name}]"
        )
        if self._ice_gathering_timeout_source is not None:
            self._ice_gathering_timeout_source.destroy()
            self._ice_gathering_timeout_source = None

    def _gst_resolve_answer(self, reason: str) -> None:
        logger.debug(
            f"_gst_resolve_answer() [reason:{reason}, thread:{threading.current_thread().name}]"
        )

        self._run_on_gst_thread(self._gst_clear_ice_gathering_timeout)

        webrtc = self._require_webrtc()
        local_desc = webrtc.get_property("local-description")
        if not local_desc:
            logger.warning("SDP ANSWER NOT READY!!!!")
            return

        # SSRC / FID signaling was applied in _gst_on_answer_created before
        # set-local-description; reuse that SDP here.
        self._answer_sdp = local_desc.sdp.as_text()

        logger.debug(
            f"_gst_resolve_answer() got sdp answer [thread:{threading.current_thread().name}]"
        )
        self._complete_future_threadsafe(self._answer_fut, self._answer_sdp)

    # =========================================================================
    # Stats (overrides base)
    # =========================================================================

    async def _get_rtc_stats(
        self,
    ) -> tuple[Optional[float], list[TrackStats]]:
        """Collect RTT and per-track stats in a single webrtcbin get-stats call.

        RTT comes from the first remote-inbound-rtp entry that has a
        round-trip-time.  Outbound tracks are keyed via _ssrc_by_mid; inbound
        tracks via _inbound_ssrc_by_track_name (populated in _gst_on_pad_added).
        """
        if self._stopping or not self._webrtc or not self._loop:
            return None, []

        loop = self._loop
        future: asyncio.Future[tuple[Optional[float], list[TrackStats]]] = (
            loop.create_future()
        )

        def _set_result(val: tuple[Optional[float], list[TrackStats]]) -> None:
            if not future.done():
                future.set_result(val)

        def _on_stats_promise(promise: Gst.Promise) -> None:
            try:
                reply = promise.get_reply()
                if reply is None:
                    loop.call_soon_threadsafe(_set_result, (None, []))
                    return

                outbound: dict[int, Gst.Structure] = {}
                remote_inbound: dict[int, Gst.Structure] = {}
                inbound: dict[int, Gst.Structure] = {}

                for i in range(reply.n_fields()):
                    s = reply.get_value(reply.nth_field_name(i))
                    if not isinstance(s, Gst.Structure) or not s.has_field("type"):
                        continue
                    stats_type = s.get_value("type")
                    if stats_type == GstWebRTC.WebRTCStatsType.REMOTE_INBOUND_RTP:
                        # RTT lives here; so does the SSRC needed for per-track loss.
                        if s.has_field("ssrc"):
                            ok, ssrc = s.get_uint("ssrc")
                            if ok:
                                remote_inbound[ssrc] = s
                    elif stats_type == GstWebRTC.WebRTCStatsType.OUTBOUND_RTP:
                        if s.has_field("ssrc"):
                            ok, ssrc = s.get_uint("ssrc")
                            if ok:
                                outbound[ssrc] = s
                    elif stats_type == GstWebRTC.WebRTCStatsType.INBOUND_RTP:
                        if s.has_field("ssrc"):
                            ok, ssrc = s.get_uint("ssrc")
                            if ok:
                                inbound[ssrc] = s

                # RTT — first remote-inbound-rtp with a positive round-trip-time.
                rtt: Optional[float] = None
                for r_in in remote_inbound.values():
                    if r_in.has_field("round-trip-time"):
                        ok, val = r_in.get_double("round-trip-time")
                        if ok and val > 0:
                            rtt = val
                            break

                # Outbound (sender) tracks.
                mid_by_ssrc: dict[int, str] = {
                    v: k for k, v in self._ssrc_by_mid.items()
                }
                tracks: list[TrackStats] = []
                for ssrc, out_s in outbound.items():
                    mid = mid_by_ssrc.get(ssrc)
                    if mid is None:
                        continue
                    track_info = self._track_by_mid.get(mid)
                    if track_info is None:
                        continue

                    packets_sent: Optional[int] = None
                    if out_s.has_field("packets-sent"):
                        ok, v = out_s.get_uint64("packets-sent")
                        if ok:
                            packets_sent = v

                    bitrate_bps: Optional[int] = None
                    fps: Optional[float] = None
                    width: Optional[int] = None
                    height: Optional[int] = None
                    codec_name: Optional[str] = None
                    qos: Optional[float] = None
                    sender = self._senders.get(mid)
                    if isinstance(sender, VideoSender):
                        bitrate_bps = sender.current_bitrate_kbps * 1000
                        fps = round(sender.fps, 2) if sender.fps is not None else None
                        width = sender.width
                        height = sender.height
                        codec_name = sender.codec_name
                        qos = sender.qos()
                    elif isinstance(sender, AudioSender):
                        bitrate_bps = sender.current_bitrate_kbps * 1000
                        codec_name = sender.codec_name

                    packet_loss: Optional[int] = None
                    jitter: Optional[float] = None
                    r_in = remote_inbound.get(ssrc)
                    if r_in is not None:
                        if r_in.has_field("packets-lost"):
                            ok, v = r_in.get_int("packets-lost")
                            if ok:
                                packet_loss = max(0, v)
                        if r_in.has_field("jitter"):
                            ok, v = r_in.get_double("jitter")
                            if ok:
                                jitter = v

                    tracks.append(
                        TrackStat(
                            name=track_info.name,
                            direction=TrackDirection.OUT,
                            fps=fps,
                            bitrate_bps=bitrate_bps,
                            packet_loss=packet_loss,
                            jitter=jitter,
                        )
                    )

                # Inbound (receiver) tracks.
                track_name_by_inbound_ssrc: dict[int, str] = {
                    v: k for k, v in self._inbound_ssrc_by_track_name.items()
                }
                for ssrc, in_s in inbound.items():
                    track_name = track_name_by_inbound_ssrc.get(ssrc)
                    if track_name is None:
                        continue

                    packets_received: Optional[int] = None
                    if in_s.has_field("packets-received"):
                        ok, v = in_s.get_uint64("packets-received")
                        if ok:
                            packets_received = v

                    packet_loss_in: Optional[int] = None
                    if in_s.has_field("packets-lost"):
                        ok, v = in_s.get_int("packets-lost")
                        if ok:
                            packet_loss_in = max(0, v)

                    jitter_in: Optional[float] = None
                    if in_s.has_field("jitter"):
                        ok, v = in_s.get_double("jitter")
                        if ok:
                            jitter_in = v

                    receiver_fps: Optional[float] = None
                    receiver_width: Optional[int] = None
                    receiver_height: Optional[int] = None
                    receiver_codec: Optional[str] = None
                    receiver = self._video_receivers.get(track_name)
                    if receiver is not None:
                        receiver_fps = (
                            round(receiver.fps, 2) if receiver.fps is not None else None
                        )
                        receiver_width = receiver.last_width
                        receiver_height = receiver.last_height
                        receiver_codec = receiver.codec_name

                    tracks.append(
                        TrackStat(
                            name=track_name,
                            direction=TrackDirection.IN,
                            fps=receiver_fps,
                            bitrate_bps=None,
                            packet_loss=packet_loss_in,
                            jitter=jitter_in,
                        )
                    )

                loop.call_soon_threadsafe(_set_result, (rtt, tracks))
            except Exception:
                loop.call_soon_threadsafe(_set_result, (None, []))

        def _emit_get_stats() -> None:
            if self._webrtc is None or self._stopping:
                loop.call_soon_threadsafe(_set_result, (None, []))
                return
            promise = Gst.Promise.new_with_change_func(_on_stats_promise)
            self._webrtc.emit("get-stats", None, promise)

        self._run_on_gst_thread(_emit_get_stats)

        try:
            return await asyncio.wait_for(future, timeout=2.0)
        except asyncio.TimeoutError:
            return None, []

    # =========================================================================
    # GStreamer thread: events
    # =========================================================================

    def _fire(self, callback: Optional[Callable[..., None]], *args: Any) -> None:
        """Invoke an inbound callback on the runtime loop, if one is registered.

        Thread-safe: marshals onto the asyncio loop the connection runs on, so a
        callback the connection registered never executes on the GLib thread.
        """
        loop = self._loop
        if loop is None or callback is None:
            return
        loop.call_soon_threadsafe(callback, *args)

    # =========================================================================
    # GStreamer thread: message sending
    # =========================================================================

    def _gst_send_datachannel_msg(self, payload: bytes | str) -> None:
        if self._stopping or self._data_channel is None:
            return
        try:
            if isinstance(payload, bytes):
                self._data_channel.emit("send-data", GLib.Bytes.new(payload))
            else:
                self._data_channel.emit("send-string", payload)
        except Exception:
            pass

    # =========================================================================
    # Sending data (implements ABC)
    # =========================================================================

    def send_media(self, bundle: MediaBundle) -> None:
        """Send a multi-track media bundle to the remote peer.

        Currently only video tracks are forwarded to the GStreamer appsrc; audio
        tracks are dropped. Thread-safe — callable from any thread. A bundle that
        arrives before the senders exist, or when the outgoing queue is full, is
        dropped rather than blocking the caller.
        """
        if self._stop_event.is_set() or self._stopping:
            return

        if len(self._senders) == 0:
            return

        try:
            self._frame_q.put_nowait(bundle)
        except queue.Full:
            pass
        except Exception as e:
            logger.warning(f"Failed to send media bundle: {e}")

    def send_message(self, payload: bytes | str) -> None:
        """Send an already-encoded frame over the data channel (text or binary).

        Thread-safe: the emit is marshalled onto the GLib thread.
        """
        if self._stop_event.is_set() or self._stopping:
            return
        self._run_on_gst_thread(self._gst_send_datachannel_msg, payload)

    async def add_ice(self, candidate: TrickleCandidate) -> None:
        """Add a trickle-ICE candidate to ``webrtcbin``.

        The client supplies each candidate with the m-line index of the
        m-section it belongs to. ``webrtcbin`` only accepts an index that
        matches the bundle policy: under ``MAX_COMPAT`` each m-line has its own
        ICE transport so the index is preserved, but for every other policy
        (notably ``MAX_BUNDLE``, the browser default) every candidate must be
        emitted with index ``0`` because everything rides on one ICE transport.
        This mirrors the normalisation in :meth:`_gst_add_remote_candidates`.

        A candidate without an m-line index is dropped: the GLib signal expects
        a ``uint`` and there is no way to tell ``webrtcbin`` which m-section to
        attach it to.
        """
        mline_index = candidate.sdp_mline_index
        if mline_index is None:
            logger.warning("Dropping trickle ICE candidate without an m-line index")
            return

        candidate_str = candidate.candidate

        def _gst_add_ice_candidate() -> None:
            if self._webrtc is None:
                return
            bundle_policy = self._webrtc.get_property("bundle-policy")
            idx = (
                mline_index
                if bundle_policy == GstWebRTC.WebRTCBundlePolicy.MAX_COMPAT
                else 0
            )
            self._webrtc.emit("add-ice-candidate", idx, candidate_str)

        self._run_on_gst_thread(_gst_add_ice_candidate)

    # =========================================================================
    # Lifecycle (implements ABC)
    # =========================================================================

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the WebRTC client.

        Args:
            timeout: Maximum time to wait for cleanup.
        """
        if self._stop_event.is_set():
            return

        self._stop_event.set()

        if not self._connected.is_set():
            self._was_superseded = True

        # Unblock any awaiters
        self._fail_future_threadsafe(
            self._answer_fut, WebRTCSupersededError("Client stopped")
        )
        self._fail_future_threadsafe(
            self._gst_ready_fut, WebRTCSupersededError("Client stopped")
        )

        # Schedule teardown on GLib thread
        if self._main_loop is not None and self._main_loop.is_running():
            self._run_on_gst_thread(self._gst_request_stop)

        # Join thread
        if self._gst_thread is not None and self._gst_thread.is_alive():
            self._gst_thread.join(timeout=timeout)
            if self._gst_thread.is_alive():
                logger.warning("GStreamer thread did not exit cleanly")

        self._gst_thread = None

    def _gst_destroy_all_sources(self) -> None:
        with self._gst_sources_lock:
            sources = list(self._gst_sources)
            self._gst_sources.clear()

        for src in sources:
            try:
                src.destroy()
            except Exception:
                pass

        if self._ice_gathering_timeout_source is not None:
            self._ice_gathering_timeout_source.destroy()
            self._ice_gathering_timeout_source = None

    def _gst_request_stop(self) -> None:
        """
        GLib thread only: begin shutdown sequence.
        """
        if self._stopping:
            return
        self._stopping = True

        self._gst_finalize_stop()

    def _gst_finalize_stop(self) -> None:
        """
        GLib thread only: set pipeline to NULL, drop refs, quit loop.
        """
        logger.debug(f"_gst_finalize_stop() [thread:{threading.current_thread().name}]")
        try:
            self._gst_destroy_all_sources()

            # Block (don't disconnect) the tracked handlers, then dispose the
            # pipeline. Setting the pipeline to NULL finalizes its elements and
            # GStreamer drops their signal handlers as part of that disposal;
            # an explicit disconnect_all() afterwards would call .disconnect()
            # on already-finalized GObjects, double-unreffing them. That is a
            # native GStreamer-CRITICAL ("ref_count > 0" assertion) / heap
            # corruption, not a catchable Python exception, and it surfaces as
            # a non-deterministic segfault under concurrent connections.
            self._signals.block_all()

            if self._pipeline is not None:
                self._pipeline.set_state(Gst.State.NULL)
                ret, cur, pend = self._pipeline.get_state(1 * Gst.SECOND)
                logger.debug(
                    "Pipeline NULL state reached", result=ret, current=cur, pending=pend
                )
        finally:
            if self._main_loop is not None and self._main_loop.is_running():
                self._main_loop.quit()

    def is_connected(self) -> bool:
        """Return ``True`` if connected and not stopped."""
        return self._connected.is_set() and not self._stop_event.is_set()

    @property
    def is_stopped(self) -> bool:
        """``True`` if :meth:`stop` has been requested."""
        return self._stop_event.is_set()

    @property
    def was_superseded(self) -> bool:
        """``True`` if this transport was stopped during setup."""
        return self._was_superseded

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        if not self._stop_event.is_set():
            try:
                self.stop(timeout=1.0)
            except Exception:
                pass


# =============================================================================
# Test environment helpers (local IP selection, GST test mode toggles)
# =============================================================================


def _use_local_ip_addr() -> bool:
    return environ.get("GST_TEST_ENV") == "true"


# =============================================================================
# Factory
# =============================================================================


async def gstreamer_peer_factory(
    conn_id: ConnId,
    offer: SdpOffer,
    tracks: TrackMap,
    config: WebRtcConfig,
) -> tuple[GStreamerPeer, SdpAnswer]:
    """Negotiate *offer* into a :class:`GStreamerPeer` and its SDP answer.

    Conforms to :data:`~reactor_runtime.transport.webrtc.peer.WebRtcPeerFactory`.
    Threads the connection's config into the pipeline and builds, from the
    client's declared track map, the mid/name lookups the media setup needs.
    """
    logger.debug("Negotiating GStreamer peer for connection %s", conn_id)
    peer = GStreamerPeer(ping_timeout_seconds=config.ping_timeout)
    peer._config = config
    peer._track_by_mid = {mt.mid: mt.info for mt in tracks.tracks}
    peer._mid_by_name = {mt.info.name: mt.mid for mt in tracks.tracks}
    peer._transport_policy = _to_gst_transport_policy(config.transport_policy)
    peer._port_range = config.port_range
    answer_sdp = await peer._negotiate(offer.sdp, list(config.ice_servers) or None)
    return peer, SdpAnswer(sdp=answer_sdp)
