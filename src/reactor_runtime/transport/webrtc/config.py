"""WebRTC connection configuration.

The tunables the acceptor threads into every connection it builds: the ICE
servers and policy that shape candidate gathering, the UDP port range, the
congestion-control bitrate limits, and the liveness timeout the connection's
ping watchdog enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NotRequired, TypedDict


class IceTransportPolicy(StrEnum):
    """Which ICE candidate types to gather.

    ``ALL`` gathers host, server-reflexive, and relay candidates; ``RELAY``
    gathers only relay (TURN) candidates, forcing every flow through a TURN
    server.
    """

    ALL = "all"
    RELAY = "relay"


@dataclass(frozen=True)
class IceServer:
    """One STUN or TURN server offered to the peer for candidate gathering.

    Attributes:
        urls: One or more STUN/TURN URLs for this server.
        username: Username for TURN authentication, or ``None`` for STUN.
        credential: Credential for TURN authentication, or ``None`` for STUN.
    """

    urls: tuple[str, ...]
    username: str | None = None
    credential: str | None = None


@dataclass(frozen=True)
class IceCredentials:
    """The ICE username fragment and password a connection answers with.

    Normally the media engine generates these itself, and nothing needs to
    supply them. They exist as a configurable value for deployments that front
    the runtime with a relaying layer that has to recognise a connection from
    its ICE credentials alone — the credentials are the only field an ICE agent
    echoes on every connectivity check, so a fronting layer can route on them
    without inspecting media.

    Both values must satisfy RFC 8445's ``ice-char`` alphabet and length
    ranges; :meth:`reactor_webrtc.SessionDescription.with_ice_credentials`
    rejects anything outside them.

    Attributes:
        ufrag: The ``a=ice-ufrag`` value to answer with.
        pwd: The ``a=ice-pwd`` value to answer with.
    """

    ufrag: str
    pwd: str


class CodecEntry(TypedDict):
    """A supported codec for SDP negotiation and encoder/decoder selection.

    Attributes:
        codec: The codec name (e.g. ``"VP9"``, ``"H264"``, ``"Opus"``).
        parameters: SDP ``fmtp`` parameters that must match, each a string value.
        payload_type: The payload type, set when a codec is read back from an
            offer; absent in a configured preference.
    """

    codec: str
    parameters: NotRequired[dict[str, str]]
    payload_type: NotRequired[int | None]


# The RTP header extension the transport mirrors in answers and on outgoing video
# RTP caps when the offer includes it: transport-wide congestion control.
_TRANSPORT_WIDE_CC = "http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01"

# Default supported codecs, in preference order: the first that appears in the
# remote offer is chosen.
_DEFAULT_VIDEO_CODECS: tuple[CodecEntry, ...] = (
    {"codec": "VP9", "parameters": {"profile-id": "0"}},
    {"codec": "VP8"},
    {"codec": "H264", "parameters": {"profile-level-id": "42e01f", "packetization-mode": "1"}},
    {"codec": "AV1", "parameters": {"profile": "0"}},
    {"codec": "H265"},
)
_DEFAULT_AUDIO_CODECS: tuple[CodecEntry, ...] = ({"codec": "Opus"},)


@dataclass(frozen=True)
class WebRtcConfig:
    """Configuration the acceptor applies to every connection it negotiates.

    This is the transport's one configuration object. The media engine reads
    every tunable from here rather than from the environment; an environment
    adapter that builds one of these is a separate, higher layer.

    Attributes:
        ice_servers: The STUN/TURN servers offered for candidate gathering.
        ice_credentials: The ICE credentials to answer with, or ``None`` — the
            default — to let the media engine generate its own. Setting these
            is only useful to a deployment whose fronting layer routes on them;
            see :class:`IceCredentials`.
        port_range: An inclusive ``(min, max)`` UDP port range to confine ICE
            to, or ``None`` to let the stack choose. A single-port range pins
            the connection to one port, which is what a fronting layer needs if
            it must know the media address before the connection exists.
        transport_policy: Which candidate types to gather.
        ping_timeout: Seconds without a client ping before the connection's
            watchdog declares it lost. ``0`` or less disables the watchdog.
        supported_video_codecs: Video codecs offered, in preference order.
        supported_audio_codecs: Audio codecs offered, in preference order.
        hw_codecs_enabled: Whether to use hardware encoder/decoder elements
            instead of software ones.
        rtp_header_extensions: RTP header extension URIs mirrored in answers
            when the offer includes them.
        bwe_min_kbps: Floor for the congestion-control bitrate estimate.
        bwe_max_kbps: Ceiling for the congestion-control bitrate estimate.
        bwe_initial_kbps: Starting bitrate before estimates arrive.
        bwe_target_update_threshold: Relative change below which a new bitrate
            estimate is ignored rather than re-applied to the encoders.
        sender_max_kbps: Ceiling for each sendonly *video* track's own encoder,
            which is
            a different limit from ``bwe_max_kbps`` and the one that actually
            caps a video stream. The two are conjunctive — the lower wins — and
            without this a sender's maximum comes from libwebrtc's
            resolution-keyed default, which is 2500 kbps for anything above
            960x540. Every frame size we send at 720p or larger would cap at
            2.5 Mbps no matter how much headroom the estimate had. Audio senders
            are left alone: that default is keyed on frame size, so there is no
            equivalent for them to clear. ``0`` or less leaves the libwebrtc
            default in place.
        sender_min_kbps: Floor for each sendonly video track's own encoder. ``0`` or
            less leaves it unset, which is the default: a floor stops the
            encoder degrading gracefully, so on a link that cannot sustain it
            the trade is lower quality for packet loss. Useful mainly when
            several tracks compete and one must be preserved.
        rtx_max_size_packets: Retransmission history depth, in packets.
        rtx_max_size_time_ms: Retransmission history depth, in milliseconds;
            ``0`` means no time limit.
        rtp_payload_mtu: The ``mtu`` (bytes) applied to every RTP payloader.
            1400 bytes exceeds the effective MTU of common tunneled paths
            (WireGuard and Tailscale links are 1280) once SRTP/UDP/IP overhead
            is added; oversized packets are silently dropped there while ICE
            checks and data-channel traffic fit and succeed. The default matches
            the ~1200-byte ceiling libwebrtc uses for its own media packets.
        ice_tcp: Whether the ICE agent gathers TCP candidates.
        upnp: Whether the ICE agent attempts UPnP port mapping.
        ice_gathering_timeout_ms: How long to wait for ICE gathering before
            resolving the SDP answer with whatever candidates are in hand.
        max_connections: The most connections the acceptor negotiates at once.
            Each holds a native media peer, so this caps the peers one session
            can be driven to build: an offer for a new connection past the
            ceiling is refused, while a re-offer on a live connection (a
            reconnect) is always admitted. ``0`` or less removes the ceiling.
        negotiation_timeout: Seconds a connection has to reach its live wire
            after its offer lands. A connection that answers but never completes
            ICE is closed and its slot freed once this passes, so a stalled or
            hostile half-open offer cannot hold a slot against ``max_connections``
            for the process's life. ``0`` or less disables the deadline.
    """

    ice_servers: tuple[IceServer, ...] = ()
    ice_credentials: IceCredentials | None = None
    port_range: tuple[int, int] | None = None
    transport_policy: IceTransportPolicy = IceTransportPolicy.ALL
    ping_timeout: float = 20.0
    supported_video_codecs: tuple[CodecEntry, ...] = _DEFAULT_VIDEO_CODECS
    supported_audio_codecs: tuple[CodecEntry, ...] = _DEFAULT_AUDIO_CODECS
    hw_codecs_enabled: bool = False
    rtp_header_extensions: tuple[str, ...] = (_TRANSPORT_WIDE_CC,)
    bwe_min_kbps: int = 500
    bwe_max_kbps: int = 10000
    bwe_initial_kbps: int = 4000
    bwe_target_update_threshold: float = 0.05
    sender_max_kbps: int = 10000
    sender_min_kbps: int = 0
    rtx_max_size_packets: int = 512
    rtx_max_size_time_ms: int = 200
    rtp_payload_mtu: int = 1200
    ice_tcp: bool = False
    upnp: bool = False
    ice_gathering_timeout_ms: int = 3000
    max_connections: int = 64
    negotiation_timeout: float = 30.0
