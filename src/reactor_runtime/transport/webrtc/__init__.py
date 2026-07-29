"""The WebRTC transport.

WebRTC reshaped as a connection: :class:`WebRTCConnection` is the wire,
:class:`WebRTCAcceptor` concentrates its SDP/ICE signalling, and
:class:`WebRtcRouter` mounts its routes. The media engine is
:class:`~reactor_runtime.transport.webrtc.peer.WebRTCPeer`, built
during offer negotiation by a :data:`WebRtcPeerFactory`.
"""

from reactor_runtime.transport.webrtc.acceptor import WebRTCAcceptor
from reactor_runtime.transport.webrtc.config import IceServer, IceTransportPolicy, WebRtcConfig
from reactor_runtime.transport.webrtc.connection import WebRTCConnection
from reactor_runtime.transport.webrtc.peer import WebRTCPeer, WebRtcPeerFactory
from reactor_runtime.transport.webrtc.router import WebRtcRouter
from reactor_runtime.transport.webrtc.signaling import (
    IceCandidate,
    MappedTrack,
    SdpAnswer,
    SdpOffer,
    TrackMap,
)
from reactor_runtime.transport.webrtc.stats import PeerStats, TrackStat

__all__ = [
    "IceCandidate",
    "IceServer",
    "IceTransportPolicy",
    "MappedTrack",
    "PeerStats",
    "SdpAnswer",
    "SdpOffer",
    "TrackMap",
    "TrackStat",
    "WebRTCAcceptor",
    "WebRTCConnection",
    "WebRTCPeer",
    "WebRtcConfig",
    "WebRtcPeerFactory",
    "WebRtcRouter",
]
