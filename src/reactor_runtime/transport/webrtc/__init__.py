"""The WebRTC transport.

WebRTC reshaped as a connection: :class:`WebRTCConnection` is the wire,
:class:`WebRTCAcceptor` concentrates its SDP/ICE signalling, and
:class:`WebRtcRouter` mounts its routes. The media engine sits behind the
:class:`WebRtcPeer` seam, supplied by a
:data:`~reactor_runtime.transport.webrtc.peer.WebRtcPeerFactory`.
"""

from reactor_runtime.transport.webrtc.acceptor import WebRTCAcceptor
from reactor_runtime.transport.webrtc.config import IceServer, IceTransportPolicy, WebRtcConfig
from reactor_runtime.transport.webrtc.connection import WebRTCConnection
from reactor_runtime.transport.webrtc.peer import (
    PeerStats,
    TrackStat,
    WebRtcPeer,
    WebRtcPeerFactory,
)
from reactor_runtime.transport.webrtc.router import WebRtcRouter
from reactor_runtime.transport.webrtc.signaling import (
    IceCandidate,
    MappedTrack,
    SdpAnswer,
    SdpOffer,
    TrackMap,
)

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
    "WebRtcConfig",
    "WebRtcPeer",
    "WebRtcPeerFactory",
    "WebRtcRouter",
]
