"""The WebRTC transport.

WebRTC reshaped as a connection: :class:`WebRTCConnection` is the wire and
:class:`WebRTCAcceptor` concentrates its SDP/ICE signalling. Its media engine
sits behind the :class:`WebRtcPeer` seam, supplied by a
:data:`~reactor_runtime.transport.webrtc.peer.WebRtcPeerFactory`, so the wire can
be built and tested without the media stack present.
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
]
