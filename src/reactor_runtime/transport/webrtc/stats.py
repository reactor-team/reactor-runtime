"""WebRTC peer statistics types.

Sampled by :class:`~reactor_runtime.transport.webrtc.peer.WebRTCPeer` and surfaced
through :class:`~reactor_runtime.transport.webrtc.connection.WebRTCConnection`. The
types live in :mod:`reactor_runtime.core.stats`, since any transport reports them.
"""

from reactor_runtime.core.stats import OutboundMediaHealth, PeerStats, TrackStat

__all__ = ["OutboundMediaHealth", "PeerStats", "TrackStat"]
