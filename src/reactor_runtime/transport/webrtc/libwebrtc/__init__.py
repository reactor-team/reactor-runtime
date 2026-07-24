"""libwebrtc WebRTC media engine.

The concrete media path behind the WebRTC peer seam, backed by the
``reactor_webrtc`` PyO3 module (libwebrtc). It offers the same
:class:`~reactor_runtime.transport.webrtc.peer.WebRtcPeer` shape as the GStreamer
engine — :func:`~reactor_runtime.transport.webrtc.libwebrtc.peer.libwebrtc_peer_factory`
negotiates an offer into a peer and its SDP answer — but delegates encoding,
RTP/RTCP, ICE, and DTLS-SRTP to libwebrtc rather than driving an explicit
pipeline. The peer and factory live in :mod:`.peer`; importing them is what pulls
in the native module, so this package root stays import-light.
"""
