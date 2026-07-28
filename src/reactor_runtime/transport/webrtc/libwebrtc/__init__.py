"""libwebrtc WebRTC media engine.

The concrete media path behind the WebRTC peer seam, backed by the
``reactor_webrtc`` PyO3 module (libwebrtc).
:func:`~reactor_runtime.transport.webrtc.libwebrtc.peer.libwebrtc_peer_factory`
negotiates an offer into a peer and its SDP answer, delegating encoding,
RTP/RTCP, ICE, and DTLS-SRTP to libwebrtc. The peer and factory live in
:mod:`.peer`; importing them is what pulls in the native module, so this package
root stays import-light.
"""
