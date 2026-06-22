"""GStreamer WebRTC media engine.

The concrete media path behind the WebRTC peer seam: encoders, decoders,
senders, receivers, and the SDP/ICE helpers that drive a ``webrtcbin``
pipeline. Ported from the original runtime and kept close to it. The peer that
orchestrates these pieces into a :class:`~reactor_runtime.transport.webrtc.peer.WebRtcPeer`
lives alongside this package.
"""
