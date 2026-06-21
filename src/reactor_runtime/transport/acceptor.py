"""The connection-acceptor base.

An acceptor negotiates one transport's handshake, builds a
:class:`~reactor_runtime.core.transport.Connection`, and registers it upward
through a :class:`~reactor_runtime.core.transport.ConnectionSink` once its wire
is live. There is one acceptor per transport that has a real handshake to run
(WebRTC's SDP/ICE); transports with no negotiation have none and let their
router play the role directly.

The handshake-specific surface (``offer`` / ``add_ice`` for WebRTC) lives on the
concrete subclass, not here: different transports negotiate differently, and the
signalling value types stay below the acceptor so nothing above it depends on
them.
"""

from __future__ import annotations

from abc import ABC


# B024: the base declares no abstract methods on purpose — each transport's
# handshake surface differs (WebRTC's offer/add_ice has no analogue in a
# handshake-less transport), so the base only names the category.
class ConnectionAcceptor(ABC):  # noqa: B024
    """Negotiate a transport handshake, build a connection, register it upward.

    The category base for the place a transport's signalling is concentrated. A
    concrete acceptor is constructed bound to a sink, owns the in-flight
    handshakes for its transport, and hands a connection to the sink only once
    the wire reaches its connected state — a half-open handshake never reaches
    the runner. The dependency arrow points acceptor to sink and never back.
    """
