"""Transport edge: the seam between a wire and the runner.

The neutral bases every transport plugs into. A
:class:`~reactor_runtime.transport.acceptor.ConnectionAcceptor` concentrates one
transport's handshake; a :class:`~reactor_runtime.transport.router.TransportRouter`
mounts its routes and binds it to the runner through the
:class:`~reactor_runtime.transport.router.SessionControl` surface. The concrete
WebRTC transport lives in :mod:`reactor_runtime.transport.webrtc`.
"""

from reactor_runtime.transport.acceptor import ConnectionAcceptor
from reactor_runtime.transport.router import (
    SessionControl,
    SessionNotRunningError,
    TransportRouter,
    UnknownSessionError,
)

__all__ = [
    "ConnectionAcceptor",
    "SessionControl",
    "SessionNotRunningError",
    "TransportRouter",
    "UnknownSessionError",
]
