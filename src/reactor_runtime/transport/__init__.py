"""Transport edge: the seam between a wire and the runner.

A transport plugs in below the runner and reaches it only through the neutral
:class:`~reactor_runtime.core.transport.Connection` and
:class:`~reactor_runtime.core.transport.ConnectionSink` protocols. A
:class:`~reactor_runtime.transport.acceptor.ConnectionAcceptor` concentrates one
transport's handshake. The concrete WebRTC transport lives in
:mod:`reactor_runtime.transport.webrtc`.
"""

from reactor_runtime.transport.acceptor import ConnectionAcceptor

__all__ = [
    "ConnectionAcceptor",
]
