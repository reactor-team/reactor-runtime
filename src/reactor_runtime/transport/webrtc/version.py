"""Map a client's WebRTC transport version to the wire codec it speaks.

A client declares one version at the handshake — its WebRTC transport version,
on the ``Reactor-WebRTC-Version`` header. The wire codec on the data channel is
a function of that single axis, not a second number the client sends: each
transport generation pins the codec it carries. This module is the one place
that mapping lives, so the transport version stays the only negotiated knob.
"""

from __future__ import annotations

from reactor_runtime.protocol import ProtocolVersion

DEFAULT_PROTOCOL_VERSION = ProtocolVersion.V0
"""The codec assumed when a client declares no recognised transport version.

Every shipped client speaks v0, so an absent or unknown transport version
falls back to it and the connection still negotiates.
"""

_TRANSPORT_TO_PROTOCOL: dict[str, ProtocolVersion] = {
    "1.0": ProtocolVersion.V0,
}


def protocol_for_transport(transport_version: str | None) -> ProtocolVersion:
    """Return the wire codec a client's WebRTC transport version pins.

    Args:
        transport_version: The ``Reactor-WebRTC-Version`` declared at the
            handshake, or ``None`` when the client sends none.

    Returns:
        The codec to speak on the connection. An absent or unrecognised
        version falls back to :data:`DEFAULT_PROTOCOL_VERSION`.
    """
    if transport_version is None:
        return DEFAULT_PROTOCOL_VERSION
    return _TRANSPORT_TO_PROTOCOL.get(transport_version, DEFAULT_PROTOCOL_VERSION)
