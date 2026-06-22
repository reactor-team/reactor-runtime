"""WebRTC connection configuration.

The tunables the acceptor threads into every connection it builds: the ICE
servers and policy that shape candidate gathering, the UDP port range, and the
liveness timeout the connection's ping watchdog enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IceTransportPolicy(StrEnum):
    """Which ICE candidate types to gather.

    ``ALL`` gathers host, server-reflexive, and relay candidates; ``RELAY``
    gathers only relay (TURN) candidates, forcing every flow through a TURN
    server.
    """

    ALL = "all"
    RELAY = "relay"


@dataclass(frozen=True)
class IceServer:
    """One STUN or TURN server offered to the peer for candidate gathering.

    Attributes:
        urls: One or more STUN/TURN URLs for this server.
        username: Username for TURN authentication, or ``None`` for STUN.
        credential: Credential for TURN authentication, or ``None`` for STUN.
    """

    urls: tuple[str, ...]
    username: str | None = None
    credential: str | None = None


@dataclass(frozen=True)
class WebRtcConfig:
    """Configuration the acceptor applies to every connection it negotiates.

    Attributes:
        ice_servers: The STUN/TURN servers offered for candidate gathering.
        port_range: An inclusive ``(min, max)`` UDP port range to confine ICE
            to, or ``None`` to let the stack choose.
        transport_policy: Which candidate types to gather.
        ping_timeout: Seconds without a client ping before the connection's
            watchdog declares it lost. ``0`` or less disables the watchdog.
    """

    ice_servers: tuple[IceServer, ...] = ()
    port_range: tuple[int, int] | None = None
    transport_policy: IceTransportPolicy = IceTransportPolicy.ALL
    ping_timeout: float = 20.0
