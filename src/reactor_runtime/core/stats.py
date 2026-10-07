"""Readings a transport takes of its own wire.

A transport that measures its wire describes what it measured in these types,
which follow a client's own stats reading (:class:`~reactor_runtime.core.ClientTrackStat`):
what is fixed about a track stays a field, and the measurements go in an open
``metrics`` mapping. Each transport names what it measures, so a new transport
reports its own measurements without a change here, and a reader holds the same
shape whichever transport took the reading.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from reactor_runtime.core.values import ClientTrackDirection, TrackKind


@dataclass(frozen=True)
class TransportTrackReading:
    """What a transport measured of one track on one connection.

    Attributes:
        track_name: The track the reading is for.
        kind: The track's kind; ``None`` when the transport does not know it.
        direction: The track's direction as the client sees it: a model
            output track is one the client receives.
        codec: The negotiated codec, e.g. ``"VP9"``, ``"opus"``; empty when
            the transport has not reported one.
        metrics: The measurements, by name. Names match a client reading's
            where both measure the same thing (``"bitrate_bps"``,
            ``"packets_lost"``, ``"frames_per_second"``), and a measurement the
            transport has no value for is left out rather than reported as
            zero.
    """

    track_name: str
    kind: TrackKind | None
    direction: ClientTrackDirection
    codec: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportReading:
    """What a transport measured of one connection's wire at one moment.

    Attributes:
        metrics: Measurements of the connection as a whole, by name, e.g.
            ``"connection_rtt_ms"`` or ``"available_outgoing_bitrate_bps"``.
        tracks: One reading per track the transport measured.
    """

    metrics: Mapping[str, float] = field(default_factory=dict)
    tracks: tuple[TransportTrackReading, ...] = ()
