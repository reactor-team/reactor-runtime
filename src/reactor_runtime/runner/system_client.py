"""The system client — the connection the runtime opens to a session itself.

A session started with a starting input or a step count gets one. It is the
sender of the starting commands, so a handler addresses a real client instead
of none, and while it is connected the session has an audience: the model
generates with no other client connected. It takes connection id ``0``, which
no transport mints, and carries no media, so the media fan-out skips it and a
model is never held to a playout rate nobody is watching. Anything sent to it
is dropped.
"""

from __future__ import annotations

from reactor_runtime.core import ConnectionCapabilities, ConnId, MediaChunk
from reactor_runtime.protocol import ProtocolVersion

SYSTEM_CONN_ID = ConnId(0)
"""The connection id the system client takes, outside the range transports mint."""


class SystemConnection:
    """A connection with no wire, conforming to :class:`~reactor_runtime.core.Connection`.

    Every send is a no-op and it advertises no media, so registering it changes
    only what the session knows about its occupancy.
    """

    id = SYSTEM_CONN_ID
    capabilities = ConnectionCapabilities()

    @property
    def protocol_version(self) -> ProtocolVersion:
        """The codec a reply to the system client is encoded in, before it is dropped."""
        return ProtocolVersion.V1

    def send_message(self, payload: bytes | str) -> None:
        """Drop a data frame."""

    def send_media(self, chunk: MediaChunk) -> None:
        """Drop a media chunk; the fan-out never sends one to a connection without media."""

    def flush_media(self) -> None:
        """Do nothing; no media is queued."""

    def set_media_rate(self, fps: float) -> None:
        """Do nothing; no media is paced."""

    def set_media_depth(self, depth: int) -> None:
        """Do nothing; no media is queued."""

    def resume_track(self, name: str) -> None:
        """Do nothing; no track is sent."""

    def pause_track(self, name: str) -> None:
        """Do nothing; no track is sent."""

    def send_control(self, payload: bytes | str) -> None:
        """Drop a control frame."""

    async def close(self) -> None:
        """Do nothing; there is no wire to close."""
