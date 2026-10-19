"""The LiveKit transport: a session connects to one LiveKit room.

A caller opens a room connection through the transport's routes. The room
connection is a system connection of the session and carries its output
tracks. The router is importable without the LiveKit SDK; the room modules
need the runtime's ``livekit`` extra.
"""

from reactor_runtime.transport.livekit.router import LiveKitRouter, livekit_available

__all__ = ["LiveKitRouter", "livekit_available"]
