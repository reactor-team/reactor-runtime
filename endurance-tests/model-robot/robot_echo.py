"""The tracked echo on ReactorRobotApp, so it takes ``observe`` messages."""

from __future__ import annotations

from tracked_echo import TrackedEcho

from reactor_runtime.apps.robot import ReactorRobotApp


class RobotEcho(TrackedEcho, ReactorRobotApp):
    """The tracked echo's streaming, plus the stamped robot state every robot app keeps."""
