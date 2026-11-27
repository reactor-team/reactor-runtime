"""Robot policies on the runtime: :class:`ReactorRobotApp` and its step types.

A robot policy subclasses :class:`ReactorRobotApp`, writes ``load()`` and
``generate(obs)``, and returns :class:`Actions` from an :class:`Observation`:
the camera frames and the robot state of one capture, joined by their capture
stamp.
"""

from reactor_runtime.apps.robot.app import ReactorRobotApp
from reactor_runtime.apps.robot.types import Actions, Observation

__all__ = ["Actions", "Observation", "ReactorRobotApp"]
