"""The tracked echo on ReactorRobotApp: it echoes the webcam frame of each joined capture."""

from __future__ import annotations

from typing import cast

from echo_model import EchoInput
from tracked_echo import TrackedEcho

from echo import _drain
from reactor_runtime.apps.robot import Observation, ReactorRobotApp
from reactor_runtime.interface.internal.input_buffer import InputBuffer


class RobotEcho(TrackedEcho, ReactorRobotApp):
    """The tracked echo's streaming, fed by the robot app's synchronizer.

    The webcam is a camera: its reader takes every frame, and a step runs on
    the newest capture whose webcam frame and ``observe`` state share a stamp.
    The model then echoes that frame as the tracked echo does.
    """

    async def process_input(self) -> EchoInput:
        """Take the newest joined capture and hand its webcam frame to the echo."""
        observation: Observation = await ReactorRobotApp.process_input(self)
        self.state._audio = _drain(cast(InputBuffer, self.media.mic))
        self.state._metadata = None
        return EchoInput(
            frame=observation.frames["webcam"],
            effect=self.state.effect,
            intensity=self.state.intensity,
            caption=self.state.caption,
        )
