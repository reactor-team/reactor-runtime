"""The HTTP shape of what a saved step delivered: the ``usage`` entries of ``result.json``.

Each kind of output a step can deliver has its own model here, tagged by a
``kind`` literal, and :data:`StepUsage` is the type of one entry. A new kind
adds its model to this module and joins :data:`StepUsage`, which then becomes
a union discriminated on ``kind``, so the route models never change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StepVideoUsage(BaseModel):
    """The measured facts of one video track the step wrote, as the file holds it."""

    kind: Literal["video"]
    track: str = Field(description="The output track the stream holds.")
    file: str = Field(description="The file the stream is in.")
    duration_s: float = Field(description="How long the stream plays, in seconds.")
    frames: int = Field(description="How many frames the stream holds.")
    fps: float = Field(description="The rate the stream plays at, in frames per second.")
    width: int = Field(description="The stream's width in pixels.")
    height: int = Field(description="The stream's height in pixels.")


StepUsage = StepVideoUsage
"""One ``usage`` entry of a step's ``result.json``."""
