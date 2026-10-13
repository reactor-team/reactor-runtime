"""What a saved step delivered, measured from the files it wrote, :class:`VideoUsage`.

Each saved step lists the measured facts of what it wrote beside the files
themselves, so a consumer of step results can account for what every step
delivered without decoding its files. Each kind of output has its own class,
and its JSON form names its ``kind``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class VideoUsage:
    """The measured facts of one video track that a saved step wrote.

    One entry describes one video stream of the step's ``output.mp4``, as the
    file holds it.

    Attributes:
        track: The name of the output track the stream holds.
        file: The name of the file the stream is in.
        duration_s: How long the stream plays, in seconds: ``frames / fps``.
        frames: How many frames the stream holds.
        fps: The rate the stream plays at, in frames per second. It is the
            model's declared ``fps`` when the model pins one, and otherwise the
            rate the step's frames were emitted at.
        width: The stream's width in pixels. An odd width is padded by one.
        height: The stream's height in pixels. An odd height is padded by one.
    """

    kind: ClassVar[str] = "video"

    track: str
    file: str
    duration_s: float
    frames: int
    fps: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Return the entry in its JSON form, tagged with its ``kind``."""
        return {"kind": self.kind, **asdict(self)}
