"""Media track declarations — the markers and the topology holders.

A model declares its media topology by subclassing :class:`Output` (and
optionally :class:`MediaInput`) with fields annotated as :class:`Video` or
:class:`Audio`::

    class GameOutput(Output):
        main_video: Video
        narration: Audio

    class GameInput(MediaInput):
        camera: Video

Declaring the subclass resolves those annotations into the track records cached
on the class — out for an :class:`Output`, in for an :class:`MediaInput`.

``Input`` is the previous name of :class:`MediaInput`. It still imports, with a
:class:`DeprecationWarning`, and resolves to the same class.
"""

from typing import Any

from reactor_runtime.interface.internal.aliases import deprecated_alias
from reactor_runtime.interface.tracks.descriptors import Audio, Track, Video
from reactor_runtime.interface.tracks.input import INPUT_REGISTRY, MediaInput, all_input_tracks
from reactor_runtime.interface.tracks.output import (
    OUTPUT_REGISTRY,
    Metadata,
    Output,
    TrackPayload,
    all_output_tracks,
)

__all__ = [
    "INPUT_REGISTRY",
    "OUTPUT_REGISTRY",
    "Audio",
    "MediaInput",
    "Metadata",
    "Output",
    "Track",
    "TrackPayload",
    "Video",
    "all_input_tracks",
    "all_output_tracks",
]


def __getattr__(name: str) -> Any:
    if name == "Input":
        return deprecated_alias("Input", "MediaInput", MediaInput)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
