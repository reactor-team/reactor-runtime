"""Media track declarations — the markers and the topology holders.

A model declares its media topology by subclassing :class:`Output` (and
optionally :class:`Input`) with fields annotated as :class:`Video` or
:class:`Audio`::

    class GameOutput(Output):
        main_video: Video
        narration: Audio

    class GameInput(Input):
        camera: Video

Declaring the subclass resolves those annotations into the track records cached
on the class — out for an :class:`Output`, in for an :class:`Input`.
"""

from reactor_runtime.interface.tracks.descriptors import Audio, Track, Video
from reactor_runtime.interface.tracks.input import INPUT_REGISTRY, Input, all_input_tracks
from reactor_runtime.interface.tracks.output import OUTPUT_REGISTRY, Output, all_output_tracks

__all__ = [
    "INPUT_REGISTRY",
    "OUTPUT_REGISTRY",
    "Audio",
    "Input",
    "Output",
    "Track",
    "Video",
    "all_input_tracks",
    "all_output_tracks",
]
