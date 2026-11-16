"""Step results: each finished step kept as a folder of files.

A step result is not a recording. It has no timeline and no seeking, and it is
written once, when the model reports the step done: the step's video in one
``output.mp4``, encoded from the step's frames or, for a step with none, the
model's own file kept as it is; any extra files the model added; and a
``result.json`` that lists them, with the measured facts of each video track
the runtime encoded.
"""

from reactor_runtime.step_results.mp4 import write_mp4
from reactor_runtime.step_results.store import SavedStep, StepStore, content_type_of
from reactor_runtime.step_results.usage import VideoUsage

__all__ = ["SavedStep", "StepStore", "VideoUsage", "content_type_of", "write_mp4"]
