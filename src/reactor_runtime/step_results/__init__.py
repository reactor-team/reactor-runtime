"""Step results: each finished step kept as a folder of files.

A step result is not a recording. It has no timeline and no seeking, and it is
written once, when the model reports the step done: the step's output in one
``output.mp4``, any extra files the model added, and a ``result.json`` that
lists them.
"""

from reactor_runtime.step_results.mp4 import write_mp4

__all__ = ["write_mp4"]
