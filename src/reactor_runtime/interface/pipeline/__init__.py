"""The generator-pattern authoring surface — :class:`ReactorPipeline`.

A way to write a model as a generator: declare a typed :class:`InputState`,
implement an ``inference()`` generator that yields outputs, and let the base
handle the connection lifecycle, session state, and emission pacing. Built on
:class:`reactor_runtime.ReactorModel`; everything that base offers still
applies.

:class:`InputState` is re-exported here because it belongs to both bases; it
lives with the model layer.
"""

from reactor_runtime.interface.model.input_state import InputState
from reactor_runtime.interface.pipeline.idle import Idle
from reactor_runtime.interface.pipeline.reactor_pipeline import ReactorPipeline

__all__ = [
    "Idle",
    "InputState",
    "ReactorPipeline",
]
