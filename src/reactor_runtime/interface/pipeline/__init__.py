"""The generator-pattern authoring surface — :class:`ReactorPipeline`.

A simplified way to write a model: declare a typed :class:`InputState`,
implement an ``inference()`` generator that yields outputs, and let the base
handle the connection lifecycle, per-connection state, and emission pacing.
Built on :class:`reactor_runtime.ReactorModel`; everything that base offers
still applies.
"""

from reactor_runtime.interface.pipeline.idle import Idle
from reactor_runtime.interface.pipeline.input_state import InputState
from reactor_runtime.interface.pipeline.reactor_pipeline import ReactorPipeline

__all__ = [
    "Idle",
    "InputState",
    "ReactorPipeline",
]
