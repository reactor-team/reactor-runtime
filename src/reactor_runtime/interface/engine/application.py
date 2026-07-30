"""Serving an engine that has no application — :func:`application_for`.

An engine satisfies the contract and nothing else, so there is a deployment
where an application would be two lines of boilerplate: bind the engine, declare
one video track. ``runtime.import`` in ``reactor.yaml`` may therefore name the
engine class itself, and this builds that application around it.

An engine served this way gets the default outbound topology — one video track.
A deployment that needs anything else writes the application and declares its
own :class:`Output`.
"""

from __future__ import annotations

from reactor_runtime.engine_contract.pipeline import StreamingPipeline
from reactor_runtime.interface.engine.engine_pipeline import EnginePipeline
from reactor_runtime.interface.tracks import Output, Video

DEFAULT_VIDEO_TRACK = "main_video"
"""The outbound track an engine served without an application emits on."""


def is_engine(candidate: object) -> bool:
    """Return whether *candidate* is a class satisfying the engine protocol."""
    return isinstance(candidate, type) and isinstance(candidate, StreamingPipeline)


def application_for(engine_cls: type) -> type[EnginePipeline]:
    """Build the application that serves *engine_cls* with the default topology.

    Args:
        engine_cls: The engine pipeline class to serve.

    Returns:
        An :class:`EnginePipeline` subclass bound to the engine, emitting on the
        default video track. The engine's own name carries into the published
        schema, so a client sees the model it asked for.
    """
    output_cls = type(
        f"{engine_cls.__name__}Output",
        (Output,),
        {"__annotations__": {DEFAULT_VIDEO_TRACK: Video}, "__module__": engine_cls.__module__},
    )
    return type(
        engine_cls.__name__,
        (EnginePipeline,),
        {
            "__doc__": engine_cls.__doc__,
            "__module__": engine_cls.__module__,
            "__annotations__": {"output": output_cls},
            "engine": engine_cls,
        },
    )
