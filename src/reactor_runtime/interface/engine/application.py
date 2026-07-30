"""Serving an engine that has no application — :func:`application_for`.

An engine satisfies the contract and nothing else, so there is a deployment
where an application would be one line of boilerplate: bind the engine.
``runtime.import`` in ``reactor.yaml`` may therefore name the engine class
itself, and this builds that application around it.
"""

from __future__ import annotations

from reactor_runtime.engine_contract.pipeline import StreamingPipeline
from reactor_runtime.interface.engine.engine_pipeline import EnginePipeline


def is_engine(candidate: object) -> bool:
    """Return whether *candidate* is a class satisfying the engine protocol."""
    return isinstance(candidate, type) and isinstance(candidate, StreamingPipeline)


def application_for(engine_cls: type) -> type[EnginePipeline]:
    """Build the application that serves *engine_cls*.

    Args:
        engine_cls: The engine pipeline class to serve.

    Returns:
        An :class:`EnginePipeline` subclass bound to the engine. The engine's
        own name carries into the published schema, so a client sees the model
        it asked for.
    """
    return type(
        engine_cls.__name__,
        (EnginePipeline,),
        {
            "__doc__": engine_cls.__doc__,
            "__module__": engine_cls.__module__,
            "engine": engine_cls,
        },
    )
