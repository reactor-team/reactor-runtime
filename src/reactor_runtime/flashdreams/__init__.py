"""Serve a FlashDreams model by naming it.

A FlashDreams model is a ``StreamInferencePipeline``: ``initialize_cache()``
starts a rollout, and ``generate()`` then ``finalize()`` run once per step. One
pipeline step is one Reactor ``generate()``. Models join a family (``action2v``,
``t2v``, ``cam2v``, ``v2v``) through an adapter registered under a slug, and the
adapter holds every per-model fact: the pipeline, frame size and rate, the seed
loader, the key mapping.

This package is the runtime's side of that seam. :mod:`.contract` is what the
two halves of a model exchange, :mod:`.model` is the generic model half every
family builds on, and :mod:`.app` is the generic application half. FlashDreams
is not a dependency of the runtime: a workspace installs it, and the modules
here import it only when a model loads. Like :mod:`reactor_runtime.distributed`,
nothing else in the runtime imports this package.
"""

from reactor_runtime.flashdreams.app import (
    FlashDreamsApp,
    FlashDreamsOutput,
    FlashDreamsState,
    RolloutRestarted,
)
from reactor_runtime.flashdreams.contract import (
    FlashDreamsResult,
    PromptSwapUnsupported,
    RolloutExhausted,
    RolloutNotStarted,
)
from reactor_runtime.flashdreams.model import FlashDreamsModel

__all__ = [
    "FlashDreamsApp",
    "FlashDreamsModel",
    "FlashDreamsOutput",
    "FlashDreamsResult",
    "FlashDreamsState",
    "PromptSwapUnsupported",
    "RolloutExhausted",
    "RolloutNotStarted",
    "RolloutRestarted",
]
