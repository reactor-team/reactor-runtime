"""The engine protocol — :class:`StreamingPipeline`, :class:`Frames`, :data:`Cache`.

What a runtime calls on an engine, and what it gets back. The pipeline is
stateless about the rollout: everything one continuous sequence needs lives in
the cache, and the caller owns the cache. Structural typing keeps an engine free
of any import from the runtime that drives it — a class that has these four
methods satisfies the protocol by having them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from reactor_runtime.engine_contract.inputs import ModelInput, UserInput

Cache = Any
"""The rollout's memory, owned and shaped entirely by the engine.

Whatever one continuous sequence needs to continue: attention history, encoder
and decoder streaming state, the held controls a fold carries across windows.
Keep it and the model continues the sequence; replace it and a new sequence
starts. A runtime only ever passes it back.
"""


class Frames:
    """The media one step produced, keyed by output track name.

    Constructed by keyword, one entry per track the deployment serves::

        return Frames(main_video=chunk)

    Attributes:
        tracks: Track name to the payload produced for it.
    """

    def __init__(self, **tracks: Any) -> None:
        """Bind one payload per named output track."""
        self.tracks: dict[str, Any] = dict(tracks)

    def __repr__(self) -> str:
        return f"Frames({', '.join(sorted(self.tracks))})"


@runtime_checkable
class StreamingPipeline(Protocol):
    """The four calls a runtime makes to drive an engine.

    ``initialize_cache`` opens a rollout; ``map_inputs`` folds one window of
    client input into one step's conditioning; ``generate`` advances the model
    by exactly one step; ``finalize`` commits that step into the rollout's
    memory. The runtime owns the cache and the step index, so the ordering the
    engine relies on is guaranteed by there being a single driver.
    """

    def initialize_cache(self, **init: Any) -> Cache:
        """Open a rollout and return its memory.

        Args:
            init: The fields of the engine's ``Init``, as the client sent them
                or as the declared defaults supply them.

        Returns:
            The cache the rest of the rollout is driven against.
        """
        ...

    def map_inputs(self, inputs: list[UserInput], cache: Cache) -> ModelInput | None:
        """Fold one window of client input into the next step's conditioning.

        The window holds every input received since the previous step — events
        and media alike — ordered by arrival. Folding is the engine's own
        business: last-value-wins for a held control, summing for a delta,
        resampling where sub-step timing matters. A value that must survive
        into the next window is rollout state and belongs on the cache.

        Args:
            inputs: The window, in arrival order.
            cache: The rollout's memory, readable and writable.

        Returns:
            The conditioning for this step, or ``None`` to skip it — the
            answer a video-to-video model gives when no complete chunk of
            source frames has arrived yet.
        """
        ...

    def generate(self, index: int, cache: Cache, input: Any) -> Frames:
        """Advance the rollout by exactly one step and return its frames.

        Args:
            index: The step's position in the rollout, advancing by one each
                time. Attention and positional state are positional, so a
                skipped or repeated index desynchronizes the cache from the
                frames.
            cache: The rollout's memory.
            input: The conditioning ``map_inputs`` produced.

        Returns:
            The chunk of media this step produced.
        """
        ...

    def finalize(self, index: int, cache: Cache) -> None:
        """Commit the step's result into the rollout's memory.

        Split from :meth:`generate` so the frames can be delivered before the
        cost of the cache update is paid.

        Args:
            index: The step that was generated.
            cache: The rollout's memory.
        """
        ...
