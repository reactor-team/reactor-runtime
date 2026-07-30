"""The engine protocol — :class:`StreamingPipeline`, :data:`VideoChunk`, :data:`Cache`.

What a runtime calls on an engine, and what it gets back. The pipeline is
stateless about the rollout: everything one continuous sequence needs lives in
the cache, and the caller owns the cache. Structural typing keeps an engine free
of any import from the runtime that drives it — a class that has these methods
satisfies the protocol by having them.

The shape is the one interactive autoregressive engines already have —
``initialize_cache`` / ``generate`` / ``finalize``, sized by
``get_num_output_frames``. ``map_inputs`` is the one addition, and it is the
whole point: it is where a window of client input becomes one step's
conditioning, the piece an engine has never had to name before.
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

VideoChunk = Any
"""Decoded video for one step, as the engine's own tensor library produces it.

Deliberately untyped: naming a tensor type here would put a deep-learning
framework in the contract's imports. A runtime takes what these engines already
emit — a device or host tensor, or a NumPy array, laid out ``[T, C, H, W]`` or
``[T, H, W, C]`` (a single frame may drop the leading axis), floating point in
the engine's own value range or ``uint8`` in ``0-255`` — and normalizes it for
the wire.
"""


@runtime_checkable
class StreamingPipeline(Protocol):
    """The calls a runtime makes to drive an engine.

    ``initialize_cache`` opens a rollout; ``map_inputs`` folds one window of
    client input into one step's conditioning; ``generate`` advances the model
    by exactly one step; ``finalize`` commits that step into the rollout's
    memory. The runtime owns the cache and the step index, so the ordering the
    engine relies on is guaranteed by there being a single driver, and it calls
    every one of these by keyword.
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

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        """Return how many frames the step at this index will produce.

        The count is not always constant: a first step that seeds from an
        initial frame commonly emits fewer than the steps after it. The mapping
        reads this to size a per-frame payload — a camera trajectory, a control
        curve — so the conditioning and the frames it produces agree.

        Args:
            autoregressive_index: The step about to be generated.

        Returns:
            The number of frames that step emits.
        """
        ...

    def map_inputs(
        self, autoregressive_index: int, cache: Cache, inputs: list[UserInput]
    ) -> ModelInput | None:
        """Fold one window of client input into the next step's conditioning.

        The window holds every input received since the previous step — events
        and media alike — ordered by arrival. Folding is the engine's own
        business: last-value-wins for a held control, summing for a delta,
        resampling where sub-step timing matters. A value that must survive into
        the next window is rollout state and belongs on the cache.

        Args:
            autoregressive_index: The step this conditioning is for, so a
                per-frame payload can be sized against
                :meth:`get_num_output_frames`.
            cache: The rollout's memory, readable and writable.
            inputs: The window, in arrival order.

        Returns:
            The conditioning for this step, or ``None`` to skip it — the answer
            a video-to-video model gives when no complete chunk of source frames
            has arrived yet, and the answer to a fresh initialization arriving
            mid-rollout.
        """
        ...

    def generate(
        self,
        autoregressive_index: int,
        cache: Cache,
        input: Any = None,
    ) -> VideoChunk:
        """Advance the rollout by exactly one step and return its frames.

        Args:
            autoregressive_index: The step's position in the rollout, advancing
                by one each time. Attention and positional state are positional,
                so a skipped or repeated index desynchronizes the cache from the
                frames.
            cache: The rollout's memory.
            input: The conditioning ``map_inputs`` produced.

        Returns:
            The decoded video this step produced.
        """
        ...

    def finalize(self, autoregressive_index: int, cache: Cache) -> dict[str, float] | None:
        """Commit the step's result into the rollout's memory.

        Split from :meth:`generate` so the frames can be delivered before the
        cost of the cache update is paid.

        Args:
            autoregressive_index: The step that was generated.
            cache: The rollout's memory.

        Returns:
            Per-step timings, when the engine measures them. A runtime does not
            require them and does not read them.
        """
        ...
