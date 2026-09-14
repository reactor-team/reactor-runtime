"""What one call to ``generate()`` did, :class:`StepOutcome`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reactor_runtime.interface.tracks import Output


@dataclass(frozen=True)
class StepOutcome:
    """What ``generate()`` did on one step.

    Built by the runtime from ``generate()``'s return value or from the
    exception it raised, and passed to ``collect_step()``. Exactly one of
    :attr:`result` and :attr:`error` is set. Author code reads it and never
    creates it.

    Attributes:
        result: What ``generate()`` returned. The type is the author's own step
            result type.
        error: What ``generate()`` raised. The type is the model's own exception.
        elapsed: Wall-clock seconds ``generate()`` took, success or failure.
    """

    result: Any = None
    error: Exception | None = None
    elapsed: float = 0.0

    def to_output(self) -> Output | None:
        """Read the result as track media.

        Only an :class:`Output` qualifies; nothing is wrapped or guessed. A bare
        array, a dataclass, or a tuple is not media until ``collect_step()``
        says which track it goes on.

        Returns:
            The :class:`Output` the model returned, or ``None`` when the step
            produced nothing to show.

        Raises:
            NotImplementedError: The result is neither an :class:`Output` nor
                ``None``. The message names the type and the two ways out.
        """
        if self.result is None:
            return None
        if isinstance(self.result, Output):
            return self.result
        raise NotImplementedError(
            f"generate() returned {type(self.result).__name__}, which is not a compatible "
            "output. Either return an Output subclass from generate(), or override "
            "collect_step() and map outcome.result into one explicitly."
        )
