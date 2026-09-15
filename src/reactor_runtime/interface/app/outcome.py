"""What one call to ``generate()`` did, :class:`StepOutcome`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reactor_runtime.interface.tracks import Output


@dataclass(frozen=True)
class StepOutcome:
    """What ``generate()`` did on one step.

    Built by the runtime from ``generate()``'s return value or from the
    exception it raised, and passed to ``collect_step()``. When :attr:`error`
    is set, :attr:`result` is ``None``. A ``None`` :attr:`result` with no
    :attr:`error` is a step that ran and produced nothing. Author code reads it
    and never creates it.

    Attributes:
        result: What ``generate()`` returned. The type is the author's own step
            result type.
        error: What ``generate()`` raised. The type is the model's own exception.
        elapsed: Wall-clock seconds ``generate()`` took, success or failure.
    """

    result: Any = None
    error: Exception | None = None
    elapsed: float = 0.0

    def __post_init__(self) -> None:
        """Reject an outcome that claims both a result and an error.

        Raises:
            ValueError: Both :attr:`result` and :attr:`error` are set.
        """
        if self.result is not None and self.error is not None:
            raise ValueError(
                "StepOutcome holds a result or an error, never both: "
                f"result={type(self.result).__name__}, error={type(self.error).__name__}"
            )

    def to_output(self) -> Output | None:
        """Read the result as track media.

        Only an :class:`Output` qualifies; nothing is wrapped or guessed. A bare
        array, a dataclass, or a tuple is not media until ``collect_step()``
        says which track it goes on. An outcome that holds an error has no
        media to read, so the error is raised: a ``collect_step()`` that calls
        this without checking :attr:`error` first does not swallow the model's
        exception.

        Returns:
            The :class:`Output` the model returned, or ``None`` when the step
            produced nothing to show.

        Raises:
            Exception: The :attr:`error` this outcome holds, when it holds one.
            NotImplementedError: The result is neither an :class:`Output` nor
                ``None``. The message names the type and the two ways out.
        """
        if self.error is not None:
            raise self.error
        if self.result is None:
            return None
        if isinstance(self.result, Output):
            return self.result
        raise NotImplementedError(
            f"generate() returned {type(self.result).__name__}, which is not a compatible "
            "output. Either return an Output subclass from generate(), or override "
            "collect_step() and map outcome.result into one explicitly."
        )
