"""The stepped-model vocabulary — :class:`SteppedModel`, :class:`StepStats`, :class:`NotReady`.

Three names an author writes against when the runtime drives the model rather
than the author owning the loop:

- :class:`SteppedModel` is the shape the model itself satisfies — one call per
  step, structurally checked, with no import of this package required.
- :class:`StepStats` is what the runtime measured about a step, handed to the
  application on the way out.
- :class:`NotReady` is how the application declines a step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class NotReady(Exception):  # noqa: N818 — the name a model author raises, not an error suffix
    """Raised to decline a step, carrying the reason as its message.

    ``map_step`` raises it when there is nothing to step on yet: input has not
    arrived, or a value the model needs is unset. The driver holds the stream,
    logs the reason, and tries again — a decline is an ordinary, frequent event
    rather than a fault.

    Sibling of :class:`~reactor_runtime.BufferClosed`, which the same
    ``try_read`` call raises: a decline continues the generation cycle, a closed
    track ends it.
    """


@dataclass(frozen=True)
class StepStats:
    """What the runtime measured about one step.

    Built by the runtime and handed to the application, never constructed by a
    model. ``to_output`` receives it by declaring a ``stats`` parameter, and
    ``on_step`` always receives it.

    Attributes:
        step: How many steps the runtime has driven, counting from zero and
            counting on. It is the runtime's own tally and nothing resets it:
            not a session ending, not a client leaving, not a reset the model
            performed. A model that has a rollout of its own numbers it itself
            and reports that number as one of the values it produces, because
            what restarts a rollout is the model's business rather than the
            runtime's.
        compute_time: Wall-clock seconds the model's ``generate`` call took.
    """

    step: int
    compute_time: float


@runtime_checkable
class SteppedModel(Protocol):
    """A model that produces its output one step at a time.

    Structural: a model satisfies this by having the method, without importing
    or subclassing anything from this package. That is the point — the model
    holds the weights, the caches, and every value that persists between steps,
    and nothing in it knows that Reactor exists.

    Bind one by annotating ``model:`` on a
    :class:`~reactor_runtime.ReactorModel`, and the runtime drives it: the
    application's ``map_step`` builds the arguments, ``generate`` produces, and
    ``to_output`` places the products on the wire.

    A model may also define ``load(config)``, which the runtime calls once
    before any client connects when the model is bound as a class-level
    default. It is deliberately absent from this protocol, so a model that
    loads itself in the application's ``load()`` still satisfies it.

    Anything else the model offers — ``reset()``, ``stats()``, a snapshot — is
    its own API, called from the application's ``@event`` handlers.
    """

    def generate(self, **inputs: Any) -> Any:
        """Produce one step's output, or ``None`` to produce nothing this step.

        Called repeatedly, once per step. Whatever the application's
        ``map_step`` returned arrives here as keyword arguments, including any
        conditioning that applies to the whole rollout: noticing that
        conditioning changed — and deciding what that costs — is this method's
        job, because everything that persists between calls is its own private
        state.

        Synchronous on purpose: a forward pass is synchronous work, and the
        runtime measures the call to pace playout from it.

        Args:
            inputs: The arguments the application's ``map_step`` returned.

        Returns:
            A mapping, which spreads into ``to_output`` as keyword arguments;
            any other value, which arrives there as one positional argument; or
            ``None`` for a step that produced nothing.
        """
        ...
