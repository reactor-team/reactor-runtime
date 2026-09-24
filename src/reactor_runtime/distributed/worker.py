"""An optional base class that spells out a worker's shape.

A worker is one GPU's worth of a model: a class with ``load()``,
``generate()``, and ``reset()``. The runner constructs it in its own process,
sets ``rank``, ``world_size``, and ``device`` on the instance, calls ``load()``
once, and then ``generate()`` per call and ``reset()`` when asked. Every rank
runs every call; only rank 0's result reaches the caller.

Nothing checks ancestry. A class that has the three methods is accepted;
:class:`DistributedWorker` only names the injected attributes with types and
adds :attr:`~DistributedWorker.is_leader`.
"""

from __future__ import annotations

from typing import Any


class DistributedWorker:
    """Optional base for a worker the runner constructs in its own process.

    Attributes:
        rank: This process's rank, ``0`` to ``world_size - 1``.
        world_size: How many ranks the runner started.
        device: ``"cuda:<rank>"`` when CUDA is present, else ``"cpu"``. Set
            before ``load()`` runs, as are the other two.
    """

    rank: int
    world_size: int
    device: str

    @property
    def is_leader(self) -> bool:
        """Whether this is rank 0, the rank whose result the runner returns."""
        return self.rank == 0

    def load(self, **kwargs: Any) -> None:
        """Put the weights on ``self.device``. Receives the runner's ``load_kwargs``."""
        raise NotImplementedError

    def generate(self, input: Any, /) -> Any:
        """Run one step on this rank and return its result as CPU data.

        Every rank runs this with the same input, in the same order. Only
        rank 0's return value crosses back to the caller. From the other ranks
        the runner receives success or the exception they raised, and nothing
        they return reaches the caller.

        A model whose ranks each produce a different piece of the output
        gathers the pieces onto rank 0 inside this method, with a
        ``torch.distributed`` collective such as ``gather`` or ``all_gather``,
        so that rank 0 returns the whole result. A model whose ranks all end
        up with the full result needs no gather.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Drop per-session state and keep the weights."""
        raise NotImplementedError


def check_shape(worker_cls: type) -> None:
    """Raise ``TypeError`` naming each of the three methods *worker_cls* lacks."""
    missing = [
        name
        for name in ("load", "generate", "reset")
        if not callable(getattr(worker_cls, name, None))
    ]
    if missing:
        raise TypeError(
            f"{worker_cls.__name__} is missing {', '.join(missing)}; "
            "a worker needs load(), generate(), and reset()"
        )
