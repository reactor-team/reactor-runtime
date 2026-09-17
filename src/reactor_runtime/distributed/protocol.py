"""The messages between a runner and its ranks.

Requests go from the runner to every rank, in the same order on every rank's
queue. Every rank answers every request. A large payload never rides on a
queue: :class:`Generate` and :class:`Answer` carry a header from
:func:`~reactor_runtime.distributed.ipc.pack`, and the bytes sit in a shared
block the header names.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Generate:
    """Run one ``generate()`` on every rank with the input the header describes."""

    header: bytes


@dataclass(frozen=True)
class Reset:
    """Call ``reset()`` on every rank."""


@dataclass(frozen=True)
class Shutdown:
    """Leave the request loop, run the exit barrier, and end the process."""


@dataclass(frozen=True)
class Loaded:
    """A rank finished ``load()`` and is ready for requests."""

    rank: int


@dataclass(frozen=True)
class Answer:
    """One rank's reply to a :class:`Generate` or :class:`Reset`.

    Attributes:
        rank: The answering rank.
        header: Rank 0's packed result on a successful ``Generate``. Other
            ranks and every ``Reset`` send ``None``.
        error: What the worker raised, or ``None`` on success.
    """

    rank: int
    header: bytes | None = None
    error: Exception | None = None
