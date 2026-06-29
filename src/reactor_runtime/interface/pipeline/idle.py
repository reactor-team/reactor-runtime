"""The :data:`Idle` sentinel a pipeline yields to skip a frame.

An ``inference()`` generator yields :data:`Idle` (or ``None``) when it has no
new output this turn — waiting on a ``start`` command, paused, or otherwise
between work. The driver treats the turn as a no-op: it emits nothing and lets
the emission buffer hold the stream live, rather than advancing the output.
"""

from __future__ import annotations


class _IdleType:
    """The type of the :data:`Idle` sentinel — a single shared instance.

    Falsy, so ``if not result`` reads naturally, and reachable by identity
    (``result is Idle``). Constructing it always returns the one instance.
    """

    _instance: _IdleType | None = None

    def __new__(cls) -> _IdleType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Idle"

    def __bool__(self) -> bool:
        return False


Idle: _IdleType = _IdleType()
"""Yield this (or ``None``) from ``inference()`` to skip a frame."""
