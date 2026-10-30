"""Where a session runtime finds its model server: one typed configuration, read in one place.

A deployment tells each session runtime where the model server listens and
which slot the session serves through the environment. :func:`link_config`
is the one function that reads those variables, into a :class:`LinkConfig`;
the client and the runtime's log fields both take that value instead of
reading the environment themselves.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

SOCKET_ENV = "REACTOR_MODEL_WORKER_SOCKET"
"""The model server's socket."""

SLOT_ENV = "REACTOR_SESSION_SLOT"
"""The slot this session runtime serves, ``0`` to the deployment's session count minus one."""

SHM_DIR_ENV = "REACTOR_SHM_DIR"
"""This session's shared-memory directory; ``/dev/shm`` when unset."""

STEP_TIMEOUT_ENV = "REACTOR_WORKER_STEP_TIMEOUT"
"""Seconds a step or a reset waits for its answer by default; 30 when unset, ``0`` for no bound."""

CONNECT_TIMEOUT_ENV = "REACTOR_WORKER_CONNECT_TIMEOUT"
"""Seconds ``connect()`` waits for the model server in all; 1800 when unset, ``0`` for no bound."""

DEFAULT_STEP_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 1800.0
"""Long enough for a model server that loads large weights after a restart."""


@dataclass(frozen=True)
class LinkConfig:
    """How a session runtime reaches its model server.

    Attributes:
        socket: The model server's socket.
        slot: The slot this session runtime serves.
        shm_dir: This session's shared-memory directory, or ``None`` for
            ``/dev/shm``.
        step_timeout: Seconds a step or a reset waits for its answer by
            default, or ``None`` for no bound.
        connect_timeout: Seconds ``connect()`` waits for the model server in
            all, or ``None`` for no bound.
    """

    socket: str
    slot: int
    shm_dir: Path | None = None
    step_timeout: float | None = DEFAULT_STEP_TIMEOUT
    connect_timeout: float | None = DEFAULT_CONNECT_TIMEOUT


def link_config() -> LinkConfig:
    """Read the session runtime's link to its model server from the environment.

    Reads :data:`SOCKET_ENV` and :data:`SLOT_ENV`, which a deployment sets on
    every session runtime of a split model, and :data:`SHM_DIR_ENV`,
    :data:`STEP_TIMEOUT_ENV` and :data:`CONNECT_TIMEOUT_ENV`, which have
    defaults.

    Raises:
        ValueError: A required variable is unset, or a value is malformed,
            naming the variable.
    """
    socket = os.getenv(SOCKET_ENV, "").strip()
    if not socket:
        raise ValueError(
            f"{SOCKET_ENV} is not set: a split model's session runtime needs the path of its "
            "model server's socket"
        )
    raw_dir = os.getenv(SHM_DIR_ENV, "").strip()
    return LinkConfig(
        socket=socket,
        slot=session_slot(required=True) or 0,
        shm_dir=Path(raw_dir) if raw_dir else None,
        step_timeout=_seconds(STEP_TIMEOUT_ENV, DEFAULT_STEP_TIMEOUT),
        connect_timeout=_seconds(CONNECT_TIMEOUT_ENV, DEFAULT_CONNECT_TIMEOUT),
    )


def session_slot(*, required: bool = False) -> int | None:
    """Read :data:`SLOT_ENV`, or return ``None`` when it is unset and not *required*.

    Raises:
        ValueError: It is unset while *required*, or not a non-negative integer.
    """
    raw = os.getenv(SLOT_ENV, "").strip()
    if not raw:
        if required:
            raise ValueError(
                f"{SLOT_ENV} is not set: a split model's session runtime serves a slot"
            )
        return None
    try:
        slot = int(raw)
    except ValueError:
        raise ValueError(f"{SLOT_ENV} {raw!r} must be a non-negative integer") from None
    if slot < 0:
        raise ValueError(f"{SLOT_ENV} {raw!r} must be a non-negative integer")
    return slot


def _seconds(name: str, default: float) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} {raw!r} must be a number of seconds") from None
    if not 0 <= value < math.inf:
        raise ValueError(f"{name} {raw!r} must be a number of seconds, at least 0")
    return None if value == 0 else value
