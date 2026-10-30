"""What a deployment tells a split model's processes: typed configuration, read in one place.

A deployment tells each session runtime where the model server listens and
which slot the session serves, and tells the model server how many sessions it
serves and where their shared memory is, through the environment.
:func:`link_config` and :func:`server_process_config` are the functions that
read those variables, into a :class:`LinkConfig` and a
:class:`ServerProcessConfig`; nothing else in the package reads the
environment.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from reactor_runtime.environment import duration_env, int_env

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


MAX_SESSIONS_ENV = "REACTOR_MAX_SESSIONS"
"""How many session runtimes the deployment runs beside the model server; required."""

SLOT_DIR_ENV = "REACTOR_SHM_SLOT_DIR"
"""Each slot's shared memory as the model server sees it, a path with ``{slot}``; required."""

HEALTH_PORT_ENV = "REACTOR_MODEL_WORKER_HEALTH_PORT"
"""The port the model server answers ``GET /health`` on; 8090 when unset."""

STUCK_ENV = "REACTOR_MODEL_WORKER_STUCK_SECONDS"
"""Seconds one step may run before the model server reports itself stuck; 120 when unset."""

DEFAULT_SOCKET = "/run/reactor-worker/sock"
DEFAULT_HEALTH_PORT = 8090
DEFAULT_STUCK_SECONDS = 120.0


@dataclass(frozen=True)
class ServerProcessConfig:
    """What the deployment tells the model server's process.

    Attributes:
        socket: Where session runtimes connect.
        max_sessions: How many session runtimes the deployment runs; slots
            ``0`` to ``max_sessions - 1`` are served.
        slot_directory: Each slot's shared memory, a path with ``{slot}``.
        host: The address ``/health`` binds.
        health_port: The port ``/health`` binds.
        stuck_after: Seconds one step may run before ``/health`` reports the
            model server stuck.
        exit_timeout: Seconds the orderly shutdown may take.
        grace_period: Seconds a draining step is given to finish.
    """

    socket: str
    max_sessions: int
    slot_directory: str
    host: str = "0.0.0.0"
    health_port: int = DEFAULT_HEALTH_PORT
    stuck_after: float = DEFAULT_STUCK_SECONDS
    exit_timeout: float = 10.0
    grace_period: float = 30.0


def server_process_config() -> ServerProcessConfig:
    """Read the model server's process settings from the environment.

    Reads :data:`MAX_SESSIONS_ENV` and :data:`SLOT_DIR_ENV`, which a deployment
    must set, and :data:`SOCKET_ENV`, ``HOST``, :data:`HEALTH_PORT_ENV`,
    :data:`STUCK_ENV`, ``EXIT_TIMEOUT_SECONDS`` and ``SIGTERM_GRACE_PERIOD``,
    which have defaults.

    Raises:
        SystemExit: A required variable is unset, or a value is malformed,
            naming the variable.
    """
    max_sessions = int_env(MAX_SESSIONS_ENV, 0)
    if max_sessions < 1:
        raise SystemExit(
            f"{MAX_SESSIONS_ENV} must be set to the deployment's session count, at least 1"
        )
    slot_directory = os.getenv(SLOT_DIR_ENV, "").strip()
    if "{slot}" not in slot_directory:
        raise SystemExit(
            f"{SLOT_DIR_ENV} must be set to each slot's shared-memory directory, a path with "
            "{slot} in it, such as /run/reactor-shm/s{slot}"
        )
    stuck_after = duration_env(STUCK_ENV, DEFAULT_STUCK_SECONDS)
    if stuck_after <= 0:
        raise SystemExit(f"{STUCK_ENV} must be above 0")
    return ServerProcessConfig(
        socket=os.getenv(SOCKET_ENV, "").strip() or DEFAULT_SOCKET,
        max_sessions=max_sessions,
        slot_directory=slot_directory,
        host=os.getenv("HOST", "").strip() or "0.0.0.0",
        health_port=int_env(HEALTH_PORT_ENV, DEFAULT_HEALTH_PORT),
        stuck_after=stuck_after,
        exit_timeout=duration_env("EXIT_TIMEOUT_SECONDS", 10.0),
        grace_period=duration_env("SIGTERM_GRACE_PERIOD", 30.0),
    )
