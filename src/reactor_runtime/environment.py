"""Read the runtime's settings from the environment: the helpers every entry point shares.

The runtime's entry points (``python -m reactor_runtime.serve`` and
``python -m reactor_runtime.model_server``) read the manifest for what the
model declares and the environment for what the deployment decides. These
helpers parse one variable each, with one rule for an unset value (the default)
and one for a malformed one (the process refuses to start, naming it).
"""

from __future__ import annotations

import logging
import math
import os


def float_env(name: str, default: float) -> float:
    """Return env var *name* as a float, or *default* when unset/empty.

    Raises:
        SystemExit: If the value is set but not a number.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} {raw!r} must be a number") from None


def duration_env(name: str, default: float) -> float:
    """Return env var *name* as a finite, non-negative number of seconds.

    For a window that arms a timer. ``float`` also parses ``inf`` and ``nan``,
    which a timer cannot hold — an infinite deadline never fires and a NaN or
    negative one fires at once — so those are refused at boot rather than
    left to disable or trip the guarantee the window exists for.

    Raises:
        SystemExit: If the value is set but not a finite number of at least zero.
    """
    value = float_env(name, default)
    if not math.isfinite(value) or value < 0:
        raise SystemExit(f"{name} {os.getenv(name, '').strip()!r} must be a finite number >= 0")
    return value


def int_env(name: str, default: int) -> int:
    """Return env var *name* as an int, or *default* when unset/empty.

    Raises:
        SystemExit: If the value is set but not an integer.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} {raw!r} must be an integer") from None


def log_level_from_env() -> int:
    """Resolve ``REACTOR_LOG_LEVEL`` to a logging level, defaulting to ``INFO``."""
    name = os.getenv("REACTOR_LOG_LEVEL", "INFO").strip().upper()
    return logging.getLevelNamesMapping().get(name, logging.INFO)
