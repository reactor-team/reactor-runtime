"""Logging helper for the ported GStreamer media engine.

The media engine logs through the runtime's structured logger, so its lines
render in the same ``key=value`` / JSON shape as the rest of the runtime and
call sites can attach structured fields as keyword arguments.
"""

from __future__ import annotations

from reactor_runtime.log import StructuredLogger, get_logger as _get_logger


def get_logger(name: str) -> StructuredLogger:
    """Return the runtime's structured logger for ``name``."""
    return _get_logger(name)
