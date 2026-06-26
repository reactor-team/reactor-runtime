"""Logging helper for the ported GStreamer media engine.

The media path logs through the standard library so the port carries no
dependency on a particular logging stack.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return the standard-library logger for ``name``."""
    return logging.getLogger(name)
