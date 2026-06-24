"""Locating a model's weights on disk.

A model author calls :func:`get_weights_path` with no arguments to find the
directory its weights live under, and the same call resolves correctly from
local development through to a deployed container. Locally it falls back to a
shared cache directory; in a deployed container the runtime sets
``REACTOR_WEIGHTS_PATH`` to the model's mounted bundle, so model code never
hard-codes a path.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_REACTOR_WEIGHTS_PATH = "REACTOR_WEIGHTS_PATH"
"""Environment variable that overrides the weights root."""

DEFAULT_WEIGHTS_PATH = "~/.cache/reactor_registry"
"""Weights root used when the environment variable is unset or empty."""


def get_weights_path() -> Path:
    """Return the directory under which this model's weights live.

    Resolution order:

    1. ``$REACTOR_WEIGHTS_PATH`` when set and non-empty.
    2. :data:`DEFAULT_WEIGHTS_PATH` (``~/.cache/reactor_registry``).

    The returned path has ``~`` expanded but is not required to exist on disk;
    resolving and validating subpaths under it is the caller's responsibility.

    Returns:
        The weights root as a :class:`~pathlib.Path`.
    """
    raw = os.environ.get(ENV_REACTOR_WEIGHTS_PATH) or DEFAULT_WEIGHTS_PATH
    return Path(os.path.expanduser(raw))
