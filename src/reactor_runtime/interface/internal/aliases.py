"""Deprecated import names and the warning they emit.

A renamed class stays importable under its old name for one major version.
The old name is served by a module-level ``__getattr__``, so it costs nothing
until a caller asks for it and warns exactly once per import site.
"""

from __future__ import annotations

import warnings


def deprecated_alias(old: str, new: str, target: object) -> object:
    """Return *target* and warn that *old* is now *new*.

    Args:
        old: The name the caller imported.
        new: The name that replaces it.
        target: The object the old name resolves to.

    Returns:
        *target*, unchanged.
    """
    warnings.warn(
        f"{old} is now {new}. Update the import; the alias is removed in the next major.",
        DeprecationWarning,
        stacklevel=3,
    )
    return target
