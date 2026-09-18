"""Deprecated import names and the warning they emit.

A renamed class stays importable under its old name for one major version.
The old name is served by a module-level ``__getattr__``, so it costs nothing
until a caller asks for it.

The resolution is reported twice, to two audiences. A :class:`DeprecationWarning`
is for tooling: test runners and linters surface it, and ``-W error`` turns it
into a failure. Python hides that warning by default outside ``__main__``, and
a model is imported by the runtime from a manifest, so an author would never see
it in a running container. The first resolution of each old name therefore
also writes one warning record to the log, which is what an author reads.
"""

from __future__ import annotations

import warnings

from reactor_runtime.log import get_logger

logger = get_logger(__name__)

_reported: set[str] = set()


PIPELINE_DEPRECATION = (
    "ReactorPipeline is deprecated. Subclass ReactorApp and write generate(); "
    "the class is removed in the next major."
)
IDLE_DEPRECATION = (
    "Idle is deprecated. Refuse a step by raising ApplicationError from process_input(); "
    "the name is removed in the next major."
)


def deprecated_alias(old: str, new: str, target: object, message: str | None = None) -> object:
    """Return *target*, warn that *old* is deprecated in favour of *new*, and log it once.

    Args:
        old: The name the caller imported.
        new: The name that replaces it.
        target: The object the old name resolves to.
        message: The warning text. The default says *old* is now *new* and
            fits a rename; a surface whose replacement is a port rather than
            an import edit passes its own.

    Returns:
        *target*, unchanged.
    """
    if message is None:
        message = f"{old} is now {new}. Update the import; the alias is removed in the next major."
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    if old not in _reported:
        _reported.add(old)
        logger.warning("deprecated name imported", old=old, new=new)
    return target
