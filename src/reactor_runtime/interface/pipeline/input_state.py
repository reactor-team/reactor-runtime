"""The previous module path of :class:`InputState`.

The class lives in :mod:`reactor_runtime.interface.app.input_state` and is
exported from :mod:`reactor_runtime`. This module keeps
``from reactor_runtime.interface.pipeline.input_state import InputState``
importable; the name is unchanged and resolves to the same class.
"""

from reactor_runtime.interface.app.input_state import InputState

__all__ = ["InputState"]
