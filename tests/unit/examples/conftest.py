"""Put each example workspace on the import path, the way ``reactor run`` does.

Every example is two modules that import each other as top-level names,
because the runtime imports a workspace with its directory first on
``sys.path``. The tests import them the same way, so one module object backs
each class.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).parents[3] / "examples"
for workspace in ("starter", "echo", "waypoint"):
    path = str(_EXAMPLES / workspace)
    if path not in sys.path:
        sys.path.insert(0, path)
