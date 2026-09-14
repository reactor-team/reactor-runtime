"""Put the waypoint workspace on the import path, the way ``reactor run`` does.

``examples/waypoint`` is two modules that import each other as top-level names,
because the runtime imports a workspace with its directory first on ``sys.path``.
The tests import them the same way, so one module object backs each class.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WAYPOINT = Path(__file__).parents[3] / "examples" / "waypoint"
if str(_WAYPOINT) not in sys.path:
    sys.path.insert(0, str(_WAYPOINT))
