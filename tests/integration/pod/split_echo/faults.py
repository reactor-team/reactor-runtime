"""One injected fault per run, for the pod test to end one session with.

``FAULT_STAGE`` (``input``, ``output`` or ``model``) and ``FAULT_SLOT`` pick
where and for whom; the fault fires once, the first time ``FAULT_MARK`` does
not exist yet.
"""

from __future__ import annotations

import os
from pathlib import Path


def fault_due(stage: str, slot: int) -> bool:
    """Whether the injected fault for *stage* fires now for *slot*."""
    if os.getenv("FAULT_STAGE") != stage or os.getenv("FAULT_SLOT") != str(slot):
        return False
    mark = Path(os.environ["FAULT_MARK"])
    if mark.exists():
        return False
    mark.write_text(stage)
    return True
