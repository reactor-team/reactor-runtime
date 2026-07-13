"""The external-consumer contract suite.

Everything under ``tests/contract/`` locks the runtime's observable surface as
an external consumer reads it: the ``/events`` SSE journal (framing, the single
``transition`` envelope, the event and state vocabulary, per-event ``detail``
payloads), the session-lifecycle routes and their status-code semantics, the
fixed transport session id, the recording id adopted from ``start_session``,
the ``/clips`` segment paths, and the seedable upload slots.

A failing test in this suite is a breaking change for the consumers built on
this surface — fix the change, do not update the test. Deliberate contract
changes must be made knowingly, with the downstream consumers migrated, and
only then may the literals here move.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from contract_helpers import Harness, running_runtime


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    """A started runtime serving the full HTTP surface over the fixture model."""
    async with running_runtime() as running:
        yield running
