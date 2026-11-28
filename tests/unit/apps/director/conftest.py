"""Fixtures for the director tests; the harness itself is in ``director_harness``."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_registries(isolate_interface_registries: None) -> None:
    """Keep the message and track classes each test declares out of the others."""
