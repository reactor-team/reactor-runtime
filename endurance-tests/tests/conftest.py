"""Fixtures for the endurance scenarios."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from harness import RuntimeProcess
from report import DEFAULT_OUTPUT_DIR


@pytest.fixture
def runtime(request: pytest.FixtureRequest) -> Iterator[RuntimeProcess]:
    """A fresh runtime process for one scenario, its log kept beside the reports."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = RuntimeProcess(DEFAULT_OUTPUT_DIR / f"{request.node.name}-runtime.log")
    try:
        proc.wait_ready()
        yield proc
    finally:
        proc.stop()
