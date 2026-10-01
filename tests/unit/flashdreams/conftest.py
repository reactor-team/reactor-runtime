"""Fixtures the FlashDreams model-half tests share: a fake registry and a fake torch."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from flashdreams_fakes import FakeApp, FakePipeline

from reactor_runtime.flashdreams import model as model_module


@pytest.fixture
def fake_flashdreams(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for FlashDreams' registry and for torch, and reset the loguru forward."""
    pipeline = FakePipeline()
    app = FakeApp(pipeline)
    registry: dict[str, Any] = {"action2v-fake": app}

    package = types.ModuleType("flashdreams")
    runtime_v2 = types.ModuleType("flashdreams.runtime_v2")
    module = types.ModuleType("flashdreams.runtime_v2.application_registry")

    def create_application(slug: str) -> Any:
        try:
            return registry[slug]
        except KeyError:
            raise LookupError(f"No FlashDreams v2 application matches {slug!r}.") from None

    module.create_application = create_application  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "flashdreams", package)
    monkeypatch.setitem(sys.modules, "flashdreams.runtime_v2", runtime_v2)
    monkeypatch.setitem(sys.modules, "flashdreams.runtime_v2.application_registry", module)

    torch = types.ModuleType("torch")
    torch.uint8 = "uint8"  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "torch", torch)

    monkeypatch.setattr(model_module, "_loguru_forwarded", False)
    for name in ("FLASHDREAMS_CACHE_DIR", "HF_HUB_CACHE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    return {"pipeline": pipeline, "app": app, "registry": registry}
