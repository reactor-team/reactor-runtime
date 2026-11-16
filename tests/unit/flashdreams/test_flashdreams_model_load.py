"""``FlashDreamsModel.load()``: the slug, the device, the caches, loguru, and what fails.

Driven against a fake registry; see ``flashdreams_fakes`` and ``conftest``.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Any

import pytest
from flashdreams_fakes import SeededModel

from reactor_runtime.flashdreams import model as model_module


def test_load_resolves_the_slug_and_builds_the_pipeline_on_the_device(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    model = SeededModel()
    model.load("action2v-fake", weights_root=str(tmp_path), device="cuda:1")
    assert model.app is fake_flashdreams["app"]
    assert model.pipeline is fake_flashdreams["pipeline"]
    assert fake_flashdreams["app"].pipeline_config.device == "cuda:1"
    assert model.device == "cuda:1"
    assert model.max_blocks == 10_000
    assert model.desc.frames_per_second_for_step == 60
    assert model.cache is None
    assert model.rollout_id is None


def test_load_builds_on_cuda_when_nothing_names_a_device(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    model = SeededModel()
    model.load("action2v-fake", weights_root=str(tmp_path))
    assert fake_flashdreams["app"].pipeline_config.device == "cuda"
    assert model.device == "cuda"


def test_load_builds_on_the_device_a_runner_set_for_the_rank(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    # A DistributedRunner sets rank, world_size, and device on the instance
    # between construction and load(), with the same load_kwargs on every rank.
    model = SeededModel()
    model.rank, model.world_size, model.device = 2, 4, "cuda:2"  # type: ignore[ty:unresolved-attribute]
    model.load("action2v-fake", weights_root=str(tmp_path))
    assert fake_flashdreams["app"].pipeline_config.device == "cuda:2"
    assert model.device == "cuda:2"


def test_an_explicit_device_overrides_the_one_a_runner_set(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    model = SeededModel()
    model.device = "cuda:2"
    model.load("action2v-fake", weights_root=str(tmp_path), device="cpu")
    assert fake_flashdreams["app"].pipeline_config.device == "cpu"
    assert model.device == "cpu"


def test_load_points_both_caches_at_the_weights_root(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    SeededModel().load("action2v-fake", weights_root=str(tmp_path))
    assert os.environ["FLASHDREAMS_CACHE_DIR"] == str(tmp_path)
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "huggingface")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_a_late_cache_pointing_warns_only_when_the_caches_were_elsewhere(
    fake_flashdreams: dict[str, Any],
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    with caplog.at_level(logging.WARNING, logger=model_module.__name__):
        # The application half pointed the caches here before it imported FlashDreams.
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "huggingface"))
        model_module._point_caches_at(str(tmp_path))
        assert caplog.records == []
        # Nothing did, or something pointed them elsewhere: the cache location is fixed.
        monkeypatch.setenv("HF_HUB_CACHE", "/somewhere/else")
        model_module._point_caches_at(str(tmp_path))
        assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "huggingface")


def test_load_keeps_an_offline_flag_the_environment_already_sets(
    fake_flashdreams: dict[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    SeededModel().load("action2v-fake", weights_root=str(tmp_path))
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_load_forwards_loguru_to_logging_once(
    fake_flashdreams: dict[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    sinks: list[Any] = []

    class FakeLoguru:
        def remove(self) -> None:
            calls.append("remove")

        def add(self, sink: Any, format: str) -> None:
            calls.append("add")
            sinks.append(sink)

    loguru = types.ModuleType("loguru")
    loguru.logger = FakeLoguru()  # type: ignore[ty:unresolved-attribute]
    monkeypatch.setitem(sys.modules, "loguru", loguru)

    SeededModel().load("action2v-fake", weights_root=str(tmp_path))
    SeededModel().load("action2v-fake", weights_root=str(tmp_path))
    assert calls == ["remove", "add"]

    received: list[logging.LogRecord] = []
    target = logging.getLogger("flashdreams.test")
    handler = logging.Handler()
    handler.emit = received.append  # type: ignore[ty:invalid-assignment]
    target.addHandler(handler)
    try:
        record = logging.LogRecord("flashdreams.test", logging.INFO, __file__, 1, "hello", (), None)
        sinks[0].emit(record)
    finally:
        target.removeHandler(handler)
    assert [r.getMessage() for r in received] == ["hello"]


def test_load_without_flashdreams_names_the_packages_to_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    for name in list(sys.modules):
        if name == "flashdreams" or name.startswith("flashdreams."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(model_module, "_loguru_forwarded", True)
    with pytest.raises(ModuleNotFoundError, match="flashdreams-action2v"):
        SeededModel().load("action2v-fake", weights_root=str(tmp_path))


def test_load_with_an_unknown_slug_is_the_registrys_error(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    with pytest.raises(LookupError, match="no-such-model"):
        SeededModel().load("no-such-model", weights_root=str(tmp_path))


def test_load_rejects_an_application_without_the_adapter_surface(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    class Bare:
        def session_desc(self) -> None:
            return None

    fake_flashdreams["registry"]["bare"] = Bare()
    with pytest.raises(TypeError, match="defaults, pipeline_config"):
        SeededModel().load("bare", weights_root=str(tmp_path))


def test_load_refuses_warmup_steps_on_a_model_that_does_not_warm_up(
    fake_flashdreams: dict[str, Any], tmp_path: Any
) -> None:
    with pytest.raises(NotImplementedError, match="warmup_steps"):
        SeededModel().load("action2v-fake", weights_root=str(tmp_path), warmup_steps=3)
