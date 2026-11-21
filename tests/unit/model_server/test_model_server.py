"""The model half's declarations: its limits, and its settings from the manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from reactor_runtime.core import (
    DEFAULT_SHM_INPUT_BYTES,
    DEFAULT_SHM_OUTPUT_BYTES,
    SHM_PAGE_BYTES,
    HealthStatus,
    ModelServerConfig,
    SharedMemoryConfig,
    shm_slot_bytes,
)
from reactor_runtime.model_server import DEFAULT_BATCH_WINDOW, ModelServer, ServerSettings


@dataclass(frozen=True)
class Tick:
    tick: int


@dataclass(frozen=True)
class Frame:
    pixels: np.ndarray


@dataclass(frozen=True)
class Preempt:
    reason: str


# -- the class ------------------------------------------------------------------------------


def test_a_subclass_declares_nothing_and_its_type_parameters_are_for_type_checkers() -> None:
    class Bare(ModelServer):
        pass

    class Typed(ModelServer[Tick, Frame]):
        pass

    assert Bare().max_batch == Typed().max_batch == 1


@pytest.mark.parametrize(
    ("attribute", "key"),
    [
        ("shm_input_bytes", r"model\.resources\.sessions\.shared_memory\.input"),
        ("shm_output_bytes", r"model\.resources\.sessions\.shared_memory\.output"),
    ],
)
def test_a_subclass_that_sets_what_the_manifest_owns_is_refused(attribute: str, key: str) -> None:
    with pytest.raises(TypeError, match=rf"Sized\.{attribute}: set it in reactor\.yaml as {key}"):
        type("Sized", (ModelServer,), {attribute: 1 << 20})


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"max_batch": 0}, "max_batch must be a positive integer"),
        ({"max_batch": 2.0}, "max_batch must be a positive integer"),
        ({"batch_window": -0.1}, "batch_window must be a number of seconds"),
        ({"batch_window": float("inf")}, "batch_window must be a number of seconds"),
    ],
)
def test_a_subclass_with_limits_out_of_range_is_refused(attrs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        type("OutOfRange", (ModelServer,), attrs)


class PlasmaModel(ModelServer[Tick, Frame]):
    batch_window = 0.002


class ClassBatchModel(PlasmaModel):
    max_batch = 4


class InheritsClassBatch(ClassBatchModel):
    pass


def test_the_default_hooks_report_healthy_and_step_must_be_written() -> None:
    model = PlasmaModel()
    assert model.health().status is HealthStatus.HEALTHY
    model.session_opened(0, 1)
    model.signal(0, Preempt(reason="barge-in"))
    model.session_closed(0, 1)
    with pytest.raises(NotImplementedError, match="PlasmaModel must define step"):
        model.step([])


# -- settings -------------------------------------------------------------------------------


def test_without_a_manifest_block_the_class_and_the_runtime_defaults_apply() -> None:
    settings = ServerSettings.resolve(PlasmaModel, None)
    assert settings == ServerSettings(
        max_batch=1,
        batch_window=0.002,
        shm_input_bytes=DEFAULT_SHM_INPUT_BYTES,
        shm_output_bytes=DEFAULT_SHM_OUTPUT_BYTES,
    )
    assert settings.shm_slot_bytes == 2 * DEFAULT_SHM_INPUT_BYTES + DEFAULT_SHM_OUTPUT_BYTES


def test_the_manifest_sets_max_batch_and_the_rest() -> None:
    config = ModelServerConfig(import_ref="plasma_model:PlasmaModel", max_batch=2, batch_window=0)
    shared_memory = SharedMemoryConfig(input_bytes=2 << 20, output_bytes=32 << 20)
    assert ServerSettings.resolve(PlasmaModel, config, shared_memory) == ServerSettings(
        max_batch=2,
        batch_window=0.0,
        shm_input_bytes=2 << 20,
        shm_output_bytes=32 << 20,
    )
    assert shared_memory.slot_bytes == 36 << 20


def test_the_slot_bound_counts_each_block_in_whole_64_kib_pages() -> None:
    page = SHM_PAGE_BYTES
    assert page == 64 << 10
    assert shm_slot_bytes(page, 4 * page) == 2 * page + 4 * page
    assert shm_slot_bytes(1, page + 1) == 2 * page + 2 * page
    assert SharedMemoryConfig(input_bytes=1, output_bytes=1).slot_bytes == 3 * page


def test_the_shared_memory_sizes_are_the_declared_ones_or_the_defaults() -> None:
    unset = SharedMemoryConfig()
    declared = SharedMemoryConfig(input_bytes=2 << 20, output_bytes=64 << 20)

    assert (unset.input_bytes, unset.output_bytes) == (
        DEFAULT_SHM_INPUT_BYTES,
        DEFAULT_SHM_OUTPUT_BYTES,
    )
    assert (declared.input_bytes, declared.output_bytes) == (2 << 20, 64 << 20)
    config = ModelServerConfig(import_ref="m:C")
    for shared_memory in (unset, declared):
        settings = ServerSettings.resolve(PlasmaModel, config, shared_memory)
        assert (settings.shm_input_bytes, settings.shm_output_bytes) == (
            shared_memory.input_bytes,
            shared_memory.output_bytes,
        )
        assert settings.shm_slot_bytes == shared_memory.slot_bytes


@pytest.mark.parametrize(
    ("sizes", "key"),
    [
        ({"input_bytes": 0}, "input"),
        ({"output_bytes": -1}, "output"),
        ({"output_bytes": True}, "output"),
        ({"input_bytes": 1.0}, "input"),
    ],
)
def test_a_shared_memory_size_that_is_not_a_positive_integer_is_refused(
    sizes: dict[str, Any], key: str
) -> None:
    with pytest.raises(
        ValueError, match=rf"model\.resources\.sessions\.shared_memory\.{key} must be a positive"
    ):
        SharedMemoryConfig(**sizes)


def test_the_default_output_block_fits_a_video_step() -> None:
    assert DEFAULT_SHM_OUTPUT_BYTES == 32 << 20


@pytest.mark.parametrize("server", [ClassBatchModel, InheritsClassBatch])
def test_a_class_max_batch_applies_when_the_manifest_sets_none(
    server: type[ModelServer],
) -> None:
    for config in (None, ModelServerConfig(import_ref="m:C")):
        assert ServerSettings.resolve(server, config).max_batch == 4
    same = ModelServerConfig(import_ref="m:C", max_batch=4)
    assert ServerSettings.resolve(server, same).max_batch == 4


def test_the_manifest_lowers_a_class_max_batch() -> None:
    config = ModelServerConfig(import_ref="plasma_model:ClassBatchModel", max_batch=2)
    assert ServerSettings.resolve(ClassBatchModel, config).max_batch == 2


def test_a_manifest_max_batch_above_the_class_is_refused() -> None:
    config = ModelServerConfig(import_ref="plasma_model:ClassBatchModel", max_batch=8)
    with pytest.raises(
        ValueError,
        match=r"runtime\.model_server\.max_batch is 8, above ClassBatchModel\.max_batch = 4",
    ):
        ServerSettings.resolve(ClassBatchModel, config)


def test_the_manifest_max_batch_applies_above_the_default() -> None:
    config = ModelServerConfig(import_ref="plasma_model:PlasmaModel", max_batch=16)
    assert ServerSettings.resolve(PlasmaModel, config).max_batch == 16


class _Plain(ModelServer):
    def step(self, requests):  # pragma: no cover - not stepped
        return []


class _ClassWindow(_Plain):
    batch_window = 0.02


class _ClassBatch(_Plain):
    max_batch = 8


class _ClassBatchAndWindow(_Plain):
    max_batch = 16
    batch_window = 0.0


@pytest.mark.parametrize(
    ("server", "max_batch", "batch_window", "expected"),
    [
        (_Plain, 8, None, DEFAULT_BATCH_WINDOW),  # the reactor.yaml form batches
        (_Plain, 1, None, 0.0),  # one request per step: nothing to wait for
        (_Plain, None, None, 0.0),  # no max_batch: 1
        (_Plain, 8, 0.0, 0.0),  # the manifest's window wins, 0 included
        (_Plain, 8, 0.03, 0.03),
        (_ClassWindow, 8, None, 0.02),  # the class's window wins over the default
        (_ClassBatch, None, None, DEFAULT_BATCH_WINDOW),  # one rule, wherever max_batch is set
        (_ClassBatch, 4, None, DEFAULT_BATCH_WINDOW),  # a manifest that lowers a class value
        (_ClassBatch, 1, None, 0.0),  # lowered to 1: nothing to wait for
        (_ClassBatchAndWindow, None, None, 0.0),  # the class's window wins, 0 included
    ],
)
def test_batch_window_defaults_to_a_few_milliseconds_when_max_batch_is_above_1(
    server: type[ModelServer], max_batch: int | None, batch_window: float | None, expected: float
) -> None:
    config = ModelServerConfig(import_ref="m:M", max_batch=max_batch, batch_window=batch_window)
    assert ServerSettings.resolve(server, config).batch_window == expected
