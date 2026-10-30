"""The model half's declarations: its limits, and its settings from the manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from reactor_runtime.core import (
    DEFAULT_SHM_INPUT_BYTES,
    DEFAULT_SHM_OUTPUT_BYTES,
    HealthStatus,
    WorkerConfig,
)
from reactor_runtime.model_server import ModelServer, ServerSettings


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


@pytest.mark.parametrize("key", ["shm_input_bytes", "shm_output_bytes"])
def test_a_subclass_that_sets_what_the_manifest_owns_is_refused(key: str) -> None:
    with pytest.raises(TypeError, match=f"runtime.worker.{key}"):
        type("Sized", (ModelServer,), {key: 1 << 20})


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
    max_batch = 4
    batch_window = 0.002


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
        max_batch=4,
        batch_window=0.002,
        shm_input_bytes=DEFAULT_SHM_INPUT_BYTES,
        shm_output_bytes=DEFAULT_SHM_OUTPUT_BYTES,
    )
    assert settings.shm_slot_bytes == DEFAULT_SHM_INPUT_BYTES + DEFAULT_SHM_OUTPUT_BYTES


def test_the_manifest_lowers_max_batch_and_sets_the_rest() -> None:
    config = WorkerConfig(
        import_ref="plasma_worker:PlasmaModel",
        max_batch=2,
        batch_window=0,
        shm_input_bytes=2 << 20,
        shm_output_bytes=32 << 20,
    )
    assert ServerSettings.resolve(PlasmaModel, config) == ServerSettings(
        max_batch=2,
        batch_window=0.0,
        shm_input_bytes=2 << 20,
        shm_output_bytes=32 << 20,
    )
    assert config.shm_slot_bytes == 34 << 20


def test_the_effective_sizes_are_the_declared_ones_or_the_defaults() -> None:
    unset = WorkerConfig(import_ref="m:C")
    declared = WorkerConfig(import_ref="m:C", shm_input_bytes=2 << 20, shm_output_bytes=64 << 20)

    assert (unset.input_bytes, unset.output_bytes) == (
        DEFAULT_SHM_INPUT_BYTES,
        DEFAULT_SHM_OUTPUT_BYTES,
    )
    assert (declared.input_bytes, declared.output_bytes) == (2 << 20, 64 << 20)
    for config in (unset, declared):
        settings = ServerSettings.resolve(PlasmaModel, config)
        assert (settings.shm_input_bytes, settings.shm_output_bytes) == (
            config.input_bytes,
            config.output_bytes,
        )
        assert settings.shm_slot_bytes == config.shm_slot_bytes


def test_the_default_output_block_fits_a_video_step() -> None:
    assert DEFAULT_SHM_OUTPUT_BYTES == 32 << 20


def test_the_manifest_cannot_raise_max_batch_above_the_class() -> None:
    config = WorkerConfig(import_ref="plasma_worker:PlasmaModel", max_batch=8)
    with pytest.raises(ValueError, match=r"above PlasmaModel\.max_batch = 4"):
        ServerSettings.resolve(PlasmaModel, config)
