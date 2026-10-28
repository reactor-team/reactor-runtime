"""The model half's declarations: data-only types, its limits, and settings from the manifest."""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest

from reactor_runtime.core import (
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_SHM_INPUT_BYTES,
    DEFAULT_SHM_OUTPUT_BYTES,
    HealthStatus,
    WorkerConfig,
)
from reactor_runtime.model_server import ModelServer, ModelServerTypes, ServerSettings


class Mode(enum.Enum):
    FAST = "fast"


@dataclass(frozen=True)
class Inner:
    label: str
    weights: list[float]


@dataclass(frozen=True)
class Tick:
    tick: int
    audio: np.ndarray
    mode: Mode
    speed: Literal["slow", "fast"]
    inner: Inner
    tags: tuple[str, ...]
    meta: dict[str, int]
    note: str | None
    raw: bytes


@dataclass(frozen=True)
class Frame:
    pixels: np.ndarray


@dataclass(frozen=True)
class Preempt:
    reason: str


TYPES = ModelServerTypes(request=Tick, result=Frame, signals=(Preempt,))


# -- declared types -------------------------------------------------------------------------


def test_data_types_are_accepted() -> None:
    assert TYPES.request is Tick
    assert TYPES.signals == (Preempt,)


@dataclass
class Mutable:
    value: int


@dataclass(frozen=True)
class HoldsAFunction:
    on_done: Any


@dataclass(frozen=True)
class HoldsALock:
    lock: threading.Lock


@dataclass(frozen=True)
class FloatKeys:
    table: dict[float, int]


@dataclass(frozen=True)
class Base:
    value: int


class Undecorated(Base):
    extra: int


@pytest.mark.parametrize(
    ("request_type", "match"),
    [
        (dict, "not a dataclass"),
        (Undecorated, "not a dataclass declared with @dataclass"),
        (Mutable, "not frozen"),
        (HoldsAFunction, r"HoldsAFunction\.on_done is typed"),
        (HoldsALock, r"HoldsALock\.lock is typed"),
        (FloatKeys, "keys must be str or int"),
    ],
)
def test_a_type_that_is_not_data_is_refused_naming_it(request_type: Any, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        ModelServerTypes(request=request_type, result=Frame)


def test_signals_are_a_tuple_of_distinct_types() -> None:
    with pytest.raises(TypeError, match="tuple"):
        ModelServerTypes(request=Tick, result=Frame, signals=[Preempt])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # a list is the mistake under test
    with pytest.raises(TypeError, match="twice"):
        ModelServerTypes(request=Tick, result=Frame, signals=(Preempt, Preempt))


# -- the class ------------------------------------------------------------------------------


def test_a_subclass_without_types_is_refused() -> None:
    with pytest.raises(TypeError, match="types must be a ModelServerTypes"):

        class Untyped(ModelServer):
            pass


@pytest.mark.parametrize("key", ["shm_input_bytes", "shm_output_bytes", "max_request_bytes"])
def test_a_subclass_that_sets_what_the_manifest_owns_is_refused(key: str) -> None:
    with pytest.raises(TypeError, match=f"runtime.worker.{key}"):
        type("Sized", (ModelServer,), {"types": TYPES, key: 1 << 20})


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
        type("OutOfRange", (ModelServer,), {"types": TYPES, **attrs})


class PlasmaModel(ModelServer):
    types = TYPES
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
        max_request_bytes=DEFAULT_MAX_REQUEST_BYTES,
        shm_input_bytes=DEFAULT_SHM_INPUT_BYTES,
        shm_output_bytes=DEFAULT_SHM_OUTPUT_BYTES,
    )
    assert settings.shm_slot_bytes == DEFAULT_SHM_INPUT_BYTES + DEFAULT_SHM_OUTPUT_BYTES


def test_the_manifest_lowers_max_batch_and_sets_the_rest() -> None:
    config = WorkerConfig(
        import_ref="plasma_worker:PlasmaModel",
        max_batch=2,
        batch_window=0,
        max_request_bytes=1 << 20,
        shm_input_bytes=2 << 20,
        shm_output_bytes=32 << 20,
    )
    assert ServerSettings.resolve(PlasmaModel, config) == ServerSettings(
        max_batch=2,
        batch_window=0.0,
        max_request_bytes=1 << 20,
        shm_input_bytes=2 << 20,
        shm_output_bytes=32 << 20,
    )
    assert config.shm_slot_bytes == 34 << 20


def test_the_manifest_cannot_raise_max_batch_above_the_class() -> None:
    config = WorkerConfig(import_ref="plasma_worker:PlasmaModel", max_batch=8)
    with pytest.raises(ValueError, match=r"above PlasmaModel\.max_batch = 4"):
        ServerSettings.resolve(PlasmaModel, config)
