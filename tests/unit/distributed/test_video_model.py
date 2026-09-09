# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Fail fast on video-adapter mistakes, before starting processes or GPU work."""

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactor_runtime import Output, Video
from reactor_runtime.distributed import DistributedVideoModel, DistributedWorker
from reactor_runtime.distributed.model import _snapshot


class RenderedOutput(Output):
    video: Video


class Worker(DistributedWorker):
    def setup(self, **setup_kwargs: Any) -> None:
        self.params = setup_kwargs

    def start_session(self, params: dict[str, Any]) -> None:
        pass

    def generate_chunk(self, index: int, controls: dict[str, Any]) -> Any:
        return np.zeros((1, 2, 2, 3), dtype=np.uint8)

    def end_session(self) -> None:
        pass


class Model(DistributedVideoModel):
    worker = Worker
    frame_shape = (1, 2, 2, 3)

    def to_output(self, frames: np.ndarray) -> RenderedOutput:
        return RenderedOutput(video=frames)


@pytest.mark.parametrize(
    ("attribute", "value", "error", "message"),
    [
        ("worker", None, TypeError, "declare worker"),
        ("worker", Worker(), TypeError, "declare worker"),
        ("frame_shape", None, TypeError, "declare frame_shape"),
        ("frame_shape", (1, 2, 3), ValueError, "frame_shape"),
        ("session_seed", -1, ValueError, "session_seed"),
        ("command_timeout", 0, ValueError, "command_timeout"),
        ("startup_timeout", float("nan"), ValueError, "startup_timeout"),
        ("shutdown_timeout", float("inf"), ValueError, "shutdown_timeout"),
    ],
)
def test_invalid_declarations_fail_before_allocating_workers(
    monkeypatch: pytest.MonkeyPatch,
    attribute,
    value,
    error,
    message,
) -> None:
    monkeypatch.setattr(Model, attribute, value)
    model = Model()
    try:
        with pytest.raises(error, match=message):
            model.load(None)
        assert model._executor is None
    finally:
        model._loop.close()


def test_missing_output_mapping_explains_the_required_hook() -> None:
    class NoOutput(DistributedVideoModel):
        worker = Worker
        frame_shape = (1, 2, 2, 3)

    model = NoOutput()
    try:
        with pytest.raises(TypeError, match="implement to_output"):
            model.load(None)
    finally:
        model._loop.close()


def test_nested_controls_are_owned_snapshots() -> None:
    original = {"nested": {"values": [1, 2]}, "image": np.zeros((2, 2), dtype=np.uint8)}
    snapshot = _snapshot(original, "controls()")
    original["nested"]["values"].append(3)
    original["image"][:] = 7
    assert snapshot["nested"]["values"] == [1, 2]
    assert (snapshot["image"] == 0).all()


def test_unpicklable_controls_fail_before_entering_a_queue() -> None:
    with pytest.raises(TypeError, match=r"controls\(\).*picklable CPU data"):
        _snapshot({"lock": threading.Lock()}, "controls()")


def test_configuration_reaches_worker_setup_without_a_load_override() -> None:
    class Configured(Model):
        def worker_setup(self, config_path: Path | None) -> dict[str, Any]:
            return {"path": str(config_path)}

    model = Configured()
    try:
        model.load(Path("weights.yaml"))
        assert model._workers is not None
        assert isinstance(model._workers._local_worker, Worker)
        assert model._workers._local_worker.params == {"path": "weights.yaml"}
    finally:
        if model._executor is not None and model._workers is not None:
            model._executor.submit(model._workers.shutdown).result()
            model._executor.shutdown(wait=True)
        model._loop.close()
