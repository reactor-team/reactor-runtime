"""Surface and behaviour checks for the batch example.

The network is checked on NumPy, the backend every environment has: a batched
step yields, for each session, exactly what that session yields stepped alone.
The application is checked through its contract and its declared capacity.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from batch import Batch, NumpyNetwork, build_network, draw_weights
from reactor_runtime import BatchedReactorApp
from reactor_runtime.interface.model.contract import ModelContract
from reactor_runtime.manifest import import_model_class, load_config
from reactor_runtime.runner.session_host import batched_capacity

_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "batch"


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(Batch)


def _network() -> NumpyNetwork:
    return NumpyNetwork(width=48, height=32, latent=8, channels=4, layers=2)


def test_the_manifest_serves_the_batched_app() -> None:
    cfg = load_config(_EXAMPLE_DIR / "reactor.yaml")
    model_cls = import_model_class(cfg.model_ref)
    assert issubclass(model_cls, BatchedReactorApp)
    assert batched_capacity(cfg.model_ref) == 4
    assert cfg.step_results.enabled
    assert cfg.step_results.queue == 64 * batched_capacity(cfg.model_ref)


def test_commands_are_the_state_setters() -> None:
    assert set(ModelContract.of(Batch).commands) == {"set_seed", "set_tint"}


def test_a_batched_step_matches_each_session_stepped_alone() -> None:
    net = _network()
    seeds, tints = [3, 5, 8], [0.0, 0.5, 1.0]
    batched = [net.latent(seed) for seed in seeds]
    alone = [[net.latent(seed)] for seed in seeds]
    for step in range(4):
        batched, frames = net.step(batched, tints, [step] * 3)
        for index in range(3):
            alone[index], solo = net.step(alone[index], [tints[index]], [step])
            np.testing.assert_array_equal(frames[index], solo[0])


def test_frames_are_the_configured_size_and_differ_by_seed() -> None:
    net = _network()
    _, frames = net.step([net.latent(1), net.latent(2)], [0.5, 0.5], [0, 0])
    assert frames.shape == (2, 32, 48, 3)
    assert frames.dtype == np.uint8
    assert not np.array_equal(frames[0], frames[1])


def test_the_weights_are_float32() -> None:
    hidden, out = draw_weights(channels=4, layers=2)
    assert {weight.dtype for weight in (*hidden, out)} == {np.dtype(np.float32)}


def test_the_numpy_backend_is_chosen_on_request() -> None:
    assert isinstance(build_network({"backend": "numpy", "latent": 4}), NumpyNetwork)
