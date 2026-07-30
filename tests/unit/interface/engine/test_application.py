from collections.abc import Callable

import pytest
from fake_engine import FakeEngine, FakeInit, Move

from reactor_runtime.interface.engine import EnginePipeline, application_for, is_engine
from reactor_runtime.interface.engine.reflection import VIDEO_TRACK
from reactor_runtime.interface.model.contract import ModelContract


class NotAnEngine:
    """Has some of the calls, so presence of one method is not enough."""

    def generate(self, autoregressive_index: int, cache: object, input: object = None) -> None:
        return None

    def finalize(self, autoregressive_index: int, cache: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolated(isolate_interface_registries: None) -> None:
    """Every application built here declares its own tracks; keep them apart."""


def test_a_class_with_the_four_calls_is_an_engine() -> None:
    assert is_engine(FakeEngine)


def test_a_class_missing_a_call_is_not_an_engine() -> None:
    # It has generate and finalize, but no map_inputs: an engine that cannot
    # fold a window is not servable.
    assert not is_engine(NotAnEngine)


def test_an_instance_is_not_an_engine_class() -> None:
    assert not is_engine(FakeEngine())


def test_an_application_is_built_around_the_engine() -> None:
    app = application_for(FakeEngine)

    assert issubclass(app, EnginePipeline)
    assert app.engine is FakeEngine


def test_the_built_application_serves_the_engines_declarations(
    register_model: Callable[[type], None],
) -> None:
    app = application_for(FakeEngine)
    register_model(app)

    contract = ModelContract.of(app)
    assert "move" in contract.commands
    assert "init" in contract.commands
    assert "camera" in contract.tracks


def test_the_built_application_emits_on_the_default_video_track(
    register_model: Callable[[type], None],
) -> None:
    app = application_for(FakeEngine)
    register_model(app)

    assert ModelContract.of(app).tracks[VIDEO_TRACK].direction == "out"


def test_the_engines_name_carries_into_the_published_model(
    register_model: Callable[[type], None],
) -> None:
    app = application_for(FakeEngine)
    register_model(app)

    assert ModelContract.of(app).model == "fake_engine"


async def test_the_built_application_runs_a_step() -> None:
    model = application_for(FakeEngine)()
    model.load(None)
    model._on_loop_ready()

    chunk = await model.step()

    assert chunk is not None
    assert chunk.shape[0] == 1


def test_the_declarations_are_the_engines_own() -> None:
    inputs = application_for(FakeEngine).__engine_inputs__

    assert inputs.events == {"move": Move}
    assert inputs.init is FakeInit


# -- resolving a manifest's model reference ------------------------------------


def test_a_manifest_may_point_straight_at_an_engine() -> None:
    from reactor_runtime.runner.runner import import_model_class

    resolved = import_model_class("scoped_engine:ScopedEngine")

    assert issubclass(resolved, EnginePipeline)
    assert resolved.engine.__name__ == "ScopedEngine"


def test_a_reference_to_neither_a_model_nor_an_engine_is_rejected() -> None:
    from reactor_runtime.runner.runner import import_model_class

    with pytest.raises(TypeError, match="neither a ReactorCore subclass nor an inference engine"):
        import_model_class("scoped_engine:ScopedStepInput")
