"""``FlashDreamsModel.generate()`` and ``reset()``: the rollout and cache lifecycle.

Driven against a fake pipeline; see ``flashdreams_fakes`` and ``conftest``.
"""

from __future__ import annotations

from typing import Any

import pytest
from flashdreams_fakes import (
    FRAMES,
    FakePipeline,
    SeededModel,
    Step,
    SwappingModel,
    loaded_model,
)

from reactor_runtime.flashdreams import (
    FlashDreamsModel,
    FlashDreamsResult,
    PromptSwapUnsupported,
    RolloutExhausted,
    RolloutNotStarted,
)


def test_generate_before_load_is_a_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        SeededModel().generate(Step(rollout_id=1, image=b"png"))


def test_a_new_rollout_id_starts_a_rollout_from_the_step(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = loaded_model(seed_index=0)
    result = model.generate(Step(rollout_id=1, image=b"png", control="ctrl"))
    assert model.started == [Step(rollout_id=1, image=b"png", control="ctrl")]
    assert model.rollout_id == 1
    # The seed sits at index 0, so the first generated step is index 1.
    assert [(index, input) for index, _, input in pipeline.generated] == [(1, "ctrl")]
    assert pipeline.finalized == [1]
    assert isinstance(result, FlashDreamsResult)
    assert result.index == 1
    assert result.rollout_id == 1
    assert result.frames is FRAMES


def test_a_cache_without_a_seed_starts_at_index_zero(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = loaded_model(seed_index=None)
    result = model.generate(Step(rollout_id=1, image=b"png"))
    assert result.index == 0
    assert pipeline.finalized == [0]


def test_the_same_rollout_id_continues_the_rollout(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = loaded_model()
    model.generate(Step(rollout_id=1, image=b"png"))
    second = model.generate(Step(rollout_id=1, image=None, control="right"))
    assert len(model.started) == 1
    assert second.index == 2
    assert second.rollout_id == 1
    assert pipeline.generated[-1][2] == "right"
    assert pipeline.finalized == [1, 2]


def test_a_new_rollout_id_drops_the_old_rollout_first(fake_flashdreams: dict[str, Any]) -> None:
    model, _ = loaded_model()
    model.generate(Step(rollout_id=1, image=b"png"))
    first_cache = model.cache
    result = model.generate(Step(rollout_id=2, image=b"jpg"))
    assert len(model.started) == 2
    assert model.cache is not first_cache
    assert result.rollout_id == 2
    assert result.index == 1


def test_a_new_rollout_without_conditioning_is_the_familys_error(
    fake_flashdreams: dict[str, Any],
) -> None:
    model, pipeline = loaded_model()
    model.generate(Step(rollout_id=1, image=b"png"))
    with pytest.raises(RolloutNotStarted):
        model.generate(Step(rollout_id=2, image=None))
    # The old rollout was dropped before the new one failed to start.
    assert model.cache is None
    assert model.rollout_id is None
    assert len(pipeline.generated) == 1


def test_the_rollout_is_exhausted_at_total_blocks(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = loaded_model(seed_index=0, total_blocks=3)
    model.generate(Step(rollout_id=1, image=b"png"))
    model.generate(Step(rollout_id=1))
    with pytest.raises(RolloutExhausted) as excinfo:
        model.generate(Step(rollout_id=1))
    assert excinfo.value.args == (3,)
    # The rollout is kept; the application decides what follows.
    assert model.rollout_id == 1
    assert model.cache is not None
    assert pipeline.finalized == [1, 2]


def test_a_pipeline_failure_drops_the_cache_and_propagates(
    fake_flashdreams: dict[str, Any],
) -> None:
    model, pipeline = loaded_model()
    model.generate(Step(rollout_id=1, image=b"png"))
    pipeline.fail_on = 2
    with pytest.raises(RuntimeError, match="device-side assert"):
        model.generate(Step(rollout_id=1))
    assert model.cache is None
    assert model.rollout_id is None
    assert pipeline.finalized == [1]
    # The next step with the same id starts the rollout again.
    result = model.generate(Step(rollout_id=1, image=b"png"))
    assert len(model.started) == 2
    assert result.index == 1


def test_a_new_prompt_in_the_same_rollout_is_refused_by_default(
    fake_flashdreams: dict[str, Any],
) -> None:
    model, pipeline = loaded_model()
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    with pytest.raises(PromptSwapUnsupported):
        model.generate(Step(rollout_id=1, prompt="a desert"))
    assert model.prompt == "a forest"
    assert model.cache is not None
    assert len(pipeline.generated) == 1


def test_the_same_prompt_or_no_prompt_does_not_swap(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = loaded_model()
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    model.generate(Step(rollout_id=1, prompt="a forest"))
    model.generate(Step(rollout_id=1, prompt=None))
    assert len(pipeline.generated) == 3
    assert model.prompt == "a forest"


def test_a_model_that_swaps_its_prompt_keeps_the_rollout(fake_flashdreams: dict[str, Any]) -> None:
    model = SwappingModel()
    pipeline = FakePipeline()
    model.pipeline = pipeline
    model.max_blocks = 100
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    result = model.generate(Step(rollout_id=1, prompt="a desert"))
    assert model.swapped == ["a desert"]
    assert model.prompt == "a desert"
    assert len(model.started) == 1
    assert result.index == 2


def test_a_new_rollout_records_its_prompt_without_swapping(
    fake_flashdreams: dict[str, Any],
) -> None:
    model = SwappingModel()
    model.pipeline = FakePipeline()
    model.max_blocks = 100
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    model.generate(Step(rollout_id=2, image=b"png", prompt="a desert"))
    assert model.swapped == []
    assert model.prompt == "a desert"


# -- reset --------------------------------------------------------------------


def test_reset_forgets_the_rollout_and_keeps_the_pipeline(fake_flashdreams: dict[str, Any]) -> None:
    model, pipeline = loaded_model()
    model.generate(Step(rollout_id=1, image=b"png", prompt="a forest"))
    model.reset()
    assert model.cache is None
    assert model.rollout_id is None
    assert model.prompt is None
    assert model.pipeline is pipeline
    result = model.generate(Step(rollout_id=1, image=b"png"))
    assert len(model.started) == 2
    assert result.index == 1


def test_initialize_cache_is_the_familys_to_write() -> None:
    with pytest.raises(NotImplementedError, match="initialize_cache"):
        FlashDreamsModel().initialize_cache(Step(rollout_id=1))
