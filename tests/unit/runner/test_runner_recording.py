"""What the session descriptor says about recording for a model with no video.

A module of its own: the test registry replays every track class a module
declares, so the audio-only model must not share one with a video model.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from reactor_runtime import Audio, Output, ReactorApp
from reactor_runtime.core import RuntimeConfig
from reactor_runtime.runner.runner import Runner


class AudioOut(Output):
    main_audio: Audio


class AudioOnlyModel(ReactorApp):
    """A model whose only output track is audio, which the recorder cannot record."""

    output: AudioOut

    def load(self, config_path: Path | None) -> None: ...

    async def run(self) -> None:
        await asyncio.sleep(60)


@pytest.fixture(autouse=True)
def _seed_registries(
    isolate_interface_registries: None, register_model: Callable[[type], None]
) -> None:
    register_model(AudioOnlyModel)


async def test_the_descriptor_reports_an_audio_only_model_as_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Recording is on by default, but the recorder records video: a model with
    # no output video track records nothing, so the start says so.
    monkeypatch.setattr(
        "reactor_runtime.runner.runner.import_model_class", lambda ref: AudioOnlyModel
    )
    runner = Runner(RuntimeConfig(model_ref="audio:Model"))
    await runner.start()
    try:
        runner.start_session({})
        assert runner.descriptor()["recording"] == {"enabled": False, "chunk_seconds": 4}
    finally:
        await runner.stop()
