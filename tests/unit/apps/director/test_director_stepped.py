"""A director app in a stepped session: one shot per request, returned as its result."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
from director_harness import Harness

from reactor_runtime import InputField, Output, Video
from reactor_runtime.apps.director import ReactorDirectorApp, Shot
from reactor_runtime.core.values import StepRefusalKind


class Frames(Output):
    main_video: Video


class Steps(ReactorDirectorApp):
    """Builds a shot whose every frame holds the number of the call that built it."""

    fps = 24

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str = InputField(), continue_from: Shot | None = None) -> Frames:
        self.calls.append({"prompt": prompt, "continue_from": continue_from})
        if prompt == "storm":
            raise RuntimeError("cannot draw a storm")
        return Frames(main_video=np.full((3, 4, 4, 3), len(self.calls), np.uint8))


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h = Harness(Steps())
    try:
        await h.start(stepped=True)
        yield h
    finally:
        await h.stop()


def _steps(h: Harness) -> Steps:
    assert isinstance(h.app, Steps)
    return h.app


def _data(h: Harness, index: int) -> dict[str, Any]:
    data = h.reports[index].data
    assert data is not None
    return json.loads(data)


async def test_a_request_builds_the_shot_it_queued_and_answers_with_it(harness: Harness) -> None:
    harness.step("r1", ("enqueue", {"prompt": "dawn"}))
    await harness.until(lambda: len(harness.reports) == 1)

    report = harness.reports[0]
    assert report.request is not None
    assert (report.request.request_id, report.request.index) == ("r1", 1)
    assert report.bundle is not None
    assert report.bundle.frame_count == 3
    assert _data(harness, 0)["inputs"]["prompt"] == "dawn"
    assert _data(harness, 0)["status"] == "built"
    assert harness.chunks == []
    assert harness.state()["playout"] == []
    assert [shot["shot_id"] for shot in harness.state()["history"]] == [
        _data(harness, 0)["shot_id"]
    ]


async def test_a_request_continues_a_shot_from_an_earlier_result(harness: Harness) -> None:
    harness.step("r1", ("enqueue", {"prompt": "dawn"}))
    await harness.until(lambda: len(harness.reports) == 1)
    first = _data(harness, 0)["shot_id"]
    harness.step("r2", ("enqueue", {"prompt": "river", "continue_from": first}))
    await harness.until(lambda: len(harness.reports) == 2)

    given = _steps(harness).calls[1]["continue_from"]
    assert given is not None
    assert given.id == first
    assert int(given.last_frame[0, 0, 0]) == 1


async def test_a_request_with_no_shot_to_build_is_refused(harness: Harness) -> None:
    harness.step("r1")
    await harness.settle()
    assert [(r.request_id, r.reason) for r in harness.refusals] == [("r1", "no shot is queued")]
    assert harness.reports == []


async def test_each_request_builds_one_shot_front_first(harness: Harness) -> None:
    harness.step("r1", ("enqueue", {"prompt": "one"}), ("enqueue", {"prompt": "two"}))
    await harness.until(lambda: len(harness.reports) == 1)
    harness.step("r2")
    await harness.until(lambda: len(harness.reports) == 2)
    assert [call["prompt"] for call in _steps(harness).calls] == ["one", "two"]
    assert [_data(harness, i)["inputs"]["prompt"] for i in range(2)] == ["one", "two"]


async def test_a_failed_shot_answers_its_request_with_the_error(harness: Harness) -> None:
    harness.step("r1", ("enqueue", {"prompt": "storm"}))
    await harness.until(lambda: len(harness.reports) == 1)
    report = harness.reports[0]
    assert report.request is not None
    assert report.request.request_id == "r1"
    assert report.error == "RuntimeError: cannot draw a storm"
    assert harness.app._crash is None

    harness.step("r2", ("enqueue", {"prompt": "calm"}))
    await harness.until(lambda: len(harness.reports) == 2)
    assert harness.reports[1].error is None


async def test_play_is_refused_in_a_stepped_session(harness: Harness) -> None:
    harness.step("r1", ("play", {}))
    await harness.settle()
    assert len(harness.refusals) == 1
    assert harness.refusals[0].kind is StepRefusalKind.COMMAND_FAILED
    assert harness.refusals[0].code == "not_streaming"
