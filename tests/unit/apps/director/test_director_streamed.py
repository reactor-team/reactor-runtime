"""A director app in a streamed session: building, steps, continuations, playout."""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from director_harness import CLIENT, Harness, encode_mp4

from reactor_runtime import Audio, InputField, LocalFile, Output, Video
from reactor_runtime.apps.director import ReactorDirectorApp, Shot
from reactor_runtime.core.model import ClientConnected, EndReason, SessionEnded
from reactor_runtime.core.values import CompletedStep, ConnId
from reactor_runtime.distributed.errors import WorkerCrashed

FPS = 120


class Clip(Output):
    main_video: Video
    main_audio: Audio


def _clip(frames: int, value: int) -> Clip:
    return Clip(
        main_video=np.full((frames, 4, 4, 3), value, np.uint8),
        main_audio=np.full((1, round(frames * 48_000 / FPS)), value, np.int16),
    )


class Studio(ReactorDirectorApp):
    """Builds a shot whose every frame holds the number of the call that built it."""

    fps = FPS

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []
        self.fail: dict[str, BaseException] = {}
        self.hold: threading.Event | None = None

    def generate(
        self,
        prompt: str = InputField(description="What the shot shows."),
        frames: int = InputField(default=6, ge=1, le=120),
        continue_from: Shot | None = None,
        ending_from: Shot | None = None,
    ) -> Clip:
        self.calls.append(
            {"prompt": prompt, "continue_from": continue_from, "ending_from": ending_from}
        )
        if self.hold is not None:
            self.hold.wait(5)
        if prompt in self.fail:
            raise self.fail[prompt]
        return _clip(frames, len(self.calls))


class SmallStudio(Studio):
    max_queued = 2
    history = 1


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h = Harness(Studio())
    try:
        yield h
    finally:
        await h.stop()


def _studio(h: Harness) -> Studio:
    assert isinstance(h.app, Studio)
    return h.app


def _ready_ids(h: Harness) -> list[str]:
    return [shot["shot_id"] for shot in h.state()["playout"]]


def _shot_json(report: CompletedStep) -> dict[str, Any]:
    described = report.files["shot.json"]
    assert isinstance(described, bytes)
    return json.loads(described)


def _hold(h: Harness) -> threading.Event:
    """Hold the next build open until the returned event is set."""
    hold = threading.Event()
    _studio(h).hold = hold
    return hold


def _frames(h: Harness) -> list[int]:
    """The value of every frame emitted, in order."""
    values: list[int] = []
    for chunk in h.chunks:
        video = chunk.bundle.tracks["main_video"].data
        values.extend(int(frame[0, 0, 0]) for frame in video)
    return values


# -- building and steps --------------------------------------------------------


async def test_a_starting_input_session_builds_its_shot_and_reports_it_as_one_step() -> None:
    h = Harness(Studio())
    try:
        await h.start(starting_input=True)
        request = h.command("enqueue", prompt="a lighthouse", metadata="scene 1")
        await h.settle()
        assert _studio(h).calls == []
        h.apply_starting_input()
        await h.until(lambda: len(h.reports) == 1)
    finally:
        await h.stop()

    report = h.reports[0]
    assert report.error is None
    assert report.bundle is not None
    assert report.bundle.frame_count == 6
    assert set(report.files) == {"shot.json", "last_frame.png"}
    described = _shot_json(report)
    assert described["shot_id"] == h.shot_id(request)
    assert described["inputs"]["prompt"] == "a lighthouse"
    assert described["metadata"] == "scene 1"
    assert report.data is not None
    assert json.loads(report.data)["shot_id"] == described["shot_id"]
    assert [shot["shot"]["shot_id"] for shot in h.sent("shot_generated")] == [described["shot_id"]]


async def test_a_queued_shot_replies_with_its_id_and_waits_in_the_state(harness: Harness) -> None:
    hold = _hold(harness)
    await harness.start()
    request = harness.command("enqueue", prompt="a fox")
    await harness.until(lambda: bool(_studio(harness).calls))
    shot_id = harness.shot_id(request)
    assert harness.state()["generation"][0]["shot_id"] == shot_id
    assert harness.state()["generation"][0]["status"] == "building"
    hold.set()
    await harness.until(lambda: _ready_ids(harness) == [shot_id])
    assert harness.state()["playout"][0]["status"] == "ready"
    assert harness.state()["playout"][0]["frames"] == 6


async def test_build_paused_holds_new_builds(harness: Harness) -> None:
    await harness.start()
    harness.command("set_build_paused", build_paused=True)
    harness.command("enqueue", prompt="a fox")
    await harness.settle()
    assert _studio(harness).calls == []
    harness.command("set_build_paused", build_paused=False)
    await harness.until(lambda: len(harness.reports) == 1)


async def test_a_continuation_waits_for_its_source_and_receives_its_last_frame(
    harness: Harness,
) -> None:
    await harness.start()
    harness.command("set_build_paused", build_paused=True)
    first = harness.command("enqueue", prompt="dawn")
    await harness.settle()
    a = harness.shot_id(first)
    # Queued ahead of its source: the scheduler skips it until the source is built.
    harness.command("enqueue", prompt="river", continue_from=a, position=0)
    await harness.settle()
    harness.command("set_build_paused", build_paused=False)
    await harness.until(lambda: len(harness.reports) == 2)

    calls = _studio(harness).calls
    assert [call["prompt"] for call in calls] == ["dawn", "river"]
    given: Shot = calls[1]["continue_from"]
    assert given.id == a
    assert given.inputs["prompt"] == "dawn"
    assert given.last_frame is not None
    assert int(given.last_frame[0, 0, 0]) == 1
    assert harness.state()["playout"][1]["inputs"]["continue_from"] == a


async def test_enqueue_refuses_a_shot_id_it_does_not_hold(harness: Harness) -> None:
    await harness.start()
    request = harness.command("enqueue", prompt="x", continue_from="no-such-shot")
    await harness.settle()
    assert harness.failure(request) == "unknown_shot"
    assert _studio(harness).calls == []


async def test_enqueue_refuses_a_shot_past_the_queue_capacity() -> None:
    h = Harness(SmallStudio())
    try:
        await h.start()
        h.command("set_build_paused", build_paused=True)
        h.command("enqueue", prompt="a")
        h.command("enqueue", prompt="b")
        third = h.command("enqueue", prompt="c")
        await h.settle()
        assert h.failure(third) == "queue_full"
        assert "enqueue" not in h.state()["valid_commands"]
    finally:
        await h.stop()


async def test_a_failed_shot_fails_its_dependents_and_the_session_goes_on(
    harness: Harness,
) -> None:
    _studio(harness).fail["storm"] = RuntimeError("cannot draw a storm")
    await harness.start()
    harness.command("set_build_paused", build_paused=True)
    bad = harness.command("enqueue", prompt="storm")
    await harness.settle()
    harness.command("enqueue", prompt="after the storm", continue_from=harness.shot_id(bad))
    harness.command("enqueue", prompt="calm")
    harness.command("set_build_paused", build_paused=False)
    await harness.until(lambda: len(harness.reports) == 3)

    failed = [message["shot"] for message in harness.sent("shot_failed")]
    assert [shot["inputs"]["prompt"] for shot in failed] == ["storm", "after the storm"]
    assert failed[0]["error"] == "RuntimeError: cannot draw a storm"
    assert "depends on" in failed[1]["error"]
    errors = [report.error for report in harness.reports]
    assert errors[0] == "RuntimeError: cannot draw a storm"
    assert errors[1] is not None
    assert "depends on" in errors[1]
    assert errors[2] is None
    assert [call["prompt"] for call in _studio(harness).calls] == ["storm", "calm"]


async def test_an_error_that_means_the_model_is_gone_crashes_it(harness: Harness) -> None:
    lost = WorkerCrashed("rank 1 died")
    _studio(harness).fail["x"] = lost
    await harness.start()
    harness.command("enqueue", prompt="x")
    await harness.until(lambda: harness.app._crash is not None)
    assert harness.app._crash is lost
    assert harness.reports == []


async def test_a_result_that_is_neither_media_nor_a_file_fails_the_shot() -> None:
    class Wrong(ReactorDirectorApp):
        def generate(self, prompt: str = InputField()) -> Any:
            return {"frames": []}

    h = Harness(Wrong())
    try:
        await h.start()
        h.command("enqueue", prompt="x")
        await h.until(lambda: len(h.reports) == 1)
    finally:
        await h.stop()
    assert h.reports[0].error is not None
    assert "Output or a LocalFile" in h.reports[0].error


# -- cancel, move, reset ----------------------------------------------------------


async def test_cancel_removes_a_queued_shot(harness: Harness) -> None:
    await harness.start()
    harness.command("set_build_paused", build_paused=True)
    request = harness.command("enqueue", prompt="x")
    await harness.settle()
    harness.command("cancel", shot_id=harness.shot_id(request))
    await harness.settle()
    assert harness.state()["generation"] == []


async def test_a_shot_cancelled_while_it_builds_reports_no_step(harness: Harness) -> None:
    hold = _hold(harness)
    await harness.start()
    request = harness.command("enqueue", prompt="x")
    await harness.until(lambda: bool(_studio(harness).calls))
    harness.command("cancel", shot_id=harness.shot_id(request))
    await harness.settle()
    hold.set()
    await harness.settle()
    await harness.settle()
    assert harness.reports == []
    assert harness.state()["playout"] == []


async def test_cancel_refuses_a_built_unknown_or_needed_shot(harness: Harness) -> None:
    await harness.start()
    built = harness.command("enqueue", prompt="a")
    await harness.until(lambda: len(harness.reports) == 1)
    harness.command("set_build_paused", build_paused=True)
    needed = harness.command("enqueue", prompt="b")
    await harness.settle()
    harness.command("enqueue", prompt="c", continue_from=harness.shot_id(needed))
    refusals = [
        harness.command("cancel", shot_id=harness.shot_id(built)),
        harness.command("cancel", shot_id="no-such-shot"),
        harness.command("cancel", shot_id=harness.shot_id(needed)),
    ]
    await harness.settle()
    assert [harness.failure(r) for r in refusals] == [
        "shot_built",
        "unknown_shot",
        "shot_referenced",
    ]


async def test_move_changes_the_build_order(harness: Harness) -> None:
    await harness.start()
    harness.command("set_build_paused", build_paused=True)
    harness.command("enqueue", prompt="first")
    last = harness.command("enqueue", prompt="second")
    await harness.settle()
    harness.command("move", shot_id=harness.shot_id(last), position=0)
    harness.command("set_build_paused", build_paused=False)
    await harness.until(lambda: len(harness.reports) == 2)
    assert [call["prompt"] for call in _studio(harness).calls] == ["second", "first"]


async def test_reset_drops_every_shot_and_restores_the_defaults(harness: Harness) -> None:
    await harness.start()
    harness.command("enqueue", prompt="a")
    await harness.until(lambda: len(harness.reports) == 1)
    harness.command("set_autoplay", autoplay=False)
    harness.command("set_hold_last_frame", hold_last_frame=True)
    harness.command("set_build_paused", build_paused=True)
    harness.command("enqueue", prompt="b")
    harness.command("reset")
    await harness.settle()
    state = harness.state()
    assert (state["generation"], state["playout"], state["history"]) == ([], [], [])
    assert (state["autoplay"], state["hold_last_frame"], state["build_paused"]) == (
        False,
        False,
        False,
    )


# -- sessions and clients --------------------------------------------------------


async def test_a_build_from_a_session_that_ended_is_not_reported(harness: Harness) -> None:
    hold = _hold(harness)
    await harness.start()
    harness.command("enqueue", prompt="x")
    await harness.until(lambda: bool(_studio(harness).calls))
    harness.post(SessionEnded("s", EndReason.STOPPED))
    await harness.settle()
    hold.set()
    await harness.settle()
    await harness.settle()
    assert harness.reports == []
    # No client is left to tell; the class holds nothing for the next session.
    assert len(harness.app._generation) == len(harness.app._playout) == 0


async def test_a_joining_client_is_greeted_with_the_state(harness: Harness) -> None:
    await harness.start()
    harness.post(ClientConnected(ConnId(2002), 2))
    await harness.settle()
    greetings = [
        message
        for conn, message, request in harness.addressed
        if conn == ConnId(2002) and request is None
    ]
    assert [message.name for message in greetings] == ["director_state"]


async def test_get_state_replies_with_the_state(harness: Harness) -> None:
    await harness.start()
    request = harness.command("get_state")
    await harness.settle()
    assert harness.reply(request).name == "director_state"


async def test_the_history_keeps_a_shot_a_queued_shot_needs() -> None:
    h = Harness(SmallStudio())
    try:
        await h.start()
        first = h.command("enqueue", prompt="a")
        await h.until(lambda: len(h.reports) == 1)
        h.command("discard", shot_id=h.shot_id(first))
        h.command("set_build_paused", build_paused=True)
        h.command("enqueue", prompt="b", continue_from=h.shot_id(first))
        await h.settle()
        h.command("enqueue", prompt="c")
        await h.settle()
        assert [shot["shot_id"] for shot in h.state()["history"]] == [h.shot_id(first)]
        h.command("set_build_paused", build_paused=False)
        await h.until(lambda: len(h.reports) == 3)
        given = _studio(h).calls[1]["continue_from"]
        assert given is not None
        assert given.id == h.shot_id(first)
    finally:
        await h.stop()


async def test_valid_commands_follow_the_queues(harness: Harness) -> None:
    await harness.start()
    request = harness.command("get_state")
    await harness.settle()
    assert harness.reply(request).to_wire_format()["data"]["valid_commands"] == [
        "enqueue",
        "get_state",
        "reset",
        "set_autoplay",
        "set_build_paused",
        "set_hold_last_frame",
    ]
    harness.command("enqueue", prompt="a")
    await harness.until(lambda: len(harness.reports) == 1)
    assert {"discard", "move", "play"} <= set(harness.state()["valid_commands"])


# -- playout ----------------------------------------------------------------------


async def test_play_emits_every_frame_in_order_then_cuts_to_black(harness: Harness) -> None:
    await harness.start()
    request = harness.command("enqueue", prompt="a", frames=7)
    await harness.until(lambda: len(harness.reports) == 1)
    harness.command("play")
    await harness.until(lambda: bool(harness.sent("shot_finished")))
    await harness.until(lambda: harness.flushes == [False])

    assert _frames(harness) == [1] * 7
    assert all(chunk.n_frames <= 3 for chunk in harness.chunks)
    samples = sum(c.bundle.tracks["main_audio"].data.shape[-1] for c in harness.chunks)
    assert samples == round(7 * 48_000 / FPS)
    assert harness.sent("shot_started")[0]["shot"]["shot_id"] == harness.shot_id(request)
    state = harness.state()
    assert state["playing"] is None
    assert [shot["shot_id"] for shot in state["history"]] == [harness.shot_id(request)]
    assert state["history"][0]["status"] == "built"


async def test_play_refuses_when_nothing_is_built_or_a_shot_plays(harness: Harness) -> None:
    await harness.start()
    empty = harness.command("play")
    harness.command("enqueue", prompt="a", frames=120)
    await harness.until(lambda: len(harness.reports) == 1)
    harness.command("play")
    again = harness.command("play")
    await harness.settle()
    assert harness.failure(empty) == "no_ready_shot"
    assert harness.failure(again) == "already_playing"


async def test_stop_cuts_the_shot_to_black(harness: Harness) -> None:
    await harness.start()
    harness.command("enqueue", prompt="a", frames=120)
    await harness.until(lambda: len(harness.reports) == 1)
    harness.command("play")
    await harness.until(lambda: bool(harness.chunks))
    harness.command("stop")
    await harness.until(lambda: bool(harness.sent("shot_stopped")))
    assert harness.flushes == [False]
    assert len(_frames(harness)) < 120
    assert harness.sent("shot_finished") == []


async def test_stop_holds_the_last_frame_when_asked(harness: Harness) -> None:
    await harness.start()
    harness.command("set_hold_last_frame", hold_last_frame=True)
    harness.command("enqueue", prompt="a", frames=120)
    await harness.until(lambda: len(harness.reports) == 1)
    harness.command("play")
    await harness.until(lambda: bool(harness.chunks))
    harness.command("stop")
    await harness.until(lambda: bool(harness.sent("shot_stopped")))
    assert harness.flushes == [True]


async def test_stop_refuses_when_nothing_plays(harness: Harness) -> None:
    await harness.start()
    request = harness.command("stop")
    await harness.settle()
    assert harness.failure(request) == "not_playing"


async def test_autoplay_plays_a_continuation_straight_on(harness: Harness) -> None:
    await harness.start()
    first = harness.command("enqueue", prompt="dawn", frames=6)
    await harness.settle()
    harness.command("enqueue", prompt="river", frames=6, continue_from=harness.shot_id(first))
    await harness.until(lambda: len(harness.reports) == 2)
    harness.command("set_autoplay", autoplay=True)
    await harness.until(lambda: len(harness.sent("shot_finished")) == 2)
    await harness.until(lambda: bool(harness.flushes))

    assert _frames(harness) == [1] * 6 + [2] * 6
    assert harness.flushes == [False]


async def test_autoplay_cuts_between_shots_that_do_not_continue(harness: Harness) -> None:
    await harness.start()
    harness.command("enqueue", prompt="a", frames=3)
    harness.command("enqueue", prompt="b", frames=3)
    await harness.until(lambda: len(harness.reports) == 2)
    harness.command("set_autoplay", autoplay=True)
    await harness.until(lambda: len(harness.sent("shot_finished")) == 2)
    await harness.until(lambda: len(harness.flushes) == 2)
    assert _frames(harness) == [1] * 3 + [2] * 3


async def test_discard_keeps_a_built_shot_continuable(harness: Harness) -> None:
    await harness.start()
    first = harness.command("enqueue", prompt="a")
    await harness.until(lambda: len(harness.reports) == 1)
    harness.command("discard", shot_id=harness.shot_id(first))
    harness.command("enqueue", prompt="b", continue_from=harness.shot_id(first))
    await harness.until(lambda: len(harness.reports) == 2)
    assert harness.state()["history"][0]["shot_id"] == harness.shot_id(first)
    given = _studio(harness).calls[1]["continue_from"]
    assert given is not None
    assert int(given.last_frame[0, 0, 0]) == 1


async def test_discard_refuses_a_shot_not_built_yet(harness: Harness) -> None:
    await harness.start()
    harness.command("set_build_paused", build_paused=True)
    request = harness.command("enqueue", prompt="a")
    await harness.settle()
    refused = harness.command("discard", shot_id=harness.shot_id(request))
    await harness.settle()
    assert harness.failure(refused) == "shot_not_ready"


async def test_a_client_leaving_mid_shot_cuts_it(harness: Harness) -> None:
    await harness.start()
    harness.command("enqueue", prompt="a", frames=120)
    await harness.until(lambda: len(harness.reports) == 1)
    harness.command("play")
    await harness.until(lambda: bool(harness.chunks))
    from reactor_runtime.core.model import ClientDisconnected

    harness.post(ClientDisconnected(CLIENT, 0))
    await harness.until(lambda: bool(harness.flushes))
    await harness.settle()
    assert harness.flushes == [False]
    assert harness.app._playing is None


# -- files and async generate ---------------------------------------------------


class Vendor(ReactorDirectorApp):
    """Hands back a finished file, as a hosted service does."""

    fps = 24

    def __init__(self, folder: Path) -> None:
        super().__init__()
        self.folder = folder
        self.made: list[Path] = []

    async def generate(self, prompt: str = InputField()) -> LocalFile:
        path = encode_mp4(
            self.folder / f"{len(self.made)}.mp4", np.full((4, 16, 16, 3), 120, np.uint8), fps=24
        )
        self.made.append(path)
        return LocalFile(path)


class PlayingVendor(Vendor):
    plays = Clip


async def test_a_returned_file_is_kept_as_the_step_output_then_deleted(tmp_path: Path) -> None:
    h = Harness(Vendor(tmp_path))
    try:
        await h.start()
        h.command("enqueue", prompt="a")
        await h.until(lambda: len(h.reports) == 1)
    finally:
        await h.stop()
    report = h.reports[0]
    assert report.error is None
    assert report.bundle is None
    assert {"output.mp4", "shot.json", "last_frame.png"} <= set(report.files)
    assert _shot_json(report)["frames"] == 4
    vendor = h.app
    assert isinstance(vendor, Vendor)
    assert not vendor.made[0].exists()
    assert h.state()["playout"] == []
    assert h.state()["history"][0]["status"] == "built"


async def test_a_returned_file_plays_on_the_declared_tracks(tmp_path: Path) -> None:
    h = Harness(PlayingVendor(tmp_path))
    try:
        await h.start()
        h.command("enqueue", prompt="a")
        await h.until(lambda: len(h.reports) == 1)
        h.command("play")
        await h.until(lambda: bool(h.sent("shot_finished")))
    finally:
        await h.stop()
    frames = sum(chunk.n_frames for chunk in h.chunks)
    assert frames == 4
    vendor = h.app
    assert isinstance(vendor, Vendor)
    assert not vendor.made[0].exists()
