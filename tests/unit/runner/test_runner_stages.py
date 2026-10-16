"""Tests for the stages the runtime times itself, per connection."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reactor_runtime.core import (
    ClientTrackDirection,
    ConnId,
    FrameStage,
    MessageStage,
    TrackKind,
    TransportReading,
    TransportTrackReading,
)
from reactor_runtime.interface.internal.reactor_core import CommandTimes
from reactor_runtime.runner.stages import RunnerStages, command_stages


def test_a_connections_stages_follow_its_transports_on_its_reading() -> None:
    stages = RunnerStages()
    stages.add_message(ConnId(2), "handler", 1.0)
    stages.add_message(ConnId(2), "dispatch", 3.0)
    wire = MessageStage("reassembly", total_ms=6.0, messages=3)

    [(_, reading)] = stages.merge([(ConnId(2), TransportReading(message_stages=(wire,)))])

    assert reading.message_stages == (
        wire,
        MessageStage("dispatch", total_ms=3.0, messages=1),
        MessageStage("handler", total_ms=1.0, messages=1),
    )


def test_a_transport_that_times_no_message_still_gets_the_runtimes_stages() -> None:
    stages = RunnerStages()
    stages.add_message(ConnId(2), "dispatch", 3.0)

    [(_, reading)] = stages.merge([(ConnId(2), TransportReading())])

    assert reading.message_stages == (MessageStage("dispatch", total_ms=3.0, messages=1),)


def test_stages_are_taken_once_and_go_only_to_their_connection() -> None:
    stages = RunnerStages()
    stages.add_message(ConnId(2), "dispatch", 3.0)
    readings = [(ConnId(1), TransportReading()), (ConnId(2), TransportReading())]

    [(_, other), (_, first)] = stages.merge(readings)
    [_, (_, second)] = stages.merge(readings)

    assert other.message_stages == ()
    assert first.message_stages != ()
    assert second.message_stages == ()


def test_a_connection_without_a_reading_keeps_its_stages_until_one_has() -> None:
    stages = RunnerStages()
    stages.add_message(ConnId(2), "dispatch", 3.0)

    assert stages.merge([]) == []
    [(_, reading)] = stages.merge([(ConnId(2), TransportReading())])

    assert reading.message_stages == (MessageStage("dispatch", total_ms=3.0, messages=1),)


def test_reset_drops_every_stage() -> None:
    stages = RunnerStages()
    stages.add_message(ConnId(2), "dispatch", 3.0)
    stages.reset()

    [(_, reading)] = stages.merge([(ConnId(2), TransportReading())])

    assert reading.message_stages == ()


def _video_reading(*tracks: str) -> TransportReading:
    return TransportReading(
        tracks=tuple(
            TransportTrackReading(name, TrackKind.VIDEO, ClientTrackDirection.SENDONLY)
            for name in tracks
        )
    )


def test_a_frames_stage_goes_to_its_track_after_the_transports() -> None:
    stages = RunnerStages()
    stages.add_frame(ConnId(2), "webcam", "input_wait", 2.0)
    wire = FrameStage("delivery", total_ms=8.0, frames=1)
    reading = _video_reading("webcam", "main")
    reading = replace(
        reading, tracks=(replace(reading.tracks[0], frame_stages=(wire,)), reading.tracks[1])
    )

    [(_, merged)] = stages.merge([(ConnId(2), reading)])

    assert merged.tracks[0].frame_stages == (wire, FrameStage("input_wait", 2.0, 1))
    assert merged.tracks[1].frame_stages == ()


def test_a_step_counts_once_on_each_track_of_each_connection() -> None:
    stages = RunnerStages()
    stages.add_step([ConnId(1), ConnId(2)], ["main"], {"process_input": 1.0, "generate": 4.0})

    merged = stages.merge(
        [(ConnId(1), _video_reading("main")), (ConnId(2), _video_reading("main"))]
    )

    for _, reading in merged:
        assert reading.tracks[0].frame_stages == (
            FrameStage("process_input", 1.0, 1),
            FrameStage("generate", 4.0, 1),
        )


def test_a_tracks_stages_wait_until_a_reading_carries_the_track() -> None:
    stages = RunnerStages()
    stages.add_frame(ConnId(2), "webcam", "input_wait", 2.0)

    [(_, before)] = stages.merge([(ConnId(2), _video_reading("main"))])
    [(_, after)] = stages.merge([(ConnId(2), _video_reading("main", "webcam"))])

    assert before.tracks[0].frame_stages == ()
    assert after.tracks[1].frame_stages == (FrameStage("input_wait", 2.0, 1),)


def test_a_commands_parts_name_dispatch_and_add_up_to_it() -> None:
    stages = RunnerStages()
    stages.add_command(
        ConnId(2),
        CommandTimes(
            received_at=10.0,
            submitted_at=10.030,
            held_ms=20.0,
            taken=10.040,
            started=10.045,
            finished=10.047,
        ),
    )

    [(_, reading)] = stages.merge([(ConnId(2), TransportReading())])

    by_name = {s.name: s for s in reading.message_stages}
    assert [s.name for s in reading.message_stages] == [
        "dispatch",
        "start_wait",
        "parse",
        "model_queue",
        "lock_wait",
        "handler",
    ]
    assert by_name["dispatch"].part_of == ""
    assert by_name["handler"].part_of == ""
    parts = [s for s in reading.message_stages if s.part_of == "dispatch"]
    assert sum(s.total_ms for s in parts) == pytest.approx(by_name["dispatch"].total_ms)
    assert by_name["parse"].total_ms == pytest.approx(10.0)


def test_a_command_with_no_queue_time_reports_no_parts_before_the_lock() -> None:
    names = command_stages(
        CommandTimes(
            received_at=1.0, submitted_at=None, held_ms=0.0, taken=1.0, started=1.0, finished=1.0
        )
    )

    assert set(names) == {"dispatch", "lock_wait", "handler"}
