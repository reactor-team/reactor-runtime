"""Tests for the stages the runtime times itself, per connection."""

from __future__ import annotations

from reactor_runtime.core import ConnId, MessageStage, TransportReading
from reactor_runtime.runner.stages import RunnerStages


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
