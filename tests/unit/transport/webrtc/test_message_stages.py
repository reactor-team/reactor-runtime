import math

from reactor_runtime.core import MessageStage
from reactor_runtime.transport.webrtc.message_stages import STAGES, MessageStageWindow
from reactor_runtime.transport.webrtc.stats import ChunkingSample


def test_added_messages_sum_per_stage() -> None:
    window = MessageStageWindow()
    window.add("send", 2.0)
    window.add("send", 4.0)
    window.add("reassembly", 0.5)

    assert window.take() == (
        MessageStage("send", total_ms=6.0, messages=2),
        MessageStage("reassembly", total_ms=0.5, messages=1),
    )


def test_a_take_starts_a_new_window() -> None:
    window = MessageStageWindow()
    window.add("send", 1.0)
    window.take()

    assert window.take() == ()


def test_what_is_no_measurement_is_left_out() -> None:
    window = MessageStageWindow()
    window.add("send", -1.0)
    window.add("send", math.nan)
    window.add("send", 1.0, messages=0)

    assert window.take() == ()


def test_stages_come_out_in_trip_order_and_an_unknown_one_last() -> None:
    window = MessageStageWindow()
    for stage in ("custom", *reversed(STAGES)):
        window.add(stage, 1.0)

    assert [s.name for s in window.take()] == [*STAGES, "custom"]


def test_the_channel_totals_fold_in_as_the_window_between_two_samples() -> None:
    window = MessageStageWindow()
    before = ChunkingSample(
        messages_sent=10,
        queue_wait_s=0.010,
        send_s=0.050,
        stall_s=0.0,
        messages_received=4,
        reassembly_s=0.100,
    )
    after = ChunkingSample(
        messages_sent=12,
        queue_wait_s=0.012,
        send_s=0.056,
        stall_s=0.004,
        messages_received=5,
        reassembly_s=0.160,
    )

    window.fold(after, before)

    stages = {s.name: s for s in window.take()}
    assert stages["queue_wait"].messages == 2
    assert math.isclose(stages["queue_wait"].total_ms, 2.0)
    assert math.isclose(stages["send"].total_ms, 6.0)
    # The pump's stopped time is spread over the messages sent in the window.
    assert stages["stall"].messages == 2
    assert math.isclose(stages["stall"].total_ms, 4.0)
    assert stages["reassembly"].messages == 1
    assert math.isclose(stages["reassembly"].total_ms, 60.0)


def test_without_a_previous_sample_or_without_chunking_nothing_folds_in() -> None:
    window = MessageStageWindow()
    sample = ChunkingSample(messages_sent=3, send_s=0.01, messages_received=2, reassembly_s=0.02)

    window.fold(sample, None)
    window.fold(None, sample)

    assert window.take() == ()


def test_a_window_with_no_messages_adds_no_stage() -> None:
    window = MessageStageWindow()
    sample = ChunkingSample(messages_sent=3, send_s=0.01, messages_received=2, reassembly_s=0.02)

    window.fold(sample, sample)

    assert window.take() == ()


def test_totals_that_went_backwards_are_a_new_channel_and_add_nothing() -> None:
    window = MessageStageWindow()
    before = ChunkingSample(messages_sent=10, send_s=0.5, messages_received=10, reassembly_s=0.5)
    after = ChunkingSample(messages_sent=1, send_s=0.01, messages_received=1, reassembly_s=0.01)

    window.fold(after, before)

    assert window.take() == ()


def test_the_steps_of_handing_a_message_to_libwebrtc_fold_in_as_parts() -> None:
    window = MessageStageWindow()
    before = ChunkingSample(messages_sent=10)
    after = ChunkingSample(
        messages_sent=12,
        enqueue_s=0.001,
        queue_wait_s=0.010,
        first_check_s=0.0002,
        first_build_s=0.0004,
        first_native_s=0.002,
        send_s=0.020,
        native_send_s=0.012,
    )

    window.fold(after, before)

    stages = {s.name: s for s in window.take()}
    assert (stages["enqueue"].part_of, stages["enqueue"].messages) == ("", 2)
    assert math.isclose(stages["enqueue"].total_ms, 1.0)
    for name, ms in (("first_check", 0.2), ("first_build", 0.4), ("first_native", 2.0)):
        assert stages[name].part_of == "queue_wait"
        assert math.isclose(stages[name].total_ms, ms)
    # Every chunk's native send but the first ones, within send.
    assert stages["rest_native"].part_of == "send"
    assert math.isclose(stages["rest_native"].total_ms, 10.0)
    assert stages["queue_wait"].part_of == ""
